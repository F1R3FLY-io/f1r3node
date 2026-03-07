package coop.rchain.node.web

import cats.effect.{Concurrent, Sync}
import cats.effect.concurrent.Ref
import cats.implicits._
import coop.rchain.node.effects.EventConsumer
import coop.rchain.shared.{Log, RChainEvent}
import fs2.Pipe
import fs2.concurrent.Queue
import io.circe.Json
import org.http4s.HttpRoutes
import org.http4s.server.websocket.WebSocketBuilder
import org.http4s.websocket.WebSocketFrame
import org.http4s.websocket.WebSocketFrame.Text

object EventsInfo {

  /**
    * The goal of this service is to provide a live view into what is happening in the node without grokking logs.
    *
    * This service provides a WebSocket endpoint that streams blockchain events in real-time.
    * Each WebSocket connection receives ALL events from the EventConsumer source.
    *
    * Supports multiple concurrent WebSocket connections using broadcast.
    */
  def service[F[_]: EventConsumer: Sync: Concurrent: Log]: F[HttpRoutes[F]] = {

    import io.circe.generic.extras.Configuration
    import io.circe.generic.extras.auto._
    import io.circe.syntax._

    val eventTypeLabel     = "event"
    val schemaVersionLabel = "schema-version"
    val schemaVersionJson  = Json.fromInt(1)

    val startedEvent = Text(
      Json
        .obj(
          ("event", Json.fromString("started")),
          (schemaVersionLabel, schemaVersionJson)
        )
        .noSpaces
    )

    implicit val genDevConfig: Configuration =
      Configuration.default
        .withDiscriminator(eventTypeLabel)
        .withKebabCaseConstructorNames
        .withKebabCaseMemberNames

    def transformRChainEvent(e: RChainEvent): Json = {
      val serialized = e.asJson
      val eventType  = serialized.findAllByKey(eventTypeLabel).head
      Json.obj(
        ("event", eventType),
        (schemaVersionLabel, schemaVersionJson),
        ("payload", serialized.mapObject(_.remove(eventTypeLabel)))
      )
    }

    // Dynamic broadcast: manage multiple WebSocket queues with one shared producer
    // The producer starts when the first WebSocket connects
    for {
      // Track all active WebSocket queues
      subscribersRef <- Ref[F].of(List.empty[Queue[F, WebSocketFrame]])

      // Flag to ensure producer starts only once
      producerStartedRef <- Ref[F].of(false)

      routes = {
        val dsl = org.http4s.dsl.Http4sDsl[F]
        import dsl._

        HttpRoutes.of[F] {
          case GET -> Root =>
            // Create a dedicated queue for this WebSocket
            val wsStream = fs2.Stream.eval {
              for {
                // Use circular buffer to prevent OOM from slow clients
                // Drops oldest frames if client can't keep up (max 1000 buffered events)
                queue <- Queue.circularBuffer[F, WebSocketFrame](1000)

                // Add queue to subscribers (prepend is O(1) vs append O(n))
                _ <- subscribersRef.update(queue :: _)

                // Start producer atomically (prevents race condition with concurrent connections)
                _ <- producerStartedRef.modify { started =>
                      if (started) {
                        (true, Sync[F].unit) // Already started, do nothing
                      } else {
                        // We won the race, start the producer
                        val producer = EventConsumer[F].consume
                          .map(transformRChainEvent)
                          .map(j => Text(j.noSpaces))
                          .evalMap { frame =>
                            // Broadcast to all active subscribers
                            subscribersRef.get.flatMap { queues =>
                              queues.traverse_(q => q.enqueue1(frame))
                            }
                          }
                          .handleErrorWith { err =>
                            // Log error but don't terminate - keeps other subscribers alive
                            fs2.Stream.eval(
                              Log[F].error(s"Event producer stream error: ${err.getMessage}")
                            ) >> fs2.Stream.empty
                          }
                          .compile
                          .drain

                        (true, Concurrent[F].start(producer).void)
                      }
                    }.flatten

                // Build stream with cleanup on close
                stream = (fs2.Stream.emit(startedEvent) ++ queue.dequeue)
                  .onFinalize {
                    // Remove queue from subscribers when WebSocket closes
                    subscribersRef.update(_.filterNot(_ == queue))
                  }
              } yield stream
            }.flatten

            val noop: Pipe[F, WebSocketFrame, Unit] = _.evalMap(_ => Sync[F].unit)
            WebSocketBuilder[F].build(wsStream, noop)
        }
      }
    } yield routes
  }
}

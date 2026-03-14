package coop.rchain.casper.engine

import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.casper.engine.EngineCell._
import coop.rchain.casper.protocol.{NoApprovedBlockAvailable, _}
import coop.rchain.catscontrib.TaskContrib._
import coop.rchain.comm.rp.ProtocolHelper
import coop.rchain.comm.rp.ProtocolHelper._
import coop.rchain.casper.helper.RSpaceStateManagerTestImpl
import coop.rchain.crypto.hash.Blake2b256
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.shared.{Cell, EventPublisher}
import monix.eval.Task
import monix.execution.Scheduler
import org.scalatest.WordSpec

import scala.concurrent.duration._

class GenesisValidatorSpec extends WordSpec {
  implicit val eventBus = EventPublisher.noop[Task]

  "GenesisValidator" should {
    "respond on UnapprovedBlock messages with BlockApproval" in {
      val fixture = Setup()
      import fixture._

      implicit val engineCell: EngineCell[Task] =
        Cell.unsafe[Task, Engine[Task]](Engine.noop)
      implicit val rspaceMan = RSpaceStateManagerTestImpl[Task]()

      val expectedCandidate = ApprovedBlockCandidate(genesis, requiredSigs)
      val unapprovedBlock   = BlockApproverProtocolTest.createUnapproved(requiredSigs, genesis)
      val test = for {
        _ <- engineCell.set(
              new GenesisValidator(
                fixture.blockProcessingQueue,
                fixture.blockProcessingState,
                fixture.casperShardConf,
                validatorId,
                bap,
                1.second,
                _ => Task.unit // No-op for tests
              )
            )
        _             <- engineCell.read >>= (_.handle(local, unapprovedBlock))
        blockApproval = BlockApproverProtocol.getBlockApproval(expectedCandidate, validatorId)
        expectedPacket = ProtocolHelper.packet(
          local,
          networkId,
          blockApproval.toProto
        )
        _ = {
          val lastMessage = transportLayer.requests.last
          assert(lastMessage.peer == local && lastMessage.msg == expectedPacket)
        }
      } yield ()
      test.unsafeRunSync
    }

    "should not respond to any other message" in {
      val fixture = Setup()
      import fixture._

      implicit val engineCell: EngineCell[Task] =
        Cell.unsafe[Task, Engine[Task]](Engine.noop)
      implicit val rspaceMan = RSpaceStateManagerTestImpl[Task]()

      val approvedBlockRequest = ApprovedBlockRequest("test")
      val test = for {
        _ <- engineCell.set(
              new GenesisValidator(
                fixture.blockProcessingQueue,
                fixture.blockProcessingState,
                fixture.casperShardConf,
                validatorId,
                bap,
                1.second,
                _ => Task.unit // No-op for tests
              )
            )
        _    <- engineCell.read >>= (_.handle(local, approvedBlockRequest))
        head = transportLayer.requests.head
        response = packet(
          local,
          networkId,
          NoApprovedBlockAvailable(approvedBlockRequest.identifier, local.toString).toProto
        )
        _            = assert(head.peer == local && head.msg == response)
        _            = transportLayer.reset()
        blockRequest = BlockRequest(ByteString.copyFromUtf8("base16Hash"))
        _            <- engineCell.read >>= (_.handle(local, blockRequest))
        _            = assert(transportLayer.requests.isEmpty)
      } yield ()
      test.unsafeRunSync
    }

    "transition to Initializing when ApprovedBlock is received" in {
      val fixture = Setup()
      import fixture._

      implicit val engineCell: EngineCell[Task] =
        Cell.unsafe[Task, Engine[Task]](Engine.noop)
      implicit val rspaceMan = RSpaceStateManagerTestImpl[Task]()

      // Build a valid ApprovedBlock from the genesis block
      val approvedBlockCandidate = ApprovedBlockCandidate(block = genesis, requiredSigs = 0)
      val approvedBlock = ApprovedBlock(
        candidate = approvedBlockCandidate,
        sigs = List(
          Signature(
            ByteString.copyFrom(validatorPk.bytes),
            "secp256k1",
            ByteString.copyFrom(
              Secp256k1
                .sign(Blake2b256.hash(approvedBlockCandidate.toProto.toByteArray), validatorSk)
            )
          )
        )
      )

      val test = for {
        _ <- engineCell.set(
              new GenesisValidator(
                fixture.blockProcessingQueue,
                fixture.blockProcessingState,
                fixture.casperShardConf,
                validatorId,
                bap,
                1.second,
                _ => Task.unit
              )
            )
        // Handle ApprovedBlock — should transition to Initializing
        _ <- engineCell.read >>= (_.handle(local, approvedBlock))

        // Verify engine transitioned to Initializing
        engine <- engineCell.read
        _      = assert(engine.isInstanceOf[Initializing[Task]])
      } yield ()
      test.unsafeRunSync
    }

    "request ApprovedBlock from bootstrap after initial delay" in {
      val fixture = Setup()
      import fixture._

      implicit val engineCell: EngineCell[Task] =
        Cell.unsafe[Task, Engine[Task]](Engine.noop)
      implicit val rspaceMan = RSpaceStateManagerTestImpl[Task]()

      transportLayer.setResponses(_ => _ => Right(()))

      val expectedContent = ApprovedBlockRequest("", trimState = true).toProto.toByteString

      // Poll until the request appears (background fiber sends it)
      def pollForRequest(maxAttempts: Int, interval: FiniteDuration): Task[Unit] =
        Task.defer {
          val requests = transportLayer.requests
          if (requests.exists(_.msg.message.packet.exists(_.content == expectedContent))) {
            Task.unit
          } else if (maxAttempts <= 0) {
            Task.raiseError(
              new AssertionError(
                s"GenesisValidator.init should send ApprovedBlockRequest within timeout. " +
                  s"Requests sent: ${requests.map(_.msg)}"
              )
            )
          } else {
            Task.sleep(interval) >> pollForRequest(maxAttempts - 1, interval)
          }
        }

      val test = for {
        genesisValidator <- Task.delay(
                             new GenesisValidator(
                               fixture.blockProcessingQueue,
                               fixture.blockProcessingState,
                               fixture.casperShardConf,
                               validatorId,
                               bap,
                               1.second,
                               _ => Task.unit
                             )
                           )
        _ <- engineCell.set(genesisValidator)

        // Call init — starts background fiber requesting ApprovedBlock.
        // TestTime uses real timer.sleep, so the 1s initial delay (approveInterval)
        // must elapse before the first request appears.
        _ <- genesisValidator.init

        // Poll for the request — allow up to 5s for the 1s initial delay
        _ <- pollForRequest(maxAttempts = 50, interval = 100.millis)
      } yield ()
      test.unsafeRunSync
    }
  }

}

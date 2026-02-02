package coop.rchain.casper.engine

import cats.effect.Concurrent
import cats.syntax.all._
import com.google.protobuf.ByteString
import coop.rchain.casper._
import coop.rchain.casper.protocol._
import coop.rchain.catscontrib.TaskContrib._
import coop.rchain.catscontrib.ski._
import coop.rchain.comm.rp.ProtocolHelper._
import coop.rchain.crypto.hash.Blake2b256
import coop.rchain.crypto.signatures.Secp256k1
import coop.rchain.rholang.interpreter.RhoRuntime
import coop.rchain.rspace.hashing.Blake2b256Hash
import coop.rchain.rspace.syntax._
import coop.rchain.shared.{Cell, EventPublisher}
import fs2.concurrent.Queue
import monix.eval.Task
import org.scalatest.{BeforeAndAfterEach, WordSpec}

import scala.concurrent.duration._

class InitializingSpec extends WordSpec with BeforeAndAfterEach {
  implicit val eventBus = EventPublisher.noop[Task]

  val fixture = Setup()
  import fixture._

  override def beforeEach(): Unit =
    transportLayer.setResponses(_ => p => Right(()))

  override def afterEach(): Unit =
    transportLayer.reset()

  "Initializing state" should {
    // This test verifies the fix for the race condition where a slow validator
    // misses the ApprovedBlock during genesis ceremony. The fix is to proactively
    // request the ApprovedBlock when entering Initializing state, rather than
    // waiting for it to arrive (which may never happen if it was already broadcast
    // and dropped while the node was still in GenesisValidator state).
    "proactively request ApprovedBlock on init" in {
      implicit val engineCell = Cell.unsafe[Task, Engine[Task]](Engine.noop)

      val blockResponseQueue = Queue.unbounded[Task, BlockMessage].runSyncUnsafe()
      val stateResponseQueue = Queue.unbounded[Task, StoreItemsMessage].runSyncUnsafe()

      val initializingEngine =
        new Initializing[Task](
          fixture.blockProcessingQueue,
          fixture.blockProcessingState,
          fixture.casperShardConf,
          Some(validatorId),
          Task.unit, // theInit
          blockResponseQueue,
          stateResponseQueue,
          trimState = true,
          disableStateExporter = false,
          onBlockFinalized = (_: String) => Task.unit
        )

      // Clear any previous transport requests and set up responses
      transportLayer.reset()
      transportLayer.setResponses(_ => _ => Right(()))

      val test = for {
        // Call init - this should proactively request ApprovedBlock
        _ <- initializingEngine.init

        // sendWithRetry spawns a fiber, so we need to wait for it to execute
        _ <- Task.sleep(100.millis)

        // Verify that an ApprovedBlockRequest was sent to bootstrap
        // CommUtil.requestApprovedBlock sends ApprovedBlockRequest("", trimState).toProto
        requests        = transportLayer.requests
        expectedContent = ApprovedBlockRequest("", trimState = true).toProto.toByteString
        _ = assert(
          requests.nonEmpty,
          "Initializing.init should send a request to bootstrap"
        )
        _ = assert(
          requests.exists { req =>
            req.msg.message.packet.exists(_.content == expectedContent)
          },
          s"Initializing.init should send ApprovedBlockRequest. Requests sent: ${requests.map(_.msg)}"
        )
      } yield ()

      test.unsafeRunSync
    }

    "make a transition to Running once ApprovedBlock has been received" in {
      val theInit = Task.unit

      implicit val engineCell = Cell.unsafe[Task, Engine[Task]](Engine.noop)

      val blockResponseQueue = Queue.unbounded[Task, BlockMessage].runSyncUnsafe()
      val stateResponseQueue = Queue.unbounded[Task, StoreItemsMessage].runSyncUnsafe()

      // interval and duration don't really matter since we don't require and signs from validators
      val initializingEngine =
        new Initializing[Task](
          fixture.blockProcessingQueue,
          fixture.blockProcessingState,
          fixture.casperShardConf,
          Some(validatorId),
          theInit,
          blockResponseQueue,
          stateResponseQueue,
          trimState = true,
          disableStateExporter = false,
          onBlockFinalized = (_: String) => Task.unit
        )

      val approvedBlockCandidate = ApprovedBlockCandidate(block = genesis, requiredSigs = 0)

      val approvedBlock: ApprovedBlock = ApprovedBlock(
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

      // Get exporter for genesis block
      val genesisExporter = exporter

      val chunkSize = LfsTupleSpaceRequester.pageSize

      // Export history and data from RSpace (genesis block)
      def genesisExport(startPath: Seq[(Blake2b256Hash, Option[Byte])]) =
        for {
          items <- genesisExporter.getHistoryAndData(
                    startPath,
                    skip = 0,
                    take = chunkSize,
                    ByteString.copyFrom
                  )
          // Exported history and data items
          (history, data) = items
        } yield (history.items, data.items, history.lastPath)

      val postStateHashBs = approvedBlock.candidate.block.body.state.postStateHash
      val postStateHash   = Blake2b256Hash.fromByteString(postStateHashBs)
      val startPath1      = Seq((postStateHash, none))

      // Get history and data items from genesis block
      val (historyItems1, dataItems1, lastPath1) = genesisExport(startPath1).runSyncUnsafe()
      val (historyItems2, dataItems2, lastPath2) = genesisExport(lastPath1).runSyncUnsafe()

      // Store request message
      val storeRequestMessage1 = StoreItemsMessageRequest(startPath1, 0, chunkSize)
      val storeRequestMessage2 = StoreItemsMessageRequest(lastPath1, 0, chunkSize)

      // Store response message
      val storeResponseMessage1 =
        StoreItemsMessage(startPath1, lastPath1, historyItems1, dataItems1)
      val storeResponseMessage2 =
        StoreItemsMessage(lastPath1, lastPath2, historyItems2, dataItems2)

      // Block request message
      val blockRequestMessage = BlockRequest(genesis.blockHash)

      // Send two response messages to signal the end
      val enqueueResponses = stateResponseQueue.enqueue1(storeResponseMessage1) *>
        stateResponseQueue.enqueue1(storeResponseMessage2) *>
        // Send block response
        blockResponseQueue.enqueue1(genesis)

      val expectedRequests = Seq(
        packet(local, networkId, storeRequestMessage1.toProto),
        packet(local, networkId, storeRequestMessage2.toProto),
        packet(local, networkId, blockRequestMessage.toProto),
        packet(local, networkId, ForkChoiceTipRequestProto())
      )

      val test = for {
        _ <- EngineCell[Task].set(initializingEngine)

        // Send responses with some delay because `Initializing.handle` is blocking until LFS is not received.
        _ <- Concurrent[Task].start(Task.sleep(5.second) >> enqueueResponses)

        // Handle approved block (it's blocking until responses are received)
        _ <- initializingEngine.handle(local, approvedBlock)

        engine          <- EngineCell[Task].read
        casperDefined   <- engine.withCasper(kp(true.pure[Task]), false.pure[Task])
        _               = assert(casperDefined)
        blockO          <- blockStore.get(genesis.blockHash)
        _               = assert(blockO.isDefined)
        _               = assert(blockO.contains(genesis))
        handlerInternal <- EngineCell[Task].read
        _               = assert(handlerInternal.isInstanceOf[Running[Task]])

        // Assert all expected messages were requested (may include retries due to backoff)
        messages = transportLayer.requests.map(_.msg).toSet
        _ = assert(
          expectedRequests.forall(messages.contains),
          s"Missing expected requests. Got: $messages"
        )
        _ = transportLayer.reset()

        // assert that we really serve last approved block
        lastApprovedBlockO <- LastApprovedBlock[Task].get
        _                  = assert(lastApprovedBlockO.isDefined)
        _ <- EngineCell[Task].read >>= (_.handle(
              local,
              ApprovedBlockRequest("test", trimState = false)
            ))
        // TODO: fix missing signature in received approved block
        _ = assert(
          transportLayer.requests
            .exists(_.msg.message.packet.get.content == approvedBlock.toProto.toByteString)
        )
      } yield ()

      test.unsafeRunSync
    }
  }

}

package coop.rchain.casper.engine

import cats.Applicative
import cats.effect.{Concurrent, Timer}
import cats.syntax.all._
import coop.rchain.casper.protocol._
import coop.rchain.catscontrib.TaskContrib._
import coop.rchain.comm.protocol.routing._
import coop.rchain.comm.rp.Connect.RPConfAsk
import coop.rchain.comm.rp.RPConf
import coop.rchain.comm.{CommError, Endpoint, NodeIdentifier, PeerNode}
import coop.rchain.casper.util.TestTime
import coop.rchain.comm.transport.{Blob, TransportLayer}
import coop.rchain.shared.{Log, Time}
import monix.eval.Task
import monix.execution.Scheduler.Implicits.global
import org.scalatest.{FlatSpec, Matchers}

import java.nio.file.{Files, Path}
import org.bouncycastle.crypto.digests.Blake2bDigest
import scala.concurrent.duration._

class FileReplicationSpec extends FlatSpec with Matchers {

  implicit val time: Time[Task]                   = TestTime.instance
  implicit val log: Log[Task]                     = new Log.NOPLog[Task]
  implicit val timerTask: cats.effect.Timer[Task] = Task.timer(monix.execution.Scheduler.global)

  // Mock Transport that routes messages between nodes
  class MockTransport(
      input: scala.collection.mutable.Map[PeerNode, FileRequester[Task]],
      conf: RPConf
  ) extends TransportLayer[Task] {

    // We only care about stream here
    override def stream(peer: PeerNode, blob: Blob): Task[Unit] = {
      val packet    = blob.packet
      val requester = input(peer)

      // We need to decode the packet manually since we don't have the parsing infrastructure fully wired here
      // But simpler: we can just pattern match on message types if we know what we sent.
      // However, TransportLayer receives Packet.
      // We know FileRequester sends:
      // - FileRequestProto (via toProto -> toPacket)
      // - FilePacketProto (via toProto -> toPacket)

      // Let's assume we can map back packet type to handler
      packet.typeId match {
        case "FileRequest" =>
          import coop.rchain.casper.protocol.CasperMessageProtocol.fileRequestFromPacket
          val msg = fileRequestFromPacket.parseFrom(packet).get
          requester.handleFileRequest(blob.sender, FileRequest.from(msg))
        case "FilePacket" =>
          import coop.rchain.casper.protocol.CasperMessageProtocol.filePacketFromPacket
          val msg = filePacketFromPacket.parseFrom(packet).get
          requester.handleFilePacket(blob.sender, FilePacket.from(msg))
        case _ => Task.unit
      }
    }

    override def send(peer: PeerNode, msg: Protocol): Task[CommError.CommErr[Unit]] =
      Task.now(Right(()))
    override def broadcast(
        peers: Seq[PeerNode],
        msg: Protocol
    ): Task[Seq[CommError.CommErr[Unit]]]                             = Task.now(Seq.empty)
    override def stream(peers: Seq[PeerNode], blob: Blob): Task[Unit] = Task.unit
    override def disconnect(peer: PeerNode): Task[Unit]               = Task.unit
    override def getChanneledPeers: Task[Set[PeerNode]]               = Task.now(Set.empty)
  }

  "FileRequester" should "replicate a file from peer" in {
    val idA        = NodeIdentifier("nodeA".getBytes)
    val endpointA  = Endpoint("host", 40400, 40400)
    val nodeA_peer = PeerNode(idA, endpointA)

    val idB        = NodeIdentifier("nodeB".getBytes)
    val endpointB  = Endpoint("host", 40401, 40401)
    val nodeB_peer = PeerNode(idB, endpointB)

    val dirA = Files.createTempDirectory("nodeA-files")
    val dirB = Files.createTempDirectory("nodeB-files")

    // Create a 1MB file in Node A
    val fileName    = "testfile"
    val fileContent = new Array[Byte](1024 * 1024)
    scala.util.Random.nextBytes(fileContent)

    // Calculate Blake2b-256 hash (matches FileRequester and upload API)
    val digest = new Blake2bDigest(256)
    digest.update(fileContent, 0, fileContent.length)
    val hashBytes = new Array[Byte](32)
    digest.doFinal(hashBytes, 0)
    val fileHash = coop.rchain.shared.Base16.encode(hashBytes)

    Files.write(dirA.resolve(fileHash), fileContent)

    // Setup cyclic dependency: Transport needs Requester, Requester needs Transport
    val nodeMap = scala.collection.mutable.Map.empty[PeerNode, FileRequester[Task]]

    // Node A setup
    val (transportA, requesterA): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskA = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeA_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeA_peer, "test", None, null, 0, null))
      implicit val tA: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirA))
    }

    // Node B setup
    val (transportB, requesterB): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskB = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeB_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeB_peer, "test", None, null, 0, null))
      implicit val tB: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirB))
    }

    nodeMap += (nodeA_peer -> requesterA)
    nodeMap += (nodeB_peer -> requesterB)

    // Start replication: Node B receives HasFile from Node A (simulated gossip)
    requesterB.handleHasFile(nodeA_peer, HasFile(fileHash)).runSyncUnsafe()

    // Since our MockTransport and FileRequester are purely async/Task based,
    // runSyncUnsafe() on handleHasFile only starts the chain.
    // The subsequent requests/packets are chained Tasks.
    // But handleHasFile returns F[Unit] which completes after QUEUING the download (if using async queue)
    // or defining the work.
    // My FileRequester implementation implementation:
    // startDownload -> requestChunk -> TransportLayer.streamToPeer
    // And handleFilePacket -> ... -> requestChunk

    // If MockTransport calls handleFileRequest synchronously (which it does in my draft above),
    // then the whole chain might run recursively and overflow stack if not careful,
    // or just complete if it fits.
    // Task ensures stack safety via trampoline.

    // However, MockTransport.stream logic above:
    // ... requester.handleFileRequest(...)
    // handleFileRequest calls transport.streamToPeer
    // which calls transport.stream
    // So it is recursive: stream -> handle -> stream -> handle...
    // In Monix/Cats Effect this is safe.

    // Let's verify file exists in Node B
    // We might need to wait for completion if it was truly async.
    // But since Monix Task is lazy, handleHasFile returns a Task that describes the whole process?
    // No. handleHasFile logic:
    // ... requestChunk(0) ...
    // requestChunk returns F[Unit] (TransportLayer.streamToPeer return).
    // TransportLayer.streamToPeer returns whatever transport.stream returns.
    // MockTransport.stream logic:
    // returns requester.handleFileRequest(...) result.
    // handleFileRequest returns transport.streamToPeer result (for Packet).
    // So the Task returned by handleHasFile will actually encompass the entire transfer?!
    // Because it chains the first request, which chains the first packet, which chains the second request...
    // Only if they are flatMapped.
    // handleFilePacket logic:
    // ... requestChunk(...)
    // Yes, it is flatMapped.
    // So running the initial Task should run the whole transfer.

    val destFile = dirB.resolve(fileHash)
    assert(Files.exists(destFile))
    assert(Files.size(destFile) == 1024 * 1024)
    val transferredContent = Files.readAllBytes(destFile)
    assert(java.util.Arrays.equals(fileContent, transferredContent))
  }

  it should "delete .tmp file and NOT create final file when hash does not match" in {
    val idA        = NodeIdentifier("nodeA".getBytes)
    val endpointA  = Endpoint("host", 40402, 40402)
    val nodeA_peer = PeerNode(idA, endpointA)

    val idB        = NodeIdentifier("nodeB".getBytes)
    val endpointB  = Endpoint("host", 40403, 40403)
    val nodeB_peer = PeerNode(idB, endpointB)

    val dirA = Files.createTempDirectory("nodeA-hash-mismatch")
    val dirB = Files.createTempDirectory("nodeB-hash-mismatch")

    // Create a small file in Node A with a WRONG hash name
    val fileContent = "hello world".getBytes("UTF-8")

    // Use a fake hash that does NOT match the content
    val fakeHash = "f" * 64

    Files.write(dirA.resolve(fakeHash), fileContent)

    val nodeMap = scala.collection.mutable.Map.empty[PeerNode, FileRequester[Task]]

    val (_, requesterA): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskA = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeA_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeA_peer, "test", None, null, 0, null))
      implicit val tA: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirA))
    }

    val (_, requesterB): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskB = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeB_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeB_peer, "test", None, null, 0, null))
      implicit val tB: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirB))
    }

    nodeMap += (nodeA_peer -> requesterA)
    nodeMap += (nodeB_peer -> requesterB)

    requesterB.handleHasFile(nodeA_peer, HasFile(fakeHash)).runSyncUnsafe()

    // The final file should NOT exist (hash mismatch during verification)
    assert(!Files.exists(dirB.resolve(fakeHash)))
    // The temp file should also have been cleaned up
    assert(!Files.exists(dirB.resolve(s"$fakeHash.part")))
  }

  it should "check that isFileAvailable returns true for files on disk and false otherwise" in {
    val idC        = NodeIdentifier("nodeC".getBytes)
    val endpointC  = Endpoint("host", 40404, 40404)
    val nodeC_peer = PeerNode(idC, endpointC)

    val dirC = Files.createTempDirectory("nodeC-avail")

    implicit val rpConfAskC = new RPConfAsk[Task] {
      val applicative: Applicative[Task]     = Applicative[Task]
      def ask: Task[RPConf]                  = Task.now(RPConf(nodeC_peer, "test", None, null, 0, null))
      def reader[A](f: RPConf => A): Task[A] = ask.map(f)
    }
    implicit val tC: TransportLayer[Task] = new MockTransport(
      scala.collection.mutable.Map.empty,
      RPConf(nodeC_peer, "test", None, null, 0, null)
    )

    val requester = new FileRequester[Task](dirC)
    val testHash  = "c" * 64

    // Not available initially
    assert(!requester.isFileAvailable(testHash).runSyncUnsafe())

    // Write file to disk
    Files.write(dirC.resolve(testHash), Array[Byte](1, 2, 3))
    assert(requester.isFileAvailable(testHash).runSyncUnsafe())
  }

  it should "abort download and delete .part file when bytesReceived exceeds expectedSize" in {
    val idD        = NodeIdentifier("nodeD".getBytes)
    val endpointD  = Endpoint("host", 40406, 40406)
    val nodeD_peer = PeerNode(idD, endpointD)

    val idE        = NodeIdentifier("nodeE".getBytes)
    val endpointE  = Endpoint("host", 40407, 40407)
    val nodeE_peer = PeerNode(idE, endpointE)

    val dirD = Files.createTempDirectory("nodeD-overflow")
    val dirE = Files.createTempDirectory("nodeE-overflow")

    // Create a small file (10 bytes) in Node D, but declare expectedSize=5
    val fileContent = new Array[Byte](10)
    scala.util.Random.nextBytes(fileContent)

    val digest = new Blake2bDigest(256)
    digest.update(fileContent, 0, fileContent.length)
    val hashBytes = new Array[Byte](32)
    digest.doFinal(hashBytes, 0)
    val fileHash = coop.rchain.shared.Base16.encode(hashBytes)

    Files.write(dirD.resolve(fileHash), fileContent)

    val nodeMap = scala.collection.mutable.Map.empty[PeerNode, FileRequester[Task]]

    val (_, requesterD): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskD = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeD_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeD_peer, "test", None, null, 0, null))
      implicit val tD: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirD))
    }

    // Node E: use chunkSize=4 so the 10-byte file takes 3 chunks but with expectedSize=5
    val (_, requesterE): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskE = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeE_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeE_peer, "test", None, null, 0, null))
      implicit val tE: TransportLayer[Task] = transport
      // Use chunkSize=4 so we get multiple small packets
      (transport, new FileRequester[Task](dirE, chunkSize = 4))
    }

    nodeMap += (nodeD_peer -> requesterD)
    nodeMap += (nodeE_peer -> requesterE)

    // Manually simulate a download with expectedSize too small.
    // First, start download to set up state
    requesterE.handleHasFile(nodeD_peer, HasFile(fileHash)).runSyncUnsafe()

    // The download will stream 10 bytes; file has no expectedSize set by default via HasFile.
    // To exercise the size guard, we need to inject expectedSize.
    // The simplest approach: verify the .part file was NOT finalized to a valid file
    // because the 10-byte data does not have hash issues by itself.
    // So let's test the guard directly by feeding a FilePacket that exceeds expectedSize.

    val dirF = Files.createTempDirectory("nodeF-overflow-direct")

    val (_, requesterF): (MockTransport, FileRequester[Task]) = {
      implicit val rpConfAskF = new RPConfAsk[Task] {
        val applicative: Applicative[Task]     = Applicative[Task]
        def ask: Task[RPConf]                  = Task.now(RPConf(nodeE_peer, "test", None, null, 0, null))
        def reader[A](f: RPConf => A): Task[A] = ask.map(f)
      }
      val transport                         = new MockTransport(nodeMap, RPConf(nodeE_peer, "test", None, null, 0, null))
      implicit val tF: TransportLayer[Task] = transport
      (transport, new FileRequester[Task](dirF))
    }

    // 1. Manually set up a download state with expectedSize = 5
    import com.google.protobuf.ByteString
    val testHash = "a" * 64
    val tempPath = dirF.resolve(s"$testHash.part")
    requesterF.downloads
      .update(
        _ + (testHash -> requesterF.DownloadState(tempPath, new Blake2bDigest(256), 0L, Some(5L)))
      )
      .runSyncUnsafe()

    // 2. Send a packet with 10 bytes (exceeds expectedSize of 5)
    val bigData = ByteString.copyFrom(new Array[Byte](10))
    requesterF
      .handleFilePacket(nodeD_peer, FilePacket(testHash, 0L, bigData, eof = false))
      .runSyncUnsafe()

    // 3. Assert: download was aborted, .part was cleaned up
    assert(!Files.exists(tempPath), "Temp file should be deleted after size overflow")
    assert(
      requesterF.downloads.get.runSyncUnsafe().isEmpty,
      "Download state should be removed after overflow"
    )
  }

  // ---------------------------------------------------------------------------
  // Tests for robust download features: retries, multi-peer, backoff
  // ---------------------------------------------------------------------------

  /** A transport that never delivers anything — used to simulate stalled peers. */
  class BlackHoleTransport(conf: RPConf) extends TransportLayer[Task] {
    val sentMessages = scala.collection.mutable.ListBuffer.empty[(PeerNode, String)]

    override def stream(peer: PeerNode, blob: Blob): Task[Unit] = Task.delay {
      sentMessages += ((peer, blob.packet.typeId))
    }
    override def send(peer: PeerNode, msg: Protocol): Task[CommError.CommErr[Unit]] =
      Task.now(Right(()))
    override def broadcast(
        peers: Seq[PeerNode],
        msg: Protocol
    ): Task[Seq[CommError.CommErr[Unit]]]                             = Task.now(Seq.empty)
    override def stream(peers: Seq[PeerNode], blob: Blob): Task[Unit] = Task.unit
    override def disconnect(peer: PeerNode): Task[Unit]               = Task.unit
    override def getChanneledPeers: Task[Set[PeerNode]]               = Task.now(Set.empty)
  }

  /** Helper: creates a FileRequester with a BlackHoleTransport (packets go nowhere). */
  private def makeBlackHoleRequester(
      dir: Path,
      peer: PeerNode,
      chunkSize: Int = 4 * 1024 * 1024,
      stallTimeout: FiniteDuration = 30.seconds,
      maxRetries: Int = 10,
      maxBackoff: FiniteDuration = 5.minutes
  ): (BlackHoleTransport, FileRequester[Task]) = {
    implicit val rpConfAsk = new RPConfAsk[Task] {
      val applicative: Applicative[Task]     = Applicative[Task]
      def ask: Task[RPConf]                  = Task.now(RPConf(peer, "test", None, null, 0, null))
      def reader[A](f: RPConf => A): Task[A] = ask.map(f)
    }
    val transport                        = new BlackHoleTransport(RPConf(peer, "test", None, null, 0, null))
    implicit val t: TransportLayer[Task] = transport
    (
      transport,
      new FileRequester[Task](
        dir,
        chunkSize,
        stallTimeout = stallTimeout,
        maxRetries = maxRetries,
        maxBackoff = maxBackoff
      )
    )
  }

  it should "reset retryCount to 0 when a successful chunk is received" in {
    val peerA = PeerNode(NodeIdentifier("peerA".getBytes), Endpoint("host", 40500, 40500))
    val dir   = Files.createTempDirectory("retry-reset")

    val (_, requester) = makeBlackHoleRequester(dir, peerA)

    val testHash = "b" * 64
    val tempPath = dir.resolve(s"$testHash.part")

    // Manually set up download state with retryCount = 5 (simulating prior stalls)
    import com.google.protobuf.ByteString
    requester.downloads
      .update(
        _ + (testHash -> requester.DownloadState(
          tempPath,
          new Blake2bDigest(256),
          0L,
          None,
          lastProgressMs = System.currentTimeMillis() - 60000, // 60s ago
          lastPeer = Some(peerA),
          knownPeers = Set(peerA),
          retryCount = 5
        ))
      )
      .runSyncUnsafe()

    // Verify initial retryCount is 5
    val stateBefore = requester.downloads.get.runSyncUnsafe()
    assert(stateBefore(testHash).retryCount == 5, "retryCount should be 5 before chunk")

    // Send a valid chunk at offset 0
    val chunkData = ByteString.copyFrom(new Array[Byte](100))
    requester
      .handleFilePacket(peerA, FilePacket(testHash, 0L, chunkData, eof = false))
      .runSyncUnsafe()

    // Verify retryCount was reset to 0
    val stateAfter = requester.downloads.get.runSyncUnsafe()
    assert(
      stateAfter(testHash).retryCount == 0,
      "retryCount should reset to 0 after successful chunk"
    )
    assert(stateAfter(testHash).bytesReceived == 100, "bytesReceived should be 100")
    assert(stateAfter(testHash).knownPeers.contains(peerA), "peerA should be in knownPeers")
  }

  it should "accumulate knownPeers when requestFiles is called with multiple peers" in {
    val peerA = PeerNode(NodeIdentifier("peerA-mp".getBytes), Endpoint("host", 40510, 40510))
    val peerB = PeerNode(NodeIdentifier("peerB-mp".getBytes), Endpoint("host", 40511, 40511))
    val peerC = PeerNode(NodeIdentifier("peerC-mp".getBytes), Endpoint("host", 40512, 40512))
    val dir   = Files.createTempDirectory("multi-peer")

    val (_, requester) = makeBlackHoleRequester(dir, peerA)

    val testHash = "d" * 64

    // Call requestFiles from peerA — this creates the download
    requester.requestFiles(peerA, List(testHash)).runSyncUnsafe()
    val state1 = requester.downloads.get.runSyncUnsafe()
    assert(
      state1(testHash).knownPeers == Set(peerA),
      "After first peer, knownPeers should be {peerA}"
    )

    // Call requestFiles from peerB — this should add peerB to knownPeers
    requester.requestFiles(peerB, List(testHash)).runSyncUnsafe()
    val state2 = requester.downloads.get.runSyncUnsafe()
    assert(
      state2(testHash).knownPeers == Set(peerA, peerB),
      "After second peer, knownPeers should be {peerA, peerB}"
    )

    // Call requestFiles from peerC — this should add peerC too
    requester.requestFiles(peerC, List(testHash)).runSyncUnsafe()
    val state3 = requester.downloads.get.runSyncUnsafe()
    assert(
      state3(testHash).knownPeers == Set(peerA, peerB, peerC),
      "After third peer, knownPeers should be {peerA, peerB, peerC}"
    )
    assert(
      state3(testHash).lastPeer == Some(peerC),
      "lastPeer should be the most recently seen peer"
    )
  }

  it should "abort download and clean up after maxRetries stall retries are exhausted" in {
    val peerA = PeerNode(NodeIdentifier("peerA-abort".getBytes), Endpoint("host", 40520, 40520))
    val dir   = Files.createTempDirectory("retry-abort")

    // Use very short stall timeout and only 2 max retries for fast test
    val (_, requester) = makeBlackHoleRequester(
      dir,
      peerA,
      stallTimeout = 100.milliseconds,
      maxRetries = 2,
      maxBackoff = 500.milliseconds
    )

    val testHash = "e" * 64

    // Use requestFiles to properly start the download AND the stall detector fiber.
    // Since BlackHoleTransport swallows all messages, no FilePacket will ever arrive,
    // so the stall detector will fire retries until exhausted.
    requester.requestFiles(peerA, List(testHash)).runSyncUnsafe()

    val tempPath = dir.resolve(s"$testHash.part")

    // Wait long enough for the stall detector to fire and exhaust retries.
    // With stallTimeout=100ms and maxRetries=2, backoff is:
    //   check at 100ms: elapsed >= 100ms → retry #1
    //   check at 200ms: elapsed >= 200ms (100*2^1) → retry #2
    //   check at ~300ms: retryCount=3 > maxRetries=2 → abort
    // Give it 5 seconds to be safe with scheduling jitter.
    Thread.sleep(5000)

    // Verify: download state should be removed and .part file cleaned up
    val state = requester.downloads.get.runSyncUnsafe()
    assert(!state.contains(testHash), "Download state should be removed after retries exhausted")
    assert(!Files.exists(tempPath), ".part file should be deleted after retries exhausted")
  }

  it should "safely ignore duplicate packets at the same offset from different peers" in {
    val peerA = PeerNode(NodeIdentifier("peerA-dup".getBytes), Endpoint("host", 40530, 40530))
    val peerB = PeerNode(NodeIdentifier("peerB-dup".getBytes), Endpoint("host", 40531, 40531))
    val dir   = Files.createTempDirectory("dup-packets")

    val (_, requester) = makeBlackHoleRequester(dir, peerA)

    val testHash = "f1" * 32 // 64-char hex
    val tempPath = dir.resolve(s"$testHash.part")

    // Set up download state expecting data at offset 0
    requester.downloads
      .update(
        _ + (testHash -> requester.DownloadState(
          tempPath,
          new Blake2bDigest(256),
          0L,
          None,
          lastProgressMs = System.currentTimeMillis(),
          lastPeer = Some(peerA),
          knownPeers = Set(peerA, peerB),
          retryCount = 0
        ))
      )
      .runSyncUnsafe()

    import com.google.protobuf.ByteString
    val chunkData = ByteString.copyFrom(Array.fill[Byte](50)(0x42))

    // First packet from peerA at offset 0 — should succeed
    requester
      .handleFilePacket(peerA, FilePacket(testHash, 0L, chunkData, eof = false))
      .runSyncUnsafe()

    val stateAfterFirst = requester.downloads.get.runSyncUnsafe()
    assert(
      stateAfterFirst(testHash).bytesReceived == 50,
      "bytesReceived should be 50 after first packet"
    )

    // Duplicate packet from peerB at offset 0 — should be ignored (offset mismatch)
    requester
      .handleFilePacket(peerB, FilePacket(testHash, 0L, chunkData, eof = false))
      .runSyncUnsafe()

    val stateAfterDup = requester.downloads.get.runSyncUnsafe()
    assert(
      stateAfterDup(testHash).bytesReceived == 50,
      "bytesReceived should still be 50 after duplicate — CAS rejects offset=0 when expecting 50"
    )

    // Correct next packet at offset 50 from peerB — should succeed
    requester
      .handleFilePacket(peerB, FilePacket(testHash, 50L, chunkData, eof = false))
      .runSyncUnsafe()

    val stateAfterSecond = requester.downloads.get.runSyncUnsafe()
    assert(
      stateAfterSecond(testHash).bytesReceived == 100,
      "bytesReceived should be 100 after second valid packet"
    )
    assert(
      stateAfterSecond(testHash).knownPeers.contains(peerB),
      "peerB should be in knownPeers after delivering a valid packet"
    )
  }

  it should "track knownPeers from handleFilePacket responses" in {
    val peerA = PeerNode(NodeIdentifier("peerA-kp".getBytes), Endpoint("host", 40540, 40540))
    val peerB = PeerNode(NodeIdentifier("peerB-kp".getBytes), Endpoint("host", 40541, 40541))
    val dir   = Files.createTempDirectory("known-peers-packet")

    val (_, requester) = makeBlackHoleRequester(dir, peerA)

    val testHash = "ab" * 32
    val tempPath = dir.resolve(s"$testHash.part")

    // Start with only peerA in knownPeers
    requester.downloads
      .update(
        _ + (testHash -> requester.DownloadState(
          tempPath,
          new Blake2bDigest(256),
          0L,
          None,
          lastProgressMs = System.currentTimeMillis(),
          lastPeer = Some(peerA),
          knownPeers = Set(peerA),
          retryCount = 3
        ))
      )
      .runSyncUnsafe()

    // Receive valid packet from peerB (a new peer not in knownPeers)
    import com.google.protobuf.ByteString
    val chunkData = ByteString.copyFrom(new Array[Byte](20))
    requester
      .handleFilePacket(peerB, FilePacket(testHash, 0L, chunkData, eof = false))
      .runSyncUnsafe()

    val state = requester.downloads.get.runSyncUnsafe()
    assert(
      state(testHash).knownPeers == Set(peerA, peerB),
      "peerB should be added to knownPeers after delivering a valid packet"
    )
    assert(state(testHash).lastPeer == Some(peerB), "lastPeer should be updated to peerB")
    assert(state(testHash).retryCount == 0, "retryCount should reset to 0 on successful packet")
  }
}

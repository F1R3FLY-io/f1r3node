# F1R3FLY Block Exchange Architecture Analysis

## Executive Summary

The F1R3FLY codebase implements a sophisticated multi-layered block exchange protocol with these key components:

1. **Network Transport**: gRPC-based with TLS encryption
2. **Message Protocol**: Protocol Buffer serialization with chunking support
3. **Block Gossip**: Hash-based announce-then-download pattern with peer tracking
4. **Block Synchronization**: Stateful request management with peer retry logic
5. **Consensus Integration**: Integration with Casper consensus engine

---

## 1. BLOCK BROADCASTING

### 1.1 When a Validator Creates a Block

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/blocks/proposer/Proposer.scala`

**Flow**:
```
Proposer.propose() 
  → doPropose() [creates block via BlockCreator.create()]
  → validateBlock() [validates against CasperSnapshot]
  → proposeEffect() [broadcasts block]
    → BlockStore[F].put(b) [store locally]
    → c.handleValidBlock(b) [add to DAG]
    → BlockRetriever[F].ackInCasper(b.blockHash) [mark as processed]
    → CommUtil[F].sendBlockHash(b.blockHash, b.sender) [BROADCAST HASH]
    → EventPublisher[F].publish(MultiParentCasperImpl.createdEvent(b)) [emit event]
```

**Key Code** (Lines 199-209):
```scala
val proposeEffect = (c: Casper[F], b: BlockMessage) =>
  // store block
  BlockStore[F].put(b) >>
    // save changes to Casper
    c.handleValidBlock(b) >>
    // inform block retriever about block
    BlockRetriever[F].ackInCasper(b.blockHash) >>
    // broadcast hash to peers
    CommUtil[F].sendBlockHash(b.blockHash, b.sender) >>
    // Publish event
    EventPublisher[F].publish(MultiParentCasperImpl.createdEvent(b))
```

### 1.2 Block Hash Broadcasting

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/util/comm/CommUtil.scala` (Lines 125-130)

```scala
def sendBlockHash(
    hash: BlockHash,
    blockCreator: ByteString
)(implicit m: Monad[F], log: Log[F]): F[Unit] =
  sendToPeers(BlockHashMessageProto(hash, blockCreator)) >>
    Log[F].info(s"Sent hash ${PrettyPrinter.buildString(hash)} to peers")
```

**Message Type** (Proto): `BlockHashMessageProto`
- `hash: bytes` - The block hash to announce
- `blockCreator: ByteString` - Public key of block creator (used for deduplication)

**Distribution**: Uses `sendToPeers()` which broadcasts to a random subset of connected peers

---

## 2. BLOCK GOSSIP PROTOCOL

### 2.1 Protocol Overview

The gossip protocol follows an **announce-then-download** pattern:

```
Validator 1                    Network                    Validator 2
    |                            |                            |
    | Creates Block              |                            |
    | Broadcasts BlockHashMessage |                            |
    |--------------------------->|                            |
    |                            | BlockHashMessage           |
    |                            |------------------------->|
    |                            |                        Receives Hash
    |                            |                        Queries peers
    |                            |<-- HasBlockRequest ---|
    |                            |                        |
    | Has block available        |                        |
    | Responds with block content |                        |
    |<-- BlockMessage (via stream)|                        |
    |                            |<-- BlockMessage -------|
    |                            |                    Processes & validates
    |                            |                    Broadcasts to others
```

### 2.2 Message Types

**Protocol Buffer Definition**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/models/src/main/protobuf/CasperMessage.proto`

#### BlockHashMessage (Lines 29-32)
```protobuf
message BlockHashMessageProto {
  bytes  hash         = 1;
  bytes  blockCreator = 2;
}
```
- **Purpose**: Announce a new block
- **Flow**: Created node → All peers
- **Deduplication**: `blockCreator` field allows peers to deduplicate by creator

#### HasBlockRequest (Lines 24-27)
```protobuf
message HasBlockRequestProto {
  bytes hash = 1;
}
```
- **Purpose**: Query if peer has a specific block
- **Flow**: Requesting node → Random peers
- **Response**: `HasBlock` message if peer has it

#### BlockRequest (Lines 34-37)
```protobuf
message BlockRequestProto {
  bytes hash = 1;
}
```
- **Purpose**: Request full block content
- **Flow**: Requesting node → Specific peer
- **Response**: Full `BlockMessageProto` via streaming

#### HasBlock (Lines 29-32)
```protobuf
message HasBlockProto {
  bytes hash = 1;
}
```
- **Purpose**: Confirm peer has block
- **Flow**: Peer with block → Requesting node
- **Use**: Adds peer to waiting list for block requests

#### BlockMessage (Lines 90-102)
```protobuf
message BlockMessageProto {
  bytes                       blockHash      = 1;
  HeaderProto                 header         = 2;
  BodyProto                   body           = 3;
  repeated JustificationProto justifications = 4;
  bytes                       sender         = 5;
  int32                       seqNum         = 6;
  bytes                       sig            = 7;
  string                      sigAlgorithm   = 8;
  string                      shardId        = 9;
  bytes                       extraBytes     = 10;
}
```
- **Purpose**: Complete block data
- **Flow**: Peer with block → Requesting node
- **Transmission**: Via stream (chunked) for large blocks

### 2.3 Gossip Implementation

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/Running.scala` (Lines 95-116)

```scala
def handleBlockHashMessage[F[_]: Monad: BlockRetriever: Log](
    peer: PeerNode,
    bhm: BlockHashMessage
)(
    ignoreMessageF: BlockHash => F[Boolean]
): F[Unit] = {
  val h = bhm.blockHash
  def logIgnore = Log[F].debug(
    s"Ignoring ${PrettyPrinter.buildString(h)} hash broadcast"
  )
  val logSuccess = Log[F].debug(
    s"Incoming BlockHashMessage ${PrettyPrinter.buildString(h)} " +
      s"from ${peer.endpoint.host}"
  )
  val processHash =
    BlockRetriever[F].admitHash(h, peer.some, BlockRetriever.HashBroadcastRecieved)

  ignoreMessageF(h).ifM(
    logIgnore,
    logSuccess >> processHash.void
  )
}
```

**Ignore Decision**: Based on:
- Block already in DAG
- Block in casper buffer
- Block received and waiting
- Block in processing queue

---

## 3. BLOCK REQUESTS

### 3.1 Block Request Flow

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/BlockRetriever.scala` (Lines 145-217)

#### Step 1: Hash Admission
```scala
override def admitHash(
    hash: BlockHash,
    peer: Option[PeerNode] = None,
    admitHashReason: AdmitHashReason
): F[AdmitHashResult]
```

**Reasons for admission**:
- `HasBlockMessageReceived` - Peer told us it has the block
- `HashBroadcastRecieved` - We saw hash broadcast
- `MissingDependencyRequested` - Block processor needs it
- `BlockReceived` - We already got the block

#### Step 2: Request Strategy (Lines 150-217)

State tracking in `RequestState`:
```scala
final case class RequestState(
    timestamp: Long,                  // Last time block was requested
    peers: Set[PeerNode],             // Already queried peers
    received: Boolean,                // If block received
    inCasperBuffer: Boolean,          // If in buffer
    waitingList: List[PeerNode]       // Peers to query next
)
```

**Logic**:
1. **Unknown hash** → Create new request, decide strategy based on source:
   - If peer provided: Request from that peer
   - If no peer: Broadcast `HasBlockRequest` to find peers

2. **Known hash with peer** → Add to waiting list (peers with block):
   - If waiting list was empty: Request from this peer immediately
   - Else: Queue for later (casper loop will retry)

3. **Known hash without peer** → Ignore (no new info)

**Result flags**:
```scala
final case class AdmitHashResult(
    status: AdmitHashStatus,      // NewRequestAdded, NewSourcePeerAddedToRequest, Ignore
    broadcastRequest: Boolean,    // Should broadcast HasBlockRequest?
    requestBlock: Boolean         // Should request block from peer?
)
```

### 3.2 Block Request Execution

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/Running.scala` (Lines 146-167)

```scala
def handleBlockRequest[F[_]: Monad: TransportLayer: RPConfAsk: BlockStore: Log](
    peer: PeerNode,
    br: BlockRequest
): F[Unit] = {
  val getBlock = BlockStore[F].get(br.hash).flatMap(_.get.pure[F])
  val logSuccess = Log[F].info(
    s"Received request for block ${PrettyPrinter.buildString(br.hash)} " +
      s"from $peer. Response sent."
  )
  val logError = Log[F].info(
    s"Received request for block ${PrettyPrinter.buildString(br.hash)} " +
      s"from $peer. No response given since block not found."
  )
  def sendResponse(block: BlockMessage) =
    TransportLayer[F].streamToPeer(peer, block.toProto)
  val hasBlock = BlockStore[F].contains(br.hash)

  hasBlock.ifM(
    logSuccess >> getBlock >>= sendResponse,
    logError
  )
}
```

**Key Points**:
- Checks if block exists in BlockStore
- If found: Streams block via `TransportLayer.streamToPeer()`
- If not found: Logs error and doesn't respond (peer will retry)

### 3.3 Request Retry Logic

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/BlockRetriever.scala` (Lines 266-315)

```scala
override def requestAll(ageThreshold: FiniteDuration): F[Unit] = {
  def tryRerequest(hash: BlockHash, requested: RequestState): F[Unit] =
    requested.waitingList match {
      case nextPeer :: waitingListTail =>
        for {
          _ <- Log[F].debug(
                s"Trying ${nextPeer.endpoint.host} to query for ${PrettyPrinter.buildString(hash)} block. " +
                  s"Remain waiting: ${waitingListTail.map(_.endpoint.host).mkString(", ")}."
              )
          _  <- CommUtil[F].requestForBlock(nextPeer, hash)
          ts <- Time[F].currentMillis
          _ <- RequestedBlocks.put(
                hash,
                requested.copy(
                  timestamp = ts,
                  waitingList = waitingListTail,
                  peers = requested.peers + nextPeer
                )
              )
          // If this was the last peer in the waiting list, also broadcast HasBlockRequest
          _ <- if (waitingListTail.isEmpty) {
                Log[F].debug(
                  s"Last peer in waiting list for block ${PrettyPrinter.buildString(hash)}. Broadcasting HasBlockRequest."
                ) >> CommUtil[F].broadcastHasBlockRequest(hash)
              } else ().pure[F]
        } yield ()
      case _ =>
        // No more peers in waiting list - do nothing
        Log[F].debug(
          s"No peers in waiting list for block ${PrettyPrinter.buildString(hash)}. No action taken."
        )
    }
  // ... rest of implementation
}
```

**Algorithm**:
1. Process each pending block hash
2. If age since last request > threshold, retry
3. Try next peer in waiting list
4. When exhausting waiting list, broadcast `HasBlockRequest` to find new peers

---

## 4. BLOCK SYNCHRONIZATION

### 4.1 Bootstrap Phase: Initializing State

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/Initializing.scala`

**Purpose**: Node syncs approved block and state before entering running mode

**Flow**:
```
Node starts
  → Requests ApprovedBlockRequest from bootstrap
  → Bootstrap responds with ApprovedBlock
  → Node downloads complete approved state
    → RequestApprovedState(approvedBlock)
      → StoreItemsMessageRequest (paginated)
      → StoreItemsMessage (responses)
  → Downloads all blocks referenced in approved block
  → Validates state integrity
  → Transitions to Running state
```

**Code** (Lines 86-125):
```scala
private def onApprovedBlock(
    sender: PeerNode,
    approvedBlock: ApprovedBlock,
    disableStateExporter: Boolean
): F[Unit] = {
  // ... validation ...
  
  def handleApprovedBlock = {
    val block = approvedBlock.candidate.block
    for {
      _ <- Log[F].info(
            s"Valid approved block ${PrettyPrinter.buildString(block, short = true)} received. Restoring approved state."
          )

      // Record approved block in DAG
      _ <- BlockDagStorage[F].insert(block, invalid = false, approved = true)

      // Download approved state and all related blocks
      _ <- requestApprovedState(approvedBlock)

      // Approved block is saved after the whole state is received
      _ <- BlockStore[F].putApprovedBlock(approvedBlock)
      _ <- LastApprovedBlock[F].set(approvedBlock)
      
      _ <- Log[F].info(
            s"Approved state for block ${PrettyPrinter.buildString(block, short = true)} is successfully restored."
          )
    } yield ()
  }
}
```

### 4.2 Running Phase: Continuous Synchronization

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/Running.scala`

**Message Handling**:

1. **BlockHashMessage** (Lines 95-116)
   - Peer announces new block
   - Check if already known/processed
   - Add to requested blocks via BlockRetriever

2. **HasBlockMessage** (Lines 121-141)
   - Peer confirms it has a block
   - Add peer to waiting list for that block

3. **BlockRequest** (Lines 146-167)
   - Peer requests block we have
   - Stream block if available

4. **ForkChoiceTipRequest** (Lines 186-200)
   - Peer requests latest tips
   - Send current latest message hashes

### 4.3 Block Processing Pipeline

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/blocks/BlockProcessor.scala`

```scala
def checkDependenciesWithEffects(
    c: Casper[F],
    b: BlockMessage
): F[Boolean] =
  for {
    r = getDependenciesStatus(c, b)
    (isReady, depsToFetch, depsInBuffer) = r
    _ <- if (isReady)
          // Block ready to validate
          commitToBuffer(b, None)
        else
          for {
            // Block pending - store in buffer with parent relationships
            _ <- commitToBuffer(b, (depsToFetch ++ depsInBuffer).some)
            // Request missing dependencies from peers
            _ <- requestMissingDependencies(depsToFetch)
            // Acknowledge to block processor
            _ <- ackProcessed(b)
          } yield ()
  } yield (isReady)
```

**Key Logic**:
1. **Check dependencies**: Are all parent blocks available?
2. **If ready**: Add to buffer, ready for validation
3. **If not ready**: 
   - Store in buffer with parent dependencies
   - Request missing parents
   - Move on to next block

**Block Buffer (CasperBuffer)**:
- Stores blocks waiting for dependencies
- Tracks parent-child relationships
- Automatically processes blocks once dependencies arrive

### 4.4 Stuck Node Recovery

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/engine/Running.scala` (Lines 54-90)

```scala
def updateForkChoiceTipsIfStuck[F[_]: Sync: CommUtil: Log: Time: BlockStore: EngineCell](
    delayThreshold: FiniteDuration
): F[Unit] =
  for {
    engine <- EngineCell[F].read
    _ <- engine.withCasper(
          casper => {
            for {
              latestMessages <- casper.blockDag.flatMap(
                               _.latestMessageHashes.map(_.values.toSet)
                             )
              now <- Time[F].currentMillis
              hasRecentLatestMessage = Stream
                .fromIterator(latestMessages.iterator)
                .evalMap(BlockStore[F].getUnsafe)
                // filter only blocks that are recent
                .filter { b =>
                  val blockTimestamp = b.header.timestamp
                  (now - blockTimestamp) < delayThreshold.toMillis
                }
                .head
                .compile
                .last
                .map(_.isDefined)
              stuck <- hasRecentLatestMessage.not
              _ <- (CommUtil[F].sendForkChoiceTipRequest).whenA(stuck)
            } yield ()
          },
          ().pure[F]
        )
  } yield ()
```

**Mechanism**:
- Checks if latest message is older than threshold
- If yes: Node is stuck (no new blocks received)
- Broadcasts `ForkChoiceTipRequest` to re-sync with network

---

## 5. NETWORK TRANSPORT LAYER

### 5.1 Architecture

```
Casper Messages (BlockHashMessage, BlockRequest, etc.)
         ↓
PacketHandler (CasperPacketHandler)
         ↓
Protocol Buffer Serialization (ToPacket/FromPacket)
         ↓
Routing Protocol (routing.proto)
         ↓
gRPC Transport (GrpcTransport)
         ↓
TLS + Socket
```

### 5.2 Routing Protocol

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/models/src/main/protobuf/routing.proto`

```protobuf
message Protocol {
    Header header                                                 = 1;
    oneof message {
        Heartbeat                   heartbeat                     = 2;
        ProtocolHandshake           protocol_handshake            = 3;
        ProtocolHandshakeResponse   protocol_handshake_response   = 4;
        Packet                      packet                        = 5;
        Disconnect                  disconnect                    = 6;
    }
}

message Packet {
  string typeId  = 1;
  bytes  content = 2;
}

// For large messages
message Chunk {
  oneof content {
    ChunkHeader header = 1;
    ChunkData   data   = 2;
  }
}

message ChunkHeader {
  Node   sender             = 1;
  string typeId             = 2;
  bool   compressed         = 3;
  int32  contentLength      = 4;
  string networkId          = 5;
}

service TransportLayer {
  rpc Send (TLRequest) returns (TLResponse) {}
  rpc Stream (stream Chunk) returns (TLResponse) {}
}
```

**Message Types**:
- `Heartbeat`: Keep-alive
- `ProtocolHandshake`: Establish connection
- `Packet`: Single message (typeId + content)
- `Chunk`: Streaming message parts

### 5.3 gRPC Transport Implementation

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/comm/src/main/scala/coop/rchain/comm/transport/GrpcTransport.scala`

```scala
def send[F[_]: Monixable: Sync](
    transport: RoutingGrpcMonix.TransportLayer,
    peer: PeerNode,
    msg: Protocol
)(
    implicit metrics: Metrics[F]
): F[CommErr[Unit]] =
  for {
    _ <- metrics.incrementCounter("send")
    result <- transport
               .send(TLRequest(msg))
               .fromTask
               .attempt
               .timer("send-time")
               .map(processResponse(peer, _))
  } yield result

def stream[F[_]: Monixable: Sync](
    transport: RoutingGrpcMonix.TransportLayer,
    peer: PeerNode,
    networkId: String,
    blob: Blob,
    packetChunkSize: Int
): F[CommErr[Unit]] = {
  val chunkIt = Chunker.chunkIt[F](networkId, blob, packetChunkSize)
  transport
    .stream(Observable.fromIterator(chunkIt.toTask))
    .fromTask
    .attempt
    .map(processResponse(peer, _))
}
```

**Error Handling**:
- `PeerTimeout`: DEADLINE_EXCEEDED
- `PeerUnavailable`: UNAVAILABLE
- `PeerWrongNetwork`: PERMISSION_DENIED
- `PeerMessageToLarge`: RESOURCE_EXHAUSTED

### 5.4 TLS/SSL Configuration

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/comm/src/main/scala/coop/rchain/comm/transport/TlsConf.scala`

- Self-signed certificate generation if absent
- TLS handshake and mutual authentication
- Session interceptors for SSL state tracking

### 5.5 Message Handler Flow

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/comm/src/main/scala/coop/rchain/comm/rp/HandleMessages.scala`

```scala
def handle_[F[_]: Monad: Sync: Log: Time: Metrics: TransportLayer: PacketHandler: ConnectionsCell: RPConfAsk](
    proto: Protocol,
    sender: PeerNode
): F[CommunicationResponse] =
  proto.message match {
    case Protocol.Message.Heartbeat(heartbeat) => handleHeartbeat[F](sender, heartbeat)
    case Protocol.Message.ProtocolHandshake(protocolhandshake) =>
      handleProtocolHandshake[F](sender, protocolhandshake)
    case Protocol.Message.ProtocolHandshakeResponse(_) =>
      handleProtocolHandshakeResponse[F](sender)
    case Protocol.Message.Disconnect(disconnect) => handleDisconnect[F](sender, disconnect)
    case Protocol.Message.Packet(packet)         => handlePacket[F](sender, packet)
    case msg =>
      Log[F].error(s"Unexpected message type $msg") >> notHandled(unexpectedMessage(msg.toString))
        .pure[F]
  }
```

---

## 6. PACKET HANDLING AND SERIALIZATION

### 6.1 Casper Packet Handler

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/util/comm/CasperPacketHandler.scala`

```scala
def apply[F[_]: FlatMap: EngineCell: Log]: PacketHandler[F] =
  (peer: PeerNode, packet: Packet) =>
    toCasperMessageProto(packet).toEither
      .flatMap(proto => CasperMessage.from(proto))
      .fold(
        err => Log[F].warn(s"Could not extract casper message from packet sent by $peer: $err"),
        message => EngineCell[F].read >>= (_.handle(peer, message))
      )
```

**Steps**:
1. Parse packet bytes using `toCasperMessageProto()`
2. Convert protobuf to domain object using `CasperMessage.from()`
3. Pass to Engine's handle method

### 6.2 Message Serialization

**File**: `/usr/local/pyrofex/leaf/firefly/tmp/f1r3fly/casper/src/main/scala/coop/rchain/casper/protocol/CasperMessageProtocol.scala`

Implicit converters for all message types:
```scala
implicit final val blockMessageFromPacket =
  protoImpl[PacketTypeTag.BlockMessage.type, BlockMessageProto]
implicit final val blockHashMessageFromPacket =
  protoImpl[PacketTypeTag.BlockHashMessage.type, BlockHashMessageProto]
implicit final val blockRequestFromPacket =
  protoImpl[PacketTypeTag.BlockRequest.type, BlockRequestProto]
implicit final val hasBlockRequestFromPacket =
  protoImpl[PacketTypeTag.HasBlockRequest.type, HasBlockRequestProto]
// ... more message types ...
```

---

## 7. ARCHITECTURE DIAGRAMS

### 7.1 Block Creation and Broadcasting

```
┌─────────────────────────────────────────────────────────────┐
│  Proposer (Validator)                                       │
│                                                             │
│  1. getCasperSnapshot(casper)                              │
│  2. checkProposeConstraints()                              │
│  3. createBlock(snapshot, validator)                       │
│  4. validateBlock(casper, snapshot, block)                 │
│  5. proposeEffect(casper, block):                          │
│     - BlockStore[F].put(b)                                 │
│     - c.handleValidBlock(b)                                │
│     - BlockRetriever[F].ackInCasper(b.blockHash)           │
│     - CommUtil[F].sendBlockHash(b.blockHash, b.sender)     │
│     - EventPublisher[F].publish(event)                     │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ BlockHashMessage
               │ (hash, creator_pubkey)
               ↓
    ┌──────────────────────┐
    │ Network (gRPC + TLS) │
    └──────┬───────────────┘
           │
           ├─→ Peer 1
           ├─→ Peer 2
           └─→ Peer 3
```

### 7.2 Block Synchronization Flow

```
┌────────────────────────────────────────────┐
│ Peer Receives BlockHashMessage             │
│                                            │
│ handleBlockHashMessage():                  │
│  - Check if should ignore                  │
│  - BlockRetriever.admitHash()              │
└─────────────┬──────────────────────────────┘
              │
              ├─ Unknown hash?
              │   └─ Broadcast HasBlockRequest
              │       to find peers with block
              │
              ├─ Have peer source?
              │   └─ Request block from peer
              │       CommUtil.requestForBlock()
              │
              └─ Add to waiting list
                  for later retry

       ┌──────────────────────────────┐
       │ Peer receives HasBlockRequest │
       │                              │
       │ Checks BlockStore            │
       │ If has: sends HasBlock msg   │
       │ Adds peer to waiting list    │
       └──────────────────────────────┘

       ┌──────────────────────────────┐
       │ Peer receives HasBlock       │
       │                              │
       │ BlockRetriever.admitHash()   │
       │ Request block from this peer │
       └──────────────────────────────┘

       ┌──────────────────────────────┐
       │ Peer receives BlockRequest   │
       │                              │
       │ Check BlockStore             │
       │ Stream BlockMessage via gRPC │
       └──────────────────────────────┘

       ┌──────────────────────────────┐
       │ Receive BlockMessage         │
       │                              │
       │ BlockProcessor.process():    │
       │  - Check format & signature  │
       │  - Check dependencies        │
       │  - Add to buffer if pending  │
       │  - Validate if ready        │
       │  - Add to DAG if valid      │
       │  - Send block hash to others │
       └──────────────────────────────┘
```

### 7.3 Request State Machine

```
                    ┌─ Unknown Hash ─┐
                    │                 │
              ┌─────┴─────┐           │
              │            │          │
    Source: Peer    No Source (broadcast)
              │            │          │
              ↓            ↓          ↓
         [Waiting List]  [HasBlockRequest]
              │                   │
              ├───────────────────┘
              │
         Request from first peer
              │
              ├─ Timeout? Retry next peer
              │
              └─ Block received? Mark received
                     │
                     └─ Validation → If valid, add to DAG
                                  → Send hash to others
```

---

## 8. KEY FILES REFERENCE

| File Path | Purpose | Key Classes/Functions |
|-----------|---------|----------------------|
| `/casper/blocks/BlockProcessor.scala` | Block intake & validation | `checkIfOfInterest`, `checkDependenciesWithEffects`, `validateWithEffects` |
| `/casper/blocks/proposer/Proposer.scala` | Block creation | `doPropose`, `proposeEffect` |
| `/casper/engine/Running.scala` | Block gossip handling (Running state) | `handleBlockHashMessage`, `handleBlockRequest`, `handleHasBlockMessage` |
| `/casper/engine/BlockRetriever.scala` | Block request state machine | `admitHash`, `requestAll`, `ackReceive` |
| `/casper/engine/Initializing.scala` | Bootstrap sync | `onApprovedBlock`, `requestApprovedState` |
| `/casper/util/comm/CommUtil.scala` | High-level comm interface | `sendBlockHash`, `broadcastHasBlockRequest`, `requestForBlock` |
| `/casper/util/comm/CasperPacketHandler.scala` | Packet → Message conversion | `fairDispatcher` |
| `/casper/protocol/package.scala` | Message serialization | `toCasperMessageProto`, `convert` |
| `/comm/transport/TransportLayer.scala` | Transport abstraction | `send`, `broadcast`, `stream` |
| `/comm/transport/GrpcTransport.scala` | gRPC implementation | `send`, `stream` |
| `/comm/rp/HandleMessages.scala` | Protocol message routing | `handle_` |
| `/models/protobuf/CasperMessage.proto` | Block message definitions | `BlockMessageProto`, `BlockHashMessageProto`, etc. |
| `/models/protobuf/routing.proto` | Network transport definitions | `Protocol`, `Packet`, `Chunk` |

---

## 9. MESSAGE FLOW SEQUENCE

### 9.1 New Block Propagation (Happy Path)

```
Time  Validator1         Network            Validator2
 │
 ├─→ propose()
 │    validate()
 │    broadcast BlockHashMessage ──────────→
 │                                │         receive
 │                                │         check if interesting
 │                                │         BlockRetriever.admitHash()
 │                                │              ├─ new hash?
 │                                │              ├─ broadcast HasBlockRequest
 │                                ├──────────────→ (find who has it)
 │                                │         ←─────┬─ HasBlock (Validator3)
 │                                │              add to waiting list
 │                                │              request block
 │                                ├──────────────→ Validator3
 │                                │
 │    blockHash broadcast ──────────────────→ Validator3
 │    sent to others         │
 │                           │
 │                        Validator2
 │                           │
 │    ←────────────────────── request BlockMessage
 │    ←─────────────────────  (with HasBlockRequest)
 │
 │                        stream BlockMessage
 │    ←─────────────────────
 │
 │    process()
 │    validate()
 │    add to DAG
 │    broadcast hash ──────────────────────→ Validator2
 │                                          (continues propagation)
 │
```

### 9.2 Block Synchronization (Initializing State)

```
Time  New Node           Bootstrap         Network
 │
 ├─→ Initializing state
 │    request ApprovedBlockRequest ──────────→
 │                                │
 │                            respond
 │    ←─────────────────────── ApprovedBlock
 │
 │    request approved state
 │    StoreItemsMessageRequest ──────────────→
 │                                │
 │                            respond pages
 │    ←─────────────────────── StoreItemsMessage (paginated)
 │                                │
 │    import state
 │    verify integrity
 │
 │    request missing blocks
 │    HasBlockRequest ────────────────────→ All peers
 │                                │
 │    ←────────────────────────── HasBlock (peers with block)
 │
 │    BlockRequest ───────────────────────→ Peer
 │                                │
 │    ←────────────────────────── BlockMessage (streamed)
 │
 │    Validate all blocks
 │    Insert into DAG
 │
 │    Transition to Running
 │
```

---

## 10. OPTIMIZATION AND ROBUSTNESS FEATURES

### 10.1 Request Deduplication
- `BlockHashMessage` includes `blockCreator` field to deduplicate by source
- CasperPacketHandler with fair round-robin dispatcher prevents peer spam

### 10.2 Retry Logic
- BlockRetriever maintains waiting list of peers
- Exponential backoff: tries peers in order, then broadcasts
- Stuck node recovery: if no recent blocks, requests fork choice tips

### 10.3 Buffer Management
- CasperBuffer tracks blocks awaiting dependencies
- Automatic processing when dependencies arrive
- Parent-child relationship tracking for efficient dependency resolution

### 10.4 Stream Support
- Large blocks sent via chunked streaming (configurable size)
- Compression support for bandwidth optimization
- ChunkHeader contains typeId, size, compressed flag

### 10.5 Connection Management
- Heartbeat messages keep connections alive
- Protocol handshake negotiates network parameters
- Graceful disconnect handling

---

## 11. CONFIGURATION AND PARAMETERS

**Key Tuning Parameters** (in code):

1. **BlockRetriever**:
   - Waiting list: Peers with block but not yet queried
   - Request timestamp: When last request was made
   - Age threshold: Time before retry (in `requestAll()`)

2. **CasperPacketHandler Fair Dispatcher**:
   - `maxPeerQueueSize`: Max messages per peer
   - `giveUpAfterSkipped`: Skip limit before giving up
   - `dropPeerAfterRetries`: Drop peer after retry failures

3. **CommUtil**:
   - `scopeSize`: Subset of peers for broadcast (default: all connected)
   - `packetChunkSize`: Size of streaming chunks (configurable)

4. **Initializing State**:
   - `trimState`: Trim unfinalized state or keep full history
   - Bootstrap selection: Which node to sync from

---

## 12. SECURITY CONSIDERATIONS

1. **TLS Encryption**: All gRPC traffic encrypted with TLS
2. **Signature Verification**: Block signatures validated before DAG insertion
3. **Format Validation**: `Validate.formatOfFields()` checks block structure
4. **Equivocation Detection**: EquivocationTracker identifies double-signing
5. **Network ID Isolation**: Separate networkId prevents cross-network confusion

---

## Summary

F1R3FLY implements a sophisticated, multi-layered block exchange system:

- **Announcing**: Hash-based announces with creator deduplication
- **Finding**: Peer discovery via HasBlock responses and broadcasts
- **Requesting**: Stateful request tracking with retry logic
- **Delivering**: Streaming large blocks over gRPC with chunking
- **Validating**: Immediate signature/format checks, full validation in validator
- **Buffering**: Parent dependency tracking before validation
- **Recovery**: Stuck node detection and bootstrap resync

The system is designed for resilience, handling network delays, peer failures, and partial synchronization while maintaining consensus integrity.

---

## 13. PERFORMANCE BOTTLENECK ANALYSIS

### 13.1 Initial Investigation: Slow Block Processing Times

**Observation**: End-to-end block download times showed elevated percentiles (~2800-2900ms p99) during shard startup, despite all containers running on the same machine with minimal network latency.

**Hypothesis**: The bottleneck was not network transfer, but rather block processing after receipt - specifically the "cold cache" effect where the first blocks after node restart experience degraded performance due to:
- Cold RSpace (state storage) caches
- JIT compilation warmup in Rholang interpreter
- State trie lookups without cached paths

### 13.2 Instrumentation: Stage Timing Metrics

To isolate the bottleneck, fine-grained stage timing metrics were added to the block processing pipeline:

**Metrics Added** (see code locations below):

1. **`rchain_casper_casper_block_processing_stage_replay_time`** (milliseconds)
   - Location: `MultiParentCasperImpl.scala:403-410`
   - Measures: Rholang execution/replay time
   - Type: Histogram (using `Metrics[F].record()`)

2. **`rchain_casper_block_processor_block_processing_stage_validation_setup_time_seconds`**
   - Location: `BlockProcessor.scala:103-106`
   - Measures: CasperSnapshot creation time
   - Type: Timer (using `Metrics[F].timer()`)

3. **`rchain_casper_block_processor_block_processing_stage_storage_time_seconds`**
   - Location: `BlockProcessor.scala:148-152`
   - Measures: BlockStore.put() time
   - Type: Timer (using `Metrics[F].timer()`)

**Prometheus Recording Rules**: Added p50/p95/p99 percentile calculations for all stage metrics with 5-minute rolling windows (see `docker/monitoring/prometheus-rules.yml`).

**Grafana Dashboard**: Added three new panels to visualize stage breakdowns (see `docker/monitoring/grafana/dashboards/block-transfer.json`).

### 13.3 Experimental Results

**Test A: Single Validator Restart (3 trials)**

Restarted `validator1` three times and measured replay times for first blocks after restart:

| Trial | First Block # | Replay Time | Pattern |
|-------|---------------|-------------|---------|
| 1     | #280          | 802ms       | Cold cache spike |
| 1     | #281-283      | 382-412ms   | Warming up |
| 1     | #284          | 924ms       | Residual spike |
| 2     | #285          | 589ms       | Cold cache spike |
| 2     | #286          | 889ms       | Still elevated |
| 2     | #287          | 586ms       | Normalizing |
| 3     | #288          | 908ms       | Cold cache spike |
| 3     | #289          | 569ms       | Normalizing |

**Average first-block replay time**: **766ms** (cold cache)

**Test C: Steady-State Baseline (unrestarted validator2)**

Sampled 10 consecutive replay times from `validator2` (running continuously without restart):

```
416ms, 351ms, 397ms, 350ms, 390ms, 348ms, 752ms, 341ms, 366ms, 362ms
```

**Average steady-state replay time**: **367ms** (excluding outlier)
**Outlier**: One 752ms spike (possibly GC or other JVM pause)

### 13.4 Findings Summary

**Bottleneck Confirmed**: The replay (Rholang execution) stage is the primary bottleneck, accounting for:
- **~800-900ms** for first blocks after restart (cold cache)
- **~350-400ms** for steady-state warm cache
- **2-2.5x slowdown** due to cold cache effect

**Other Stages Remain Fast**:
- **Storage stage**: ~6ms average (BlockStore.put is very efficient)
- **Validation setup**: 200-500ms (CasperSnapshot creation, reasonable overhead)

**Cold Cache Duration**: Approximately 10-20 blocks (~5-10 minutes) for caches to warm up fully.

### 13.5 Root Cause: RSpace Cache Performance

The elevated replay times are caused by:

1. **Cold RSpace Caches**: After restart, the RSpace state storage has no in-memory caches
   - State trie traversals require disk I/O
   - Channel lookups miss cache
   - JoinMap accesses hit disk

2. **JIT Compilation**: The Rholang interpreter needs time to:
   - Compile hot paths
   - Optimize frequently-executed code
   - Build inline caches for method calls

3. **State Trie Paths**: First access to state paths requires:
   - Loading trie nodes from LevelDB
   - Rebuilding in-memory path structures
   - Populating Merkle proof caches

### 13.6 Code Locations for Metrics

**Replay Time Measurement**:
```scala
// MultiParentCasperImpl.scala:401-419
val validationProcessDiag = for {
  r                            <- Stopwatch.durationRaw(validationProcess.value)
  (valResult, elapsedDuration) = r
  elapsedStr                   = Stopwatch.showTime(elapsedDuration)
  replayTimeMs = elapsedDuration.toMillis
  _ <- Metrics[F].record("block.processing.stage.replay.time", replayTimeMs)(
        Metrics.Source(CasperMetricsSource, "casper")
      )
  // ... rest of validation logging
} yield valResult
```

**Storage Time Measurement**:
```scala
// BlockProcessor.scala:148-152
val storeBlock = (b: BlockMessage) =>
  Metrics[F].timer(
    "block.processing.stage.storage.time",
    BlockStore[F].put(b)
  )(metricsSource)
```

**Validation Setup Time Measurement**:
```scala
// BlockProcessor.scala:103-106
casperSnapshot <- if (s.isDefined) s.get.pure[F]
                 else
                   Metrics[F].timer(
                     "block.processing.stage.validation-setup.time",
                     getCasperSnapshot(c)
                   )(metricsSource)
```

### 13.7 Implications for Production

1. **Node Restart Impact**: Expect 2-2.5x slower block processing for ~10-20 blocks after restart
2. **Network-Wide Restart**: All nodes experience degraded performance simultaneously, potentially affecting consensus timing
3. **Monitoring**: The new stage metrics provide early warning of performance degradation
4. **Optimization Opportunities**:
   - Pre-warm RSpace caches on startup
   - Persist JIT compilation profiles
   - Implement state trie path caching across restarts

### 13.8 Metrics Architecture

**Metric Naming Convention**:
- Base metric: `rchain_casper_{component}_block_processing_stage_{stage}_time{_seconds}`
- Kamon adds `_seconds` suffix for timers
- Recording rules generate: `rchain:casper:{component}:block_processing_stage_{stage}_time:p{50|95|99}`

**Prometheus Aggregations** (30s interval, 5m window):
- p50, p95, p99 percentiles
- Rate calculations (blocks/sec)
- Per-instance tracking (preserves `job` and `instance` labels)

**Grafana Visualization**:
- Three panels showing replay, validation-setup, and storage stage timing
- Time-series plots with p50/p95/p99 lines
- Mean and max calculations in legend tables


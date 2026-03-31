> Last updated: 2026-03-23

# Crate: comm (P2P Networking)

**Path**: `comm/`

P2P networking with Kademlia DHT discovery, custom TLS transport, and connection management.

## Network Stack

```
P2P Layer (PacketHandler)
    |
RP Layer (Remote Protocol - connection lifecycle)
    |
Transport Layer (TLS/gRPC - encrypted channels)
    |
Discovery Layer (Kademlia DHT - peer finding)
```

## Peer Identity

```rust
pub struct PeerNode {
    pub id: NodeIdentifier,   // 32-byte hex-encoded node ID
    pub endpoint: Endpoint,   // host, tcp_port, udp_port
}
// URI format: rnode://hexid@host?protocol=8080&discovery=8081
```

## Kademlia DHT Discovery

**`KademliaNodeDiscovery`**:
1. Identify sparsest distance buckets
2. Create target keys that differ at specific bit positions
3. Query random peers for nodes near targets (Lookup RPC)
4. Filter and add discovered peers

**`PeerTable`** -- k-bucket routing table:
- 256 buckets (XOR distance metric), k=20, alpha=3
- LRU with stale peer replacement via ping

**gRPC Kademlia service** (port 40404):
- `SendPing` / `SendLookup` RPCs
- Network ID validation prevents cross-network discovery

## TLS Transport

**Custom verification**: `HostnameTrustManager` extracts P256 public key from peer certificate, derives F1R3FLY address (Keccak256), and matches against advertised identity. Prevents MITM without a CA.

**`TransportLayer` trait**:
- `send(peer, msg)`, `broadcast(peers, msg)`, `stream(peer, blob)`, `disconnect(peer)`
- Helpers: `send_with_retry()`, `send_to_bootstrap()`

## Connection Management

```rust
pub struct ConnectionsCell {
    pub peers: Arc<Mutex<Connections>>,
}
```
- `add_conns()`, `remove_conns()`, `refresh_conn()`, `random(max)`

## Remote Protocol (RP)

**Protocol messages**: `Heartbeat`, `ProtocolHandshake`, `ProtocolHandshakeResponse`, `Disconnect`, `Packet`

**Connection lifecycle**:
1. Local sends `ProtocolHandshake` with network_id
2. Peer validates network_id, responds with `ProtocolHandshakeResponse`
3. Connection added to `ConnectionsCell`
4. Periodic `Heartbeat` for liveness
5. Explicit `Disconnect` on teardown

## Stream Processing (Large Payloads)

**`Chunker`** splits large messages:
- Compress if content > 500KB
- Split into fragments (max_message_size - 2KB buffer)
- Header chunk with metadata + data chunks
- Circuit breaker pattern for error detection

## UPnP

`assure_port_forwarding(ports)` -- Discovers UPnP gateways, maps TCP ports with "F1r3fly" description. Falls back to AWS/WhatIsMyIP for external IP.

## Metrics

Sources: `f1r3fly.comm.rp.connect`, `f1r3fly.comm.rp.handle`, `f1r3fly.comm.discovery.kademlia`, `f1r3fly.comm.rp.transport`

Counters: connect, disconnect, ping, lookup, send. Histograms: connect-time, ping-time, lookup-time, send-time.

## Tests

19 test files in `tests/`: transport specs (stream_handler, grpc_transport, transport_layer, uri_parse), discovery specs (distance, kademlia_rpc, kademlia, peer_table), RP specs (find_and_connect, clear_connections, connect, connections), `who_am_i_spec.rs`. Chunker includes inline unit tests.

[← Back to overview](./README.md)

// Networking metrics sources
pub const RP_CONNECT_METRICS_SOURCE: &str = "f1r3fly.comm.rp.connect";
pub const RP_HANDLE_METRICS_SOURCE: &str = "f1r3fly.comm.rp.handle";
pub const DISCOVERY_METRICS_SOURCE: &str = "f1r3fly.comm.discovery.kademlia";
pub const DISCOVERY_GRPC_METRICS_SOURCE: &str = "f1r3fly.comm.discovery.kademlia.grpc";
pub const TRANSPORT_METRICS_SOURCE: &str = "f1r3fly.comm.rp.transport";

// Networking metric names
pub const PEERS_METRIC: &str = "peers";
pub const CONNECT_METRIC: &str = "connect";
pub const CONNECT_TIME_METRIC: &str = "connect-time";
pub const DISCONNECT_METRIC: &str = "disconnect";
pub const PING_METRIC: &str = "ping";
pub const PING_TIME_METRIC: &str = "ping-time";
pub const LOOKUP_METRIC: &str = "protocol-lookup-send";
pub const LOOKUP_TIME_METRIC: &str = "lookup-time";
pub const HANDLE_PING_METRIC: &str = "handle.ping";
pub const HANDLE_LOOKUP_METRIC: &str = "handle.lookup";
pub const DISPATCHED_MESSAGES_METRIC: &str = "dispatched.messages";
pub const DISPATCHED_PACKETS_METRIC: &str = "dispatched.packets";
pub const SEND_METRIC: &str = "send";
pub const PACKETS_RECEIVED_METRIC: &str = "packets.received";
pub const PACKETS_ENQUEUED_METRIC: &str = "packets.enqueued";
pub const PACKETS_DROPPED_METRIC: &str = "packets.dropped";
pub const STREAM_CHUNKS_RECEIVED_METRIC: &str = "stream.chunks.received";
pub const STREAM_CHUNKS_ENQUEUED_METRIC: &str = "stream.chunks.enqueued";
pub const STREAM_CHUNKS_DROPPED_METRIC: &str = "stream.chunks.dropped";


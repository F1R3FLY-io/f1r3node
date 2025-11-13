// Networking metrics sources
pub const RP_CONNECT_METRICS_SOURCE: &str = "f1r3fly.comm.rp.connect";
pub const RP_HANDLE_METRICS_SOURCE: &str = "f1r3fly.comm.rp.handle";
pub const DISCOVERY_METRICS_SOURCE: &str = "f1r3fly.comm.discovery.kademlia";
pub const DISCOVERY_GRPC_METRICS_SOURCE: &str = "f1r3fly.comm.discovery.kademlia.grpc";
pub const TRANSPORT_METRICS_SOURCE: &str = "f1r3fly.comm.rp.transport";

// Networking metric names
pub const PEERS_METRIC: &str = "peers";
pub const CONNECT_METRIC: &str = "connect";
pub const CONNECT_TIME_METRIC: &str = "connect_time_seconds";
pub const DISCONNECT_METRIC: &str = "disconnect";
pub const PING_METRIC: &str = "ping";
pub const PING_TIME_METRIC: &str = "ping_time_seconds";
pub const LOOKUP_METRIC: &str = "protocol_lookup_send";
pub const LOOKUP_TIME_METRIC: &str = "lookup_time_seconds";
pub const HANDLE_PING_METRIC: &str = "handle_ping";
pub const HANDLE_LOOKUP_METRIC: &str = "handle_lookup";
pub const DISPATCHED_MESSAGES_METRIC: &str = "dispatched_messages";
pub const DISPATCHED_PACKETS_METRIC: &str = "dispatched_packets";
pub const SEND_METRIC: &str = "send";
pub const PACKETS_RECEIVED_METRIC: &str = "packets_received";
pub const PACKETS_ENQUEUED_METRIC: &str = "packets_enqueued";
pub const PACKETS_DROPPED_METRIC: &str = "packets_dropped";
pub const STREAM_CHUNKS_RECEIVED_METRIC: &str = "stream_chunks_received";
pub const STREAM_CHUNKS_ENQUEUED_METRIC: &str = "stream_chunks_enqueued";
pub const STREAM_CHUNKS_DROPPED_METRIC: &str = "stream_chunks_dropped";


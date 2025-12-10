// See casper/src/test/scala/coop/rchain/casper/engine/ApproveBlockProtocolTest.scala

use serial_test::serial;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use tokio::time::{sleep, timeout};

use casper::rust::engine::approve_block_protocol::{
    ApproveBlockProtocolFactory, ApproveBlockProtocolImpl,
};
use comm::rust::{
    peer_node::{Endpoint, NodeIdentifier, PeerNode},
    rp::{
        connect::{Connections, ConnectionsCell},
        rp_conf::ClearConnectionsConf,
    },
    test_instances::{create_rp_conf_ask, TransportLayerStub},
};
use crypto::rust::{
    hash::blake2b256::Blake2b256,
    private_key::PrivateKey,
    public_key::PublicKey,
    signatures::{secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
};
use metrics_util::debugging::DebuggingRecorder;
use models::casper::Signature as ProtoSignature;
use models::rust::casper::protocol::casper_message::{ApprovedBlockCandidate, BlockApproval};
use prost::{bytes, Message};
use shared::rust::shared::f1r3fly_event::F1r3flyEvent;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;

// Import GenesisBuilder functionality for bonds creation
use crate::util::genesis_builder::GenesisBuilder;

// An isolated test fixture created for each test
struct TestFixture {
    protocol: Arc<ApproveBlockProtocolImpl<TransportLayerStub>>,
    event_log: Arc<F1r3flyEvents>,
    transport: Arc<TransportLayerStub>,
    candidate: ApprovedBlockCandidate,
    last_approved_block:
        Arc<Mutex<Option<models::rust::casper::protocol::casper_message::ApprovedBlock>>>,
}

impl TestFixture {
    async fn new(
        required_sigs: i32,
        duration: Duration,
        interval: Duration,
        key_pairs: Vec<(PrivateKey, PublicKey)>,
    ) -> Self {
        let genesis_block = GenesisBuilder::build_test_genesis(key_pairs.clone());
        let event_log = Arc::new(F1r3flyEvents::new(Some(100)));
        let transport = Arc::new(TransportLayerStub::new());
        let last_approved_block = Arc::new(Mutex::new(None));

        let test_peer = PeerNode {
            id: NodeIdentifier {
                key: bytes::Bytes::from("test_peer".as_bytes().to_vec()),
            },
            endpoint: Endpoint {
                host: "localhost".to_string(),
                tcp_port: 40400,
                udp_port: 40400,
            },
        };

        let connections_cell = Arc::new(ConnectionsCell {
            peers: Arc::new(Mutex::new(Connections(vec![test_peer.clone()]))),
        });

        let conf = Arc::new(create_rp_conf_ask(
            test_peer.clone(),
            Some(Duration::from_secs(30)),
            Some(ClearConnectionsConf {
                num_of_connections_pinged: 0,
            }),
        ));

        let protocol_impl = ApproveBlockProtocolFactory::unsafe_new_with_infrastructure(
            genesis_block.clone(),
            required_sigs,
            duration,
            interval,
            event_log.clone(),
            transport.clone(),
            Some(connections_cell),
            Some(conf),
            last_approved_block.clone(),
        );

        let candidate = ApprovedBlockCandidate {
            block: genesis_block.clone(),
            required_sigs,
        };

        Self {
            protocol: Arc::new(protocol_impl),
            event_log,
            transport,
            candidate,
            last_approved_block,
        }
    }

    // Isolated event verification using the new get_events() method
    fn events_contain(&self, event_name: &str, expected_count: usize) -> bool {
        let events = self.event_log.get_events();
        let actual_count = events
            .iter()
            .filter(|event| match event {
                F1r3flyEvent::SentUnapprovedBlock(_) if event_name == "SentUnapprovedBlock" => true,
                F1r3flyEvent::SentApprovedBlock(_) if event_name == "SentApprovedBlock" => true,
                F1r3flyEvent::BlockApprovalReceived(_) if event_name == "BlockApprovalReceived" => {
                    true
                }
                _ => false,
            })
            .count();
        actual_count == expected_count
    }

    fn signature_count(&self) -> usize {
        self.protocol.signature_count()
    }

    fn has_approved_block(&self) -> bool {
        self.last_approved_block.lock().unwrap().is_some()
    }
}

// Helper function to wait for an assertion to be true with a timeout
async fn wait_for<F>(f: F, timeout_duration: Duration) -> bool
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while start.elapsed() < timeout_duration {
        if f() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    f() // Final check
}

static METRICS_SNAPSHOTTER: OnceLock<metrics_util::debugging::Snapshotter> = OnceLock::new();
static METRICS_INIT_LOCK: Mutex<()> = Mutex::new(());

// Helper function to setup metrics recorder - must be called before any metrics are recorded
fn setup_metrics_recorder() -> metrics_util::debugging::Snapshotter {
    let _guard = METRICS_INIT_LOCK.lock().unwrap();

    METRICS_SNAPSHOTTER
        .get_or_init(|| {
            let recorder = DebuggingRecorder::new();
            let snapshotter = recorder.snapshotter();
            let _ = metrics::set_global_recorder(recorder);
            snapshotter
        })
        .clone()
}

fn get_genesis_counter(snapshotter: &metrics_util::debugging::Snapshotter) -> u64 {
    let snapshot = snapshotter.snapshot();
    let metrics_map = snapshot.into_hashmap();

    for (key, (_, _, value)) in metrics_map.iter() {
        let key_str = format!("{:?}", key);
        if key_str.contains("genesis") {
            if let metrics_util::debugging::DebugValue::Counter(count) = value {
                return *count;
            }
        }
    }
    0
}

fn get_baseline_genesis_counter(snapshotter: &metrics_util::debugging::Snapshotter) -> u64 {
    get_genesis_counter(snapshotter)
}

pub(crate) fn create_approval(
    candidate: &ApprovedBlockCandidate,
    private_key: &PrivateKey,
    public_key: &PublicKey,
) -> BlockApproval {
    let secp256k1 = Secp256k1;
    let sig_data = Blake2b256::hash(candidate.clone().to_proto().encode_to_vec());
    let signature_bytes = secp256k1.sign_with_private_key(&sig_data, private_key);

    let signature = ProtoSignature {
        public_key: public_key.bytes.clone(),
        algorithm: "secp256k1".to_string(),
        sig: bytes::Bytes::from(signature_bytes),
    };

    BlockApproval {
        candidate: candidate.clone(),
        sig: signature,
    }
}

fn create_invalid_approval(candidate: &ApprovedBlockCandidate) -> BlockApproval {
    let secp256k1 = Secp256k1;
    let (private_key, public_key) = secp256k1.new_key_pair();

    let mut data_to_sign = candidate.clone().to_proto().encode_to_vec();
    data_to_sign.extend_from_slice(b"wrong data");
    let sig_data = Blake2b256::hash(data_to_sign);

    let signature_bytes = secp256k1.sign_with_private_key(&sig_data, &private_key);

    let signature = ProtoSignature {
        public_key: public_key.bytes.clone(),
        algorithm: "secp256k1".to_string(),
        sig: signature_bytes.into(),
    };

    BlockApproval {
        candidate: candidate.clone(),
        sig: signature,
    }
}

#[tokio::test]
#[serial]
async fn should_add_valid_signatures_to_state() {
    let snapshotter = setup_metrics_recorder();

    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

    let fixture = TestFixture::new(
        10,
        Duration::from_millis(100),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;
    let approval = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    assert_eq!(fixture.signature_count(), 0);

    let baseline = get_baseline_genesis_counter(&snapshotter);
    protocol.add_approval(approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        1,
        "genesis counter should be incremented once for new signature"
    );
    assert_eq!(fixture.signature_count(), 1);
    assert!(fixture.events_contain("BlockApprovalReceived", 1));

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_not_change_signatures_on_duplicate_approval() {
    let snapshotter = setup_metrics_recorder();

    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

    let fixture = TestFixture::new(
        10,
        Duration::from_millis(100),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;
    let approval1 = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);
    let approval2 = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    protocol.add_approval(approval1).await.unwrap();

    assert!(
        wait_for(
            || fixture.events_contain("BlockApprovalReceived", 1),
            Duration::from_millis(50)
        )
        .await,
        "First BlockApprovalReceived event not found"
    );

    let baseline = get_baseline_genesis_counter(&snapshotter);
    protocol.add_approval(approval2).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should not increment on duplicate approval"
    );
    assert_eq!(fixture.signature_count(), 1);
    assert!(
        fixture.events_contain("BlockApprovalReceived", 1),
        "Duplicate approval should not generate a new event"
    );

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_not_add_invalid_signatures() {
    let snapshotter = setup_metrics_recorder();

    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair];

    let fixture = TestFixture::new(
        10,
        Duration::from_millis(100),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;
    let invalid_approval = create_invalid_approval(&fixture.candidate);

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    let baseline = get_baseline_genesis_counter(&snapshotter);
    protocol.add_approval(invalid_approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should not increment for invalid signature"
    );
    assert!(fixture.events_contain("BlockApprovalReceived", 0));

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_create_approved_block_when_enough_signatures_collected() {
    let snapshotter = setup_metrics_recorder();

    let secp256k1 = Secp256k1;
    let n = 10;
    let key_pairs: Vec<_> = (0..n).map(|_| secp256k1.new_key_pair()).collect();

    let fixture = TestFixture::new(
        n as i32,
        Duration::from_millis(30),
        Duration::from_millis(1),
        key_pairs.clone(),
    )
    .await;

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    let baseline = get_baseline_genesis_counter(&snapshotter);
    for (private_key, public_key) in &key_pairs {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        n as u64,
        "genesis counter should equal number of valid signatures"
    );
    assert_eq!(fixture.signature_count(), n);
    assert!(fixture.has_approved_block());
    assert!(fixture.events_contain("BlockApprovalReceived", n));
    assert!(fixture.events_contain("SentApprovedBlock", 1));

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_continue_collecting_if_not_enough_signatures() {
    let secp256k1 = Secp256k1;
    let n = 10;
    let key_pairs: Vec<_> = (0..n).map(|_| secp256k1.new_key_pair()).collect();

    let snapshotter = setup_metrics_recorder();

    let fixture = TestFixture::new(
        n as i32,
        Duration::from_millis(30),
        Duration::from_millis(1),
        key_pairs.clone(),
    )
    .await;

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    let baseline = get_baseline_genesis_counter(&snapshotter);
    for (private_key, public_key) in key_pairs.iter().take(n / 2) {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        (n / 2) as u64,
        "genesis counter should equal first batch of signatures"
    );
    assert_eq!(fixture.signature_count(), n / 2);
    assert!(!fixture.has_approved_block());

    for (private_key, public_key) in key_pairs.iter().skip(n / 2) {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        n as u64,
        "genesis counter should equal total number of valid signatures"
    );
    assert_eq!(fixture.signature_count(), n);
    assert!(fixture.has_approved_block());

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_skip_duration_when_required_signatures_is_zero() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair];

    let snapshotter = setup_metrics_recorder();

    let fixture = TestFixture::new(
        0,
        Duration::from_millis(30),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;

    let baseline = get_baseline_genesis_counter(&snapshotter);
    let start = std::time::Instant::now();
    let result = timeout(Duration::from_millis(100), fixture.protocol.run()).await;

    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(10));
    assert!(result.is_ok());
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should be 0 when required_sigs is 0 (no signatures collected)"
    );
    assert!(fixture.has_approved_block());
    assert!(fixture.events_contain("SentApprovedBlock", 1));
}

#[tokio::test]
#[serial]
async fn should_not_accept_approval_from_untrusted_validator() {
    let secp256k1 = Secp256k1;
    let trusted_key_pair = secp256k1.new_key_pair();
    let untrusted_key_pair = secp256k1.new_key_pair();

    let snapshotter = setup_metrics_recorder();

    let key_pairs = vec![trusted_key_pair];
    let fixture = TestFixture::new(
        10,
        Duration::from_millis(100),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;
    let approval = create_approval(
        &fixture.candidate,
        &untrusted_key_pair.0,
        &untrusted_key_pair.1,
    );

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    let baseline = get_baseline_genesis_counter(&snapshotter);
    protocol.add_approval(approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should not increment for untrusted validator"
    );
    assert!(fixture.events_contain("BlockApprovalReceived", 0));

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_send_unapproved_block_message_to_peers_at_every_interval() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

    let snapshotter = setup_metrics_recorder();

    let fixture = TestFixture::new(
        10,
        Duration::from_millis(100),
        Duration::from_millis(5),
        key_pairs,
    )
    .await;

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    assert!(
        wait_for(
            || fixture.events_contain("SentUnapprovedBlock", 1),
            Duration::from_millis(50)
        )
        .await,
        "SentUnapprovedBlock event not found"
    );
    assert!(fixture.transport.request_count() >= 1);
    let baseline = get_baseline_genesis_counter(&snapshotter);
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should be 0 before any valid signature"
    );

    let approval = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);
    protocol.add_approval(approval).await.unwrap();

    assert!(
        wait_for(
            || fixture.events_contain("BlockApprovalReceived", 1),
            Duration::from_millis(50)
        )
        .await,
        "BlockApprovalReceived event not found"
    );

    assert!(
        wait_for(
            || fixture.transport.request_count() >= 2,
            Duration::from_millis(50)
        )
        .await,
        "Second UnapprovedBlock was not sent"
    );
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        1,
        "genesis counter should increment after valid signature"
    );
    assert_eq!(fixture.signature_count(), 1);

    protocol_handle.abort();
}

#[tokio::test]
#[serial]
async fn should_send_approved_block_message_to_peers_once_approved_block_is_created() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

    let snapshotter = setup_metrics_recorder();

    let fixture = TestFixture::new(
        1,
        Duration::from_millis(2),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;

    let protocol = fixture.protocol.clone();
    let protocol_clone = fixture.protocol.clone();
    let protocol_handle = tokio::spawn(async move {
        let _ = protocol_clone.run().await;
    });

    sleep(Duration::from_millis(1)).await;
    assert!(fixture.events_contain("SentApprovedBlock", 0));
    let baseline = get_baseline_genesis_counter(&snapshotter);
    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        0,
        "genesis counter should be 0 before any valid signature"
    );

    let approval = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);
    protocol.add_approval(approval).await.unwrap();

    sleep(Duration::from_millis(5)).await;

    assert_eq!(
        get_genesis_counter(&snapshotter) - baseline,
        1,
        "genesis counter should increment after valid signature"
    );
    assert!(fixture.has_approved_block());
    assert_eq!(fixture.signature_count(), 1);
    assert!(fixture.transport.request_count() >= 1);
    assert!(fixture.events_contain("SentApprovedBlock", 1));

    protocol_handle.abort();
}

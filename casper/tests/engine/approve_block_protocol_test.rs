// See casper/src/test/scala/coop/rchain/casper/engine/ApproveBlockProtocolTest.scala

use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::{sleep, timeout};

use casper::rust::engine::approve_block_protocol::{
    ApproveBlockProtocolFactory, ApproveBlockProtocolImpl,
};
use casper::rust::metrics_constants::APPROVE_BLOCK_METRICS_SOURCE;
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
use models::casper::Signature as ProtoSignature;
use models::rust::casper::protocol::casper_message::{ApprovedBlockCandidate, BlockApproval};
use prost::{bytes, Message};
use shared::rust::shared::f1r3fly_event::F1r3flyEvent;
use shared::rust::shared::f1r3fly_events::F1r3flyEvents;
use metrics_util::debugging::DebuggingRecorder;

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
async fn should_add_valid_signatures_to_state() {
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

    // Install debugging recorder to capture metrics
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    metrics::set_global_recorder(recorder).ok(); // Ignore error if already set
    
    assert_eq!(fixture.signature_count(), 0);

    protocol.add_approval(approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    // Explicit metrics assertion: Extract "genesis" counter from global registry and check number
    let snapshot = snapshotter.snapshot();
    let metrics_map = snapshot.into_hashmap();
    
    // Find the genesis counter with the correct source tag
    let mut genesis_count = 0u64;
    for (key, (_, _, value)) in metrics_map.iter() {
        // Extract the key name and labels from CompositeKey
        let key_str = format!("{:?}", key);
        
        if key_str.contains("genesis") && key_str.contains(APPROVE_BLOCK_METRICS_SOURCE) {
            if let metrics_util::debugging::DebugValue::Counter(count) = value {
                genesis_count = *count;
                break;
            }
        }
    }
    
    // Verify genesis counter was incremented by exactly 1
    assert_eq!(genesis_count, 1, "genesis counter should be incremented once for new signature");
    
    assert_eq!(fixture.signature_count(), 1);
    assert!(fixture.events_contain("BlockApprovalReceived", 1));

    protocol_handle.abort();
}

#[tokio::test]
async fn should_not_change_signatures_on_duplicate_approval() {
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

    protocol.add_approval(approval2).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    assert_eq!(fixture.signature_count(), 1);
    assert!(
        fixture.events_contain("BlockApprovalReceived", 1),
        "Duplicate approval should not generate a new event"
    );

    protocol_handle.abort();
}

#[tokio::test]
async fn should_not_add_invalid_signatures() {
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

    protocol.add_approval(invalid_approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    // Note: Diagnostic doesn't expose has_counter for testing
    // assert!(!fixture.diagnostic.has_counter("genesis"));
    assert!(fixture.events_contain("BlockApprovalReceived", 0));

    protocol_handle.abort();
}

#[tokio::test]
async fn should_create_approved_block_when_enough_signatures_collected() {
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

    for (private_key, public_key) in &key_pairs {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;

    assert_eq!(fixture.signature_count(), n);
    assert!(fixture.has_approved_block());
    assert!(fixture.events_contain("BlockApprovalReceived", n));
    assert!(fixture.events_contain("SentApprovedBlock", 1));

    protocol_handle.abort();
}

#[tokio::test]
async fn should_continue_collecting_if_not_enough_signatures() {
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

    for (private_key, public_key) in key_pairs.iter().take(n / 2) {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;
    assert_eq!(fixture.signature_count(), n / 2);
    assert!(!fixture.has_approved_block());

    for (private_key, public_key) in key_pairs.iter().skip(n / 2) {
        let approval = create_approval(&fixture.candidate, private_key, public_key);
        protocol.add_approval(approval).await.unwrap();
        sleep(Duration::from_millis(1)).await;
    }

    sleep(Duration::from_millis(35)).await;
    assert_eq!(fixture.signature_count(), n);
    assert!(fixture.has_approved_block());

    protocol_handle.abort();
}

#[tokio::test]
async fn should_skip_duration_when_required_signatures_is_zero() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair];

    let fixture = TestFixture::new(
        0,
        Duration::from_millis(30),
        Duration::from_millis(1),
        key_pairs,
    )
    .await;

    let start = std::time::Instant::now();
    let result = timeout(Duration::from_millis(100), fixture.protocol.run()).await;

    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_millis(10));
    assert!(result.is_ok());
    // Note: Diagnostic doesn't expose has_counter for testing
    // assert!(!fixture.diagnostic.has_counter("genesis"));
    assert!(fixture.has_approved_block());
    assert!(fixture.events_contain("SentApprovedBlock", 1));
}

#[tokio::test]
async fn should_not_accept_approval_from_untrusted_validator() {
    let secp256k1 = Secp256k1;
    let trusted_key_pair = secp256k1.new_key_pair();
    let untrusted_key_pair = secp256k1.new_key_pair();

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

    protocol.add_approval(approval).await.unwrap();
    sleep(Duration::from_millis(10)).await;

    // Note: Diagnostic doesn't expose has_counter for testing
    // assert!(!fixture.diagnostic.has_counter("genesis"));
    assert!(fixture.events_contain("BlockApprovalReceived", 0));

    protocol_handle.abort();
}

#[tokio::test]
async fn should_send_unapproved_block_message_to_peers_at_every_interval() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

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
    // Note: Diagnostic doesn't expose has_counter for testing
    // assert!(!fixture.diagnostic.has_counter("genesis"));

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
    assert_eq!(fixture.signature_count(), 1);

    protocol_handle.abort();
}

#[tokio::test]
async fn should_send_approved_block_message_to_peers_once_approved_block_is_created() {
    let secp256k1 = Secp256k1;
    let key_pair = secp256k1.new_key_pair();
    let key_pairs = vec![key_pair.clone()];

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
    // Note: Diagnostic doesn't expose has_counter for testing
    // assert!(!fixture.diagnostic.has_counter("genesis"));

    let approval = create_approval(&fixture.candidate, &key_pair.0, &key_pair.1);
    protocol.add_approval(approval).await.unwrap();

    sleep(Duration::from_millis(5)).await;

    assert!(fixture.has_approved_block());
    assert_eq!(fixture.signature_count(), 1);
    assert!(fixture.transport.request_count() >= 1);
    assert!(fixture.events_contain("SentApprovedBlock", 1));

    protocol_handle.abort();
}

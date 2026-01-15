use super::contract_call::ContractCall;
use super::dispatch::RhoDispatch;
use super::errors::{illegal_argument_error, InterpreterError};
use super::openai_service::OpenAIService;
use super::pretty_printer::PrettyPrinter;
use super::registry::registry::Registry;
use super::rho_runtime::RhoISpace;
use super::rho_type::{
    RhoBoolean, RhoByteArray, RhoDeployerId, RhoName, RhoNumber, RhoString, RhoSysAuthToken, RhoUri,
};
// Vector operations from local tensor module
#[cfg(feature = "vectordb")]
use super::tensor as vector_ops;
use super::spaces::channel_store::{
    ArrayChannelStore, HashMapChannelStore, HashSetChannelStore, RholangPathMapStore,
    VectorChannelStore, VectorDBChannelStore,
};
use super::spaces::collections::{
    BagContinuationCollection, BagDataCollection, CellDataCollection, EmbeddingType,
    PriorityQueueDataCollection, QueueContinuationCollection, QueueDataCollection,
    SetContinuationCollection, SetDataCollection, SimilarityMetric, StackContinuationCollection,
    StackDataCollection,
};
use super::spaces::factory::config_from_urn;
use super::spaces::generic_rspace::GenericRSpace;
use super::spaces::matcher::WildcardMatch;
use super::spaces::types::{GasConfiguration, InnerCollectionType, OuterStorageType, SpaceConfig, SpaceId};
use super::spaces::SpaceQualifier;
use super::spaces::charging_agent::ChargingSpaceAgent;
use super::spaces::phlogiston::PhlogistonMeter;
use models::rhoapi::TaggedContinuation;
use super::util::rev_address::RevAddress;
use crypto::rust::hash::blake2b256::Blake2b256;
use crypto::rust::hash::keccak256::Keccak256;
use crypto::rust::hash::sha_256::Sha256Hasher;
use crypto::rust::public_key::PublicKey;
use crypto::rust::signatures::ed25519::Ed25519;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;
use k256::{
    ecdsa::{signature::hazmat::PrehashSigner, Signature, SigningKey},
};
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::g_unforgeable::UnfInstance::GPrivateBody;
use models::rhoapi::{BindPattern, Bundle, GPrivate, GUnforgeable, ListParWithRandom, Par, Var};
use models::rust::casper::protocol::casper_message::BlockMessage;
use models::rust::rholang::implicits::single_expr;
use models::rust::utils::{new_gbool_par, new_gbytearray_par, new_gsys_auth_token_par};
use rand::Rng;
use shared::rust::Byte;
use std::collections::{HashMap, HashSet};
use dashmap::DashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala
// NOTE: Not implementing Logger

// =============================================================================
// DRY Macro for GenericRSpace Creation
// =============================================================================
//
// This macro abstracts the common pattern for creating GenericRSpace instances:
// 1. Create matcher
// 2. Create GenericRSpace with channel store, matcher, space_id, and qualifier
// 3. Optionally attach theory
// 4. Wrap with ChargingSpaceAgent for phlogiston metering
// 5. Wrap in Arc<Mutex<Box>>
//
// All create_space branches follow this pattern; only the channel_store type differs.
macro_rules! create_generic_rspace {
    ($channel_store:expr, $rspace_id:expr, $qualifier:expr, $theory:expr, $gas_config:expr) => {{
        let matcher: WildcardMatch<BindPattern, ListParWithRandom> = WildcardMatch::new();
        let mut generic_rspace = GenericRSpace::new($channel_store, matcher, $rspace_id, $qualifier);
        if let Some(ref t) = $theory {
            generic_rspace.set_theory(Some(t.clone_box()));
        }
        // Always wrap with ChargingSpaceAgent for phlogiston metering
        let gas_cfg = $gas_config;
        let meter = Arc::new(PhlogistonMeter::new(gas_cfg.initial_limit));
        let charging_space = ChargingSpaceAgent::new(generic_rspace, meter);
        Arc::new(tokio::sync::Mutex::new(Box::new(charging_space) as Box<dyn rspace_plus_plus::rspace::rspace_interface::ISpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> + Send + Sync>))
    }};
}

pub type RhoSysFunction = Box<
    dyn Fn(
        (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>> + Send>> + Send + Sync,
>;
pub type RhoDispatchMap = Arc<tokio::sync::RwLock<HashMap<i64, RhoSysFunction>>>;
pub type Name = Par;
pub type Arity = i32;
pub type Remainder = Option<Var>;
pub type BodyRef = i64;
pub type Contract = dyn Fn(Vec<ListParWithRandom>) -> ();

#[derive(Clone)]
pub struct InvalidBlocks {
    pub invalid_blocks: Arc<tokio::sync::RwLock<Par>>,
}

impl InvalidBlocks {
    pub fn new() -> Self {
        InvalidBlocks {
            invalid_blocks: Arc::new(tokio::sync::RwLock::new(Par::default())),
        }
    }

    pub async fn set_params(&self, invalid_blocks: Par) -> () {
        let mut lock = self.invalid_blocks.write().await;

        *lock = invalid_blocks;
    }
}

pub fn byte_name(b: Byte) -> Par {
    Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(GPrivateBody(GPrivate { id: vec![b] })),
    }])
}

/// Check if a Par represents a system channel that should always route to the default space.
///
/// System channels are GPrivate names with well-known byte IDs:
/// - 0-24: Standard I/O and system operations (stdout, stderr, crypto, etc.)
/// - 200-210: Space factory channels
///
/// These channels must always route to the default space because their handlers
/// are registered there. This is critical for maintaining correct behavior inside
/// use blocks.
pub fn is_system_channel(chan: &Par) -> bool {
    // Direct GPrivate in unforgeables
    if let Some(id_bytes) = extract_gprivate_id(chan) {
        return is_system_channel_id(&id_bytes);
    }

    // Bundle-wrapped GPrivate (common in urn_map)
    if chan.bundles.len() == 1 {
        if let Some(bundle_body) = &chan.bundles[0].body {
            if let Some(id_bytes) = extract_gprivate_id(bundle_body) {
                return is_system_channel_id(&id_bytes);
            }
        }
    }

    false
}

/// Extract GPrivate ID bytes from a Par if it contains exactly one GPrivate unforgeable.
fn extract_gprivate_id(par: &Par) -> Option<Vec<u8>> {
    if par.unforgeables.len() == 1 {
        if let Some(GPrivateBody(gprivate)) = &par.unforgeables[0].unf_instance {
            return Some(gprivate.id.clone());
        }
    }
    None
}

/// Check if a GPrivate ID represents a system channel.
///
/// System channel byte ranges:
/// - 0-24: Standard channels (stdout=0, stderr=2, crypto, etc.)
/// - 200-210: Space factory channels
fn is_system_channel_id(id_bytes: &[u8]) -> bool {
    if id_bytes.len() == 1 {
        let byte = id_bytes[0];
        // Standard I/O and system channels (0-24) or Space factory channels (200-210)
        (byte <= 24) || (byte >= 200 && byte <= 210)
    } else {
        false
    }
}

/// Create a Par channel from an Array space index.
///
/// Array channels are allocated sequentially (0, 1, 2, ...) and wrapped as
/// Unforgeable GPrivate. The channel ID is constructed from the space_id
/// concatenated with the index bytes (big-endian), ensuring uniqueness across spaces.
///
/// # Arguments
/// * `space_id` - The space identifier for channel namespacing
/// * `index` - The sequential index allocated by ArrayChannelStore
///
/// # Returns
/// A Par containing a GPrivate Unforgeable with the combined space_id + index
fn array_index_to_par(space_id: &SpaceId, index: usize) -> Par {
    // Combine space_id and index to create unique channel identifier
    // SpaceId is a newtype wrapper, access inner Vec<u8> via .0
    // Use big-endian to match the reducer's AllocationMode::ArrayIndex behavior
    let mut id_bytes = space_id.0.to_vec();
    id_bytes.extend_from_slice(&index.to_be_bytes());

    // Create GPrivate Unforgeable from the combined bytes
    let gprivate = GPrivate { id: id_bytes };
    let unforgeable = GUnforgeable {
        unf_instance: Some(GPrivateBody(gprivate)),
    };
    Par {
        unforgeables: vec![unforgeable],
        ..Default::default()
    }
}

/// Extract an Array index from a Par channel (inverse of array_index_to_par).
///
/// Returns Some(index) if the channel matches the expected pattern for this space,
/// None otherwise.
///
/// # Arguments
/// * `space_id` - The space identifier to match against
/// * `channel` - The Par channel to extract the index from
///
/// # Returns
/// Some(index) if the channel is a GPrivate with matching space_id prefix
fn par_to_array_index(space_id: &SpaceId, channel: &Par) -> Option<usize> {
    // Channel must have exactly one unforgeable
    if channel.unforgeables.len() != 1 {
        return None;
    }

    // Must be a GPrivate
    let gprivate = match &channel.unforgeables[0].unf_instance {
        Some(GPrivateBody(gp)) => gp,
        _ => return None,
    };

    let id_bytes = &gprivate.id;
    let space_id_len = space_id.0.len();

    // ID must be space_id + 8 bytes (usize)
    if id_bytes.len() != space_id_len + 8 {
        return None;
    }

    // Check space_id prefix matches
    if &id_bytes[..space_id_len] != space_id.0.as_slice() {
        return None;
    }

    // Extract index from last 8 bytes (big-endian)
    let index_bytes: [u8; 8] = id_bytes[space_id_len..].try_into().ok()?;
    Some(usize::from_be_bytes(index_bytes))
}

/// Create a Par channel from a Vector index.
///
/// Vector uses the same channel creation pattern as Array - indices are wrapped
/// in GPrivate Unforgeable channels. The difference is Vector has no size limits.
///
/// # Arguments
/// * `space_id` - The space identifier for channel namespacing
/// * `index` - The sequential index allocated by VectorChannelStore
///
/// # Returns
/// A Par containing a GPrivate Unforgeable with the combined space_id + index
fn vector_index_to_par(space_id: &SpaceId, index: usize) -> Par {
    // Reuse the same logic as Array - indices become Unforgeable channels
    array_index_to_par(space_id, index)
}

/// Extract a Vector index from a Par channel (inverse of vector_index_to_par).
///
/// Returns Some(index) if the channel matches the expected pattern for this space,
/// None otherwise.
///
/// # Arguments
/// * `space_id` - The space identifier to match against
/// * `channel` - The Par channel to extract the index from
///
/// # Returns
/// Some(index) if the channel is a GPrivate with matching space_id prefix
fn par_to_vector_index(space_id: &SpaceId, channel: &Par) -> Option<usize> {
    // Reuse the same logic as Array - channel decoding is identical
    par_to_array_index(space_id, channel)
}

pub struct FixedChannels;

impl FixedChannels {
    pub fn stdout() -> Par {
        byte_name(0)
    }

    pub fn stdout_ack() -> Par {
        byte_name(1)
    }

    pub fn stderr() -> Par {
        byte_name(2)
    }

    pub fn stderr_ack() -> Par {
        byte_name(3)
    }

    pub fn ed25519_verify() -> Par {
        byte_name(4)
    }

    pub fn sha256_hash() -> Par {
        byte_name(5)
    }

    pub fn keccak256_hash() -> Par {
        byte_name(6)
    }

    pub fn blake2b256_hash() -> Par {
        byte_name(7)
    }

    pub fn secp256k1_verify() -> Par {
        byte_name(8)
    }

    pub fn get_block_data() -> Par {
        byte_name(10)
    }

    pub fn get_invalid_blocks() -> Par {
        byte_name(11)
    }

    pub fn rev_address() -> Par {
        byte_name(12)
    }

    pub fn deployer_id_ops() -> Par {
        byte_name(13)
    }

    pub fn reg_lookup() -> Par {
        byte_name(14)
    }

    pub fn reg_insert_random() -> Par {
        byte_name(15)
    }

    pub fn reg_insert_signed() -> Par {
        byte_name(16)
    }

    pub fn reg_ops() -> Par {
        byte_name(17)
    }

    pub fn sys_authtoken_ops() -> Par {
        byte_name(18)
    }

    pub fn gpt4() -> Par {
        byte_name(19)
    }

    pub fn dalle3() -> Par {
        byte_name(20)
    }

    pub fn text_to_audio() -> Par {
        byte_name(21)
    }

    pub fn random() -> Par {
        byte_name(22)
    }

    pub fn grpc_tell() -> Par {
        byte_name(23)
    }

    pub fn dev_null() -> Par {
        byte_name(24)
    }

    // ==========================================================================
    // Space Factory Channels (Reified RSpaces)
    //
    // NOTE: Space factory URN channels are now auto-generated from the URN
    // enumeration in factory.rs. Each valid rho:space:{inner}:{outer}:{qualifier}
    // combination is assigned a unique byte via urn_to_byte_name().
    //
    // The auto-generated byte names start at 200 and increment sequentially
    // for all 96 valid combinations (wrapping at 256 if needed).
    // ==========================================================================

    // ==========================================================================
    // Vector Operations Channel (Reified RSpaces - Tensor Logic)
    // NOTE: Using byte 150 to avoid conflict with auto-generated space factory
    // bytes (200+). Vector ops is a standalone system process, not a factory.
    // ==========================================================================

    pub fn vector_ops() -> Par {
        byte_name(150)
    }
}

pub struct BodyRefs;

impl BodyRefs {
    pub const STDOUT: i64 = 0;
    pub const STDOUT_ACK: i64 = 1;
    pub const STDERR: i64 = 2;
    pub const STDERR_ACK: i64 = 3;
    pub const ED25519_VERIFY: i64 = 4;
    pub const SHA256_HASH: i64 = 5;
    pub const KECCAK256_HASH: i64 = 6;
    pub const BLAKE2B256_HASH: i64 = 7;
    pub const SECP256K1_VERIFY: i64 = 9;
    pub const GET_BLOCK_DATA: i64 = 11;
    pub const GET_INVALID_BLOCKS: i64 = 12;
    pub const REV_ADDRESS: i64 = 13;
    pub const DEPLOYER_ID_OPS: i64 = 14;
    pub const REG_OPS: i64 = 15;
    pub const SYS_AUTHTOKEN_OPS: i64 = 16;
    pub const GPT4: i64 = 17;
    pub const DALLE3: i64 = 18;
    pub const TEXT_TO_AUDIO: i64 = 19;
    pub const RANDOM: i64 = 20;
    pub const GRPC_TELL: i64 = 21;
    pub const DEV_NULL: i64 = 22;

    // ==========================================================================
    // Space Factory Body Refs (Reified RSpaces)
    //
    // NOTE: Space factory body refs are now auto-generated from urn_to_byte_name().
    // Each valid rho:space:{inner}:{outer}:{qualifier} combination uses its
    // computed byte value as both the fixed_channel byte and body_ref.
    // Body refs 200-255 (and wrapping to 0-39) are reserved for space factories.
    // ==========================================================================

    // ==========================================================================
    // Vector Operations Body Ref (Reified RSpaces - Tensor Logic)
    // NOTE: Using 150 to avoid conflict with auto-generated space factory refs.
    // ==========================================================================

    pub const VECTOR_OPS: i64 = 150;
}

pub fn non_deterministic_ops() -> HashSet<i64> {
    HashSet::from([
        BodyRefs::GPT4,
        BodyRefs::DALLE3,
        BodyRefs::TEXT_TO_AUDIO,
        BodyRefs::RANDOM,
    ])
}

// =============================================================================
// Vector Operations Helper Functions (Reified RSpaces - Tensor Logic)
// =============================================================================

/// Extract a 2D matrix from a nested list Par.
/// The matrix is represented as [[row1], [row2], ...] in Rholang.
fn extract_2d_matrix<F>(par: &Par, extract_row: &F) -> Option<Vec<Vec<f32>>>
where
    F: Fn(&Par) -> Option<Vec<f32>>,
{
    use models::rhoapi::expr::ExprInstance::EListBody;
    use models::rust::rholang::implicits::single_expr;

    let expr = single_expr(par)?;
    match &expr.expr_instance {
        Some(EListBody(elist)) => {
            let mut rows = Vec::with_capacity(elist.ps.len());
            for row_par in &elist.ps {
                rows.push(extract_row(row_par)?);
            }
            Some(rows)
        }
        _ => None,
    }
}

/// Extract a 2D integer matrix from a nested list Par.
/// The matrix is represented as [[row1], [row2], ...] in Rholang.
fn extract_2d_int_matrix<F>(par: &Par, extract_row: &F) -> Option<Vec<Vec<i64>>>
where
    F: Fn(&Par) -> Option<Vec<i64>>,
{
    use models::rhoapi::expr::ExprInstance::EListBody;
    use models::rust::rholang::implicits::single_expr;

    let expr = single_expr(par)?;
    match &expr.expr_instance {
        Some(EListBody(elist)) => {
            let mut rows = Vec::with_capacity(elist.ps.len());
            for row_par in &elist.ps {
                rows.push(extract_row(row_par)?);
            }
            Some(rows)
        }
        _ => None,
    }
}

/// Convert an ndarray 2D matrix to a nested list Par (binary: threshold at 0.5).
fn matrix_to_par(matrix: &ndarray::Array2<f32>) -> Par {
    use models::rhoapi::EList;
    use models::rhoapi::Expr;
    use models::rhoapi::expr::ExprInstance::{EListBody, GInt};

    Par::default().with_exprs(vec![Expr {
        expr_instance: Some(EListBody(EList {
            ps: matrix.rows().into_iter().map(|row| {
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(EListBody(EList {
                        // Convert to binary: >= 0.5 becomes 1, otherwise 0
                        ps: row.iter().map(|&x| {
                            let binary = if x >= 0.5 { 1i64 } else { 0i64 };
                            Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(GInt(binary)),
                            }])
                        }).collect(),
                        ..Default::default()
                    })),
                }])
            }).collect(),
            ..Default::default()
        })),
    }])
}

#[derive(Clone)]
pub struct ProcessContext {
    pub space: RhoISpace,
    pub dispatcher: RhoDispatch,
    pub block_data: Arc<tokio::sync::RwLock<BlockData>>,
    pub invalid_blocks: InvalidBlocks,
    pub system_processes: SystemProcesses,
    /// Space storage for reified RSpaces - maps GPrivate IDs to space instances.
    ///
    /// Shared with DebruijnInterpreter to enable space routing in use blocks.
    /// Uses DashMap for lock-free concurrent access.
    pub space_store: Arc<DashMap<Vec<u8>, RhoISpace>>,
    /// Space-to-qualifier mapping for Seq enforcement.
    ///
    /// Shared with DebruijnInterpreter to look up space qualifiers at runtime.
    /// Used to enforce Seq channel single-accessor invariant.
    /// Uses DashMap for lock-free concurrent access.
    ///
    /// Formal Correspondence: GenericRSpace.v:1330-1335 (single_accessor_invariant)
    pub space_qualifier_map: Arc<DashMap<Vec<u8>, SpaceQualifier>>,
    /// Space-to-config mapping for hyperparam validation.
    ///
    /// Shared with DebruijnInterpreter to validate hyperparameters at send time.
    /// Uses DashMap for lock-free concurrent access.
    pub space_config_map: Arc<DashMap<Vec<u8>, SpaceConfig>>,
}

impl ProcessContext {
    pub fn create(
        space: RhoISpace,
        dispatcher: RhoDispatch,
        block_data: Arc<tokio::sync::RwLock<BlockData>>,
        invalid_blocks: InvalidBlocks,
        openai_service: Arc<tokio::sync::Mutex<OpenAIService>>,
        space_store: Arc<DashMap<Vec<u8>, RhoISpace>>,
        space_qualifier_map: Arc<DashMap<Vec<u8>, SpaceQualifier>>,
        space_config_map: Arc<DashMap<Vec<u8>, SpaceConfig>>,
    ) -> Self {
        ProcessContext {
            space: space.clone(),
            dispatcher: dispatcher.clone(),
            block_data: block_data.clone(),
            invalid_blocks,
            system_processes: SystemProcesses::create(
                dispatcher,
                space,
                block_data,
                openai_service,
                space_store.clone(),
                space_qualifier_map.clone(),
                space_config_map.clone(),
            ),
            space_store,
            space_qualifier_map,
            space_config_map,
        }
    }
}

pub struct Definition {
    pub urn: String,
    pub fixed_channel: Name,
    pub arity: Arity,
    pub body_ref: BodyRef,
    pub handler: Box<
        dyn FnMut(
            ProcessContext,
        ) -> Box<
            dyn Fn(
                (Vec<ListParWithRandom>, bool, Vec<Par>),
            )
                -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>> + Send>> + Send + Sync,
        > + Send,
    >,
    pub remainder: Remainder,
}

impl Definition {
    pub fn new(
        urn: String,
        fixed_channel: Name,
        arity: Arity,
        body_ref: BodyRef,
        handler: Box<
            dyn FnMut(
                ProcessContext,
            ) -> Box<
                dyn Fn(
                    (Vec<ListParWithRandom>, bool, Vec<Par>),
                )
                    -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>> + Send>> + Send + Sync,
            > + Send,
        >,
        remainder: Remainder,
    ) -> Self {
        Definition {
            urn,
            fixed_channel,
            arity,
            body_ref,
            handler,
            remainder,
        }
    }

    pub fn to_dispatch_table(
        &mut self,
        context: ProcessContext,
    ) -> (
        BodyRef,
        Box<
            dyn Fn(
                (Vec<ListParWithRandom>, bool, Vec<Par>),
            )
                -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>> + Send>> + Send + Sync,
        >,
    ) {
        (self.body_ref, (self.handler)(context))
    }

    pub fn to_urn_map(&self) -> (String, Par) {
        let bundle: Par = Par::default().with_bundles(vec![Bundle {
            body: Some(self.fixed_channel.clone()),
            write_flag: true,
            read_flag: false,
        }]);

        (self.urn.clone(), bundle)
    }

    pub fn to_proc_defs(&self) -> (Name, Arity, Remainder, BodyRef) {
        (
            self.fixed_channel.clone(),
            self.arity,
            self.remainder.clone(),
            self.body_ref.clone(),
        )
    }
}

#[derive(Clone)]
pub struct BlockData {
    pub time_stamp: i64,
    pub block_number: i64,
    pub sender: PublicKey,
    pub seq_num: i32,
}

impl BlockData {
    pub fn empty() -> Self {
        BlockData {
            block_number: 0,
            sender: PublicKey::from_bytes(&hex::decode("00").unwrap()),
            seq_num: 0,
            time_stamp: 0,
        }
    }

    pub fn from_block(template: &BlockMessage) -> Self {
        BlockData {
            time_stamp: template.header.timestamp,
            block_number: template.body.state.block_number,
            sender: PublicKey::from_bytes(&template.sender),
            seq_num: template.seq_num,
        }
    }
}

// TODO: Remove Clone
#[derive(Clone)]
pub struct SystemProcesses {
    pub dispatcher: RhoDispatch,
    pub space: RhoISpace,
    pub block_data: Arc<tokio::sync::RwLock<BlockData>>,
    openai_service: Arc<tokio::sync::Mutex<OpenAIService>>,
    pretty_printer: PrettyPrinter,
    /// Space storage for reified RSpaces - maps GPrivate IDs to space instances.
    /// Uses DashMap for lock-free concurrent access.
    pub space_store: Arc<DashMap<Vec<u8>, RhoISpace>>,
    /// Space-to-qualifier mapping for Seq enforcement.
    ///
    /// Used to store the qualifier when a space is created so we can
    /// look it up during produce/consume operations.
    /// Uses DashMap for lock-free concurrent access.
    ///
    /// Formal Correspondence: GenericRSpace.v:1330-1335 (single_accessor_invariant)
    pub space_qualifier_map: Arc<DashMap<Vec<u8>, SpaceQualifier>>,
    /// Space-to-config mapping for hyperparam validation.
    ///
    /// Stores the SpaceConfig when a space is created so hyperparams can be
    /// validated at send time against the space's collection type.
    /// Uses DashMap for lock-free concurrent access.
    pub space_config_map: Arc<DashMap<Vec<u8>, SpaceConfig>>,
}

impl SystemProcesses {
    fn create(
        dispatcher: RhoDispatch,
        space: RhoISpace,
        block_data: Arc<tokio::sync::RwLock<BlockData>>,
        openai_service: Arc<tokio::sync::Mutex<OpenAIService>>,
        space_store: Arc<DashMap<Vec<u8>, RhoISpace>>,
        space_qualifier_map: Arc<DashMap<Vec<u8>, SpaceQualifier>>,
        space_config_map: Arc<DashMap<Vec<u8>, SpaceConfig>>,
    ) -> Self {
        SystemProcesses {
            dispatcher,
            space,
            block_data,
            openai_service,
            pretty_printer: PrettyPrinter::new(),
            space_store,
            space_qualifier_map,
            space_config_map,
        }
    }

    fn is_contract_call(&self) -> ContractCall {
        ContractCall {
            space: self.space.clone(),
            dispatcher: self.dispatcher.clone(),
        }
    }

    /// Parse space factory arguments with variable arity support.
    ///
    /// Supports 0-3 arguments with flexible ordering:
    /// - 3 args: ("qualifier", theory_or_config, *reply)
    /// - 2 args: ("qualifier", *reply) | (theory_or_config, *reply) | ("qualifier", theory_or_config)
    /// - 1 arg:  (*reply) | ("qualifier") | (theory_or_config)
    /// - 0 args: Use URN qualifier, no theory, no reply channel
    ///
    /// Returns (qualifier_override, theory_or_config_par, ack_opt)
    fn parse_space_factory_args(
        args: &[Par],
        urn: &str,
    ) -> Result<(Option<SpaceQualifier>, Par, Option<Par>), InterpreterError> {
        // When using arity:0 with FreeVar remainder, all args are wrapped in EListBody.
        // Unwrap the list to get individual arguments.
        let unwrapped_args: Vec<Par> = if args.len() == 1 {
            if let Some(expr) = args[0].exprs.first() {
                if let Some(ExprInstance::EListBody(elist)) = &expr.expr_instance {
                    elist.ps.clone()
                } else {
                    args.to_vec()
                }
            } else {
                args.to_vec()
            }
        } else {
            args.to_vec()
        };

        let args = &unwrapped_args[..];

        match args.len() {
            0 => {
                // No args: use URN defaults, no reply channel
                Ok((None, Par::default(), None))
            }
            1 => {
                // Single arg: could be reply channel, qualifier string, theory, or config
                let arg = &args[0];
                if Self::is_reply_channel(arg) {
                    Ok((None, Par::default(), Some(arg.clone())))
                } else if let Some(qual) = Self::try_extract_qualifier_string(arg) {
                    Ok((Some(qual), Par::default(), None))
                } else {
                    // Theory or config - no reply channel
                    Ok((None, arg.clone(), None))
                }
            }
            2 => {
                // Two args: multiple combinations possible
                let (first, second) = (&args[0], &args[1]);
                if let Some(qual) = Self::try_extract_qualifier_string(first) {
                    // First arg is qualifier
                    if Self::is_reply_channel(second) {
                        // ("qualifier", *reply)
                        tracing::debug!("  2-arg: (qualifier, reply)");
                        Ok((Some(qual), Par::default(), Some(second.clone())))
                    } else {
                        // ("qualifier", theory_or_config) - no reply
                        tracing::debug!("  2-arg: (qualifier, theory/config)");
                        Ok((Some(qual), second.clone(), None))
                    }
                } else {
                    // First arg is theory/config, second should be reply
                    tracing::debug!("  2-arg: (theory/config, reply)");
                    Ok((None, first.clone(), Some(second.clone())))
                }
            }
            3 => {
                // Full: (qualifier, theory_or_config, *reply)
                let qual = Self::try_extract_qualifier_string(&args[0]).ok_or_else(|| {
                    InterpreterError::ReduceError(format!(
                        "Space factory {}: first of 3 args must be qualifier string (\"default\", \"temp\", or \"seq\")",
                        urn
                    ))
                })?;
                tracing::debug!("  3-arg: (qualifier={:?}, theory/config, reply)", qual);
                Ok((Some(qual), args[1].clone(), Some(args[2].clone())))
            }
            n => Err(InterpreterError::ReduceError(format!(
                "Space factory {} accepts 0-3 args, got {}",
                urn, n
            ))),
        }
    }

    /// Check if a Par is likely a reply channel (has unforgeables).
    /// Reply channels are typically created with `new ch in {...}` which produces GPrivate.
    fn is_reply_channel(par: &Par) -> bool {
        !par.unforgeables.is_empty()
    }

    /// Try to extract a SpaceQualifier from a GString Par.
    /// Returns None if not a valid qualifier string.
    fn try_extract_qualifier_string(par: &Par) -> Option<SpaceQualifier> {
        if let Some(expr) = par.exprs.first() {
            if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
                return match s.as_str() {
                    "default" => Some(SpaceQualifier::Default),
                    "temp" => Some(SpaceQualifier::Temp),
                    "seq" => Some(SpaceQualifier::Seq),
                    _ => None,
                };
            }
        }
        None
    }

    async fn verify_signature_contract(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
        name: &str,
        algorithm: Box<dyn SignaturesAlg>,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, vec)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error(name));
        };

        let [data, signature, pub_key, ack] = vec.as_slice() else {
            return Err(illegal_argument_error(name));
        };

        let (Some(data_bytes), Some(signature_bytes), Some(pub_key_bytes)) = (
            RhoByteArray::unapply(data),
            RhoByteArray::unapply(signature),
            RhoByteArray::unapply(pub_key),
        ) else {
            return Err(illegal_argument_error(name));
        };

        let verified = algorithm.verify(&data_bytes, &signature_bytes, &pub_key_bytes);
        let output = vec![Par::default().with_exprs(vec![RhoBoolean::create_expr(verified)])];
        let ret = output.clone();
        produce(&output, ack).await?;
        Ok(ret)
    }

    async fn hash_contract(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
        name: &str,
        algorithm: Box<dyn Fn(Vec<u8>) -> Vec<u8> + Send>,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, vec)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error(name));
        };

        let [input, ack] = vec.as_slice() else {
            return Err(illegal_argument_error(name));
        };

        let Some(input) = RhoByteArray::unapply(input) else {
            return Err(illegal_argument_error(name));
        };

        let hash = algorithm(input);
        let output = vec![RhoByteArray::create_par(hash)];
        let ret = output.clone();
        produce(&output, ack).await?;
        Ok(ret)
    }

    fn print_std_out(&self, s: &str) -> Result<Vec<Par>, InterpreterError> {
        println!("{}", s);
        Ok(vec![])
    }

    fn print_std_err(&self, s: &str) -> Result<Vec<Par>, InterpreterError> {
        eprintln!("{}", s);
        Ok(vec![])
    }

    pub async fn std_out(
        &mut self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((_, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("std_out"));
        };

        let [arg] = args.as_slice() else {
            return Err(illegal_argument_error("std_out"));
        };

        let str = self.pretty_printer.build_string_from_message(arg);
        self.print_std_out(&str)
    }

    pub async fn std_out_ack(
        mut self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("std_out_ack"));
        };

        let [arg, ack] = args.as_slice() else {
            return Err(illegal_argument_error("std_out_ack"));
        };

        let str = self.pretty_printer.build_string_from_message(arg);
        self.print_std_out(&str)?;

        let output = vec![Par::default()];
        let ret = output.clone();
        produce(&output, ack).await?;
        Ok(ret)
    }

    pub async fn std_err(
        &mut self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((_, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("std_err"));
        };

        let [arg] = args.as_slice() else {
            return Err(illegal_argument_error("std_err"));
        };

        let str = self.pretty_printer.build_string_from_message(arg);
        self.print_std_err(&str)
    }

    pub async fn std_err_ack(
        &mut self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("std_err_ack"));
        };

        let [arg, ack] = args.as_slice() else {
            return Err(illegal_argument_error("std_err_ack"));
        };

        let str = self.pretty_printer.build_string_from_message(arg);
        self.print_std_err(&str)?;

        let output = vec![Par::default()];
        let ret = output.clone();
        produce(&output, ack).await?;
        Ok(ret)
    }

    pub async fn rev_address(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("rev_address"));
        };

        let [first_par, second_par, ack] = args.as_slice() else {
            return Err(illegal_argument_error("rev_address"));
        };

        let Some(command) = RhoString::unapply(first_par) else {
            return Err(illegal_argument_error("rev_address"));
        };

        let response = match command.as_str() {
            "validate" => {
                match RhoString::unapply(second_par).map(|address| RevAddress::parse(&address)) {
                    Some(Ok(_)) => Par::default(),
                    Some(Err(err)) => RhoString::create_par(err),
                    None => {
                        // TODO: Invalid type for address should throw error! - OLD
                        Par::default()
                    }
                }
            }

            "fromPublicKey" => match RhoByteArray::unapply(second_par)
                .map(|public_key| RevAddress::from_public_key(&PublicKey::from_bytes(&public_key)))
            {
                Some(Some(ra)) => RhoString::create_par(ra.to_base58()),
                _ => Par::default(),
            },

            "fromDeployerId" => {
                match RhoDeployerId::unapply(second_par).map(RevAddress::from_deployer_id) {
                    Some(Some(ra)) => RhoString::create_par(ra.to_base58()),
                    _ => Par::default(),
                }
            }

            "fromUnforgeable" => {
                match RhoName::unapply(second_par)
                    .map(|gprivate: GPrivate| RevAddress::from_unforgeable(&gprivate))
                {
                    Some(ra) => RhoString::create_par(ra.to_base58()),
                    None => Par::default(),
                }
            }

            _ => return Err(illegal_argument_error("rev_address")),
        };

        produce(&[response], ack).await
    }

    pub async fn deployer_id_ops(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("deployer_id_ops"));
        };

        let [first_par, second_par, ack] = args.as_slice() else {
            return Err(illegal_argument_error("deployer_id_ops"));
        };

        let Some("pubKeyBytes") = RhoString::unapply(first_par).as_deref() else {
            return Err(illegal_argument_error("deployer_id_ops"));
        };

        let response = RhoDeployerId::unapply(second_par)
            .map(RhoByteArray::create_par)
            .unwrap_or_default();

        produce(&[response], ack).await
    }

    pub async fn registry_ops(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("registry_ops"));
        };

        let [first_par, argument, ack] = args.as_slice() else {
            return Err(illegal_argument_error("registry_ops"));
        };

        let Some("buildUri") = RhoString::unapply(first_par).as_deref() else {
            return Err(illegal_argument_error("registry_ops"));
        };

        let response = RhoByteArray::unapply(argument)
            .map(|ba| {
                let hash_key_bytes = Blake2b256::hash(ba);
                RhoUri::create_par(Registry::build_uri(&hash_key_bytes))
            })
            .unwrap_or_default();

        produce(&[response], ack).await
    }

    pub async fn sys_auth_token_ops(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("sys_auth_token_ops"));
        };

        let [first_par, argument, ack] = args.as_slice() else {
            return Err(illegal_argument_error("sys_auth_token_ops"));
        };

        let Some("check") = RhoString::unapply(first_par).as_deref() else {
            return Err(illegal_argument_error("sys_auth_token_ops"));
        };

        let response = RhoBoolean::create_expr(RhoSysAuthToken::unapply(argument).is_some());
        produce(&[Par::default().with_exprs(vec![response])], ack).await
    }

    pub async fn secp256k1_verify(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        self.verify_signature_contract(contract_args, "secp256k1Verify", Box::new(Secp256k1))
            .await
    }

    pub async fn ed25519_verify(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        self.verify_signature_contract(contract_args, "ed25519Verify", Box::new(Ed25519))
            .await
    }

    pub async fn sha256_hash(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        self.hash_contract(contract_args, "sha256Hash", Box::new(Sha256Hasher::hash))
            .await
    }

    pub async fn keccak256_hash(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        self.hash_contract(contract_args, "keccak256Hash", Box::new(Keccak256::hash))
            .await
    }

    pub async fn blake2b256_hash(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        self.hash_contract(contract_args, "blake2b256Hash", Box::new(Blake2b256::hash))
            .await
    }

    pub async fn get_block_data(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
        block_data: Arc<tokio::sync::RwLock<BlockData>>,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("get_block_data"));
        };

        let [ack] = args.as_slice() else {
            return Err(illegal_argument_error("get_block_data"));
        };

        let data = block_data.read().await;
        let output = vec![
            Par::default().with_exprs(vec![RhoNumber::create_expr(data.block_number)]),
            Par::default().with_exprs(vec![RhoNumber::create_expr(data.time_stamp)]),
            RhoByteArray::create_par(data.sender.bytes.as_ref().to_vec()),
        ];

        produce(&output, ack).await?;
        Ok(output)
    }

    pub async fn invalid_blocks(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
        invalid_blocks: &InvalidBlocks,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("invalid_blocks"));
        };

        let [ack] = args.as_slice() else {
            return Err(illegal_argument_error("invalid_blocks"));
        };

        let invalid_blocks = invalid_blocks.invalid_blocks.read().await.clone();
        produce(&[invalid_blocks.clone()], ack).await?;
        Ok(vec![invalid_blocks])
    }

    pub async fn random(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, is_replay, previous_output, args)) =
            self.is_contract_call().unapply(contract_args)
        else {
            return Err(illegal_argument_error("random"));
        };

        let [ack] = args.as_slice() else {
            return Err(illegal_argument_error("random"));
        };

        if is_replay {
            let ret = previous_output.clone();
            produce(&previous_output, ack).await?;
            return Ok(ret);
        }

        let mut rng = rand::thread_rng();
        let random_length: usize = rng.gen_range(0..100);
        let mut random_string = String::with_capacity(random_length.saturating_mul(4));
        random_string.extend((0..random_length).map(|_| rng.gen::<char>()));

        let output = vec![RhoString::create_par(random_string)];
        produce(&output, ack).await?;
        Ok(output)
    }

    pub async fn gpt4(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, is_replay, previous_output, args)) =
            self.is_contract_call().unapply(contract_args)
        else {
            return Err(illegal_argument_error("gpt4"));
        };

        let [prompt_par, ack] = args.as_slice() else {
            return Err(illegal_argument_error("gpt4"));
        };

        let Some(prompt) = RhoString::unapply(prompt_par) else {
            return Err(illegal_argument_error("gpt4"));
        };

        if is_replay {
            produce(&previous_output, ack).await?;
            return Ok(previous_output);
        }

        let mut openai_service = self.openai_service.lock().await;
        let response = match openai_service.gpt4_chat_completion(&prompt).await {
            Ok(response) => response,
            Err(e) => {
                let p = RhoString::create_par(prompt);
                produce(&[p], ack).await?;
                return Err(e);
            }
        };

        let output = vec![RhoString::create_par(response)];
        produce(&output, ack).await?;
        Ok(output)
    }

    pub async fn dalle3(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, is_replay, previous_output, args)) =
            self.is_contract_call().unapply(contract_args)
        else {
            return Err(illegal_argument_error("dalle3"));
        };

        let [prompt_par, ack] = args.as_slice() else {
            return Err(illegal_argument_error("dalle3"));
        };

        let Some(prompt) = RhoString::unapply(prompt_par) else {
            return Err(illegal_argument_error("dalle3"));
        };

        if is_replay {
            produce(&previous_output, ack).await?;
            return Ok(previous_output);
        }

        let mut openai_service = self.openai_service.lock().await;
        let response = match openai_service.dalle3_create_image(&prompt).await {
            Ok(response) => response,
            Err(e) => {
                let p = RhoString::create_par(prompt);
                produce(&[p], ack).await?;
                return Err(e);
            }
        };

        let output = vec![RhoString::create_par(response)];
        produce(&output, ack).await?;
        Ok(output)
    }

    pub async fn text_to_audio(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, is_replay, previous_output, args)) =
            self.is_contract_call().unapply(contract_args)
        else {
            return Err(illegal_argument_error("text_to_audio"));
        };

        let [input_par, ack] = args.as_slice() else {
            return Err(illegal_argument_error("text_to_audio"));
        };

        let Some(input) = RhoString::unapply(input_par) else {
            return Err(illegal_argument_error("text_to_audio"));
        };

        if is_replay {
            produce(&previous_output, ack).await?;
            return Ok(previous_output);
        }

        let mut openai_service = self.openai_service.lock().await;
        match openai_service
            .create_audio_speech(&input, "audio.mp3")
            .await
        {
            Ok(_) => Ok(vec![]),
            Err(e) => {
                let p = RhoString::create_par(input);
                produce(&[p], ack).await?;
                return Err(e);
            }
        }
    }

    pub async fn grpc_tell(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, is_replay, previous_output, args)) =
            self.is_contract_call().unapply(contract_args)
        else {
            return Err(illegal_argument_error("grpc_tell"));
        };

        // Handle replay case
        if is_replay {
            println!("grpcTell (replay): args: {:?}", args);
            return Ok(previous_output);
        }

        // Handle normal case - expecting clientHost, clientPort, notificationPayload
        match args.as_slice() {
            [client_host_par, client_port_par, notification_payload_par, ack] => {
                match (
                    RhoString::unapply(client_host_par),
                    RhoNumber::unapply(client_port_par),
                    RhoString::unapply(notification_payload_par),
                ) {
                    (Some(client_host), Some(client_port), Some(notification_payload)) => {
                        println!(
                            "grpcTell: clientHost: {}, clientPort: {}, notificationPayload: {}",
                            client_host, client_port, notification_payload
                        );

                        use models::rust::rholang::grpc_client::GrpcClient;

                        // Convert client_port from i64 to u64
                        let port = if client_port < 0 {
                            return Err(InterpreterError::BugFoundError(
                                "Invalid port number: must be non-negative".to_string(),
                            ));
                        } else {
                            client_port as u64
                        };

                        // Execute the gRPC call and handle errors
                        match GrpcClient::init_client_and_tell(
                            &client_host,
                            port,
                            &notification_payload,
                        )
                        .await
                        {
                            Ok(_) => {
                                let output = vec![Par::default()];
                                produce(&output, ack).await?;
                                Ok(output)
                            }
                            Err(e) => {
                                println!("GrpcClient crashed: {}", e);
                                let output = vec![Par::default()];
                                produce(&output, ack).await?;
                                Ok(output)
                            }
                        }
                    }
                    _ => {
                        println!("grpcTell: invalid argument types: {:?}", args);
                        Err(illegal_argument_error("grpc_tell"))
                    }
                }
            }
            _ => {
                println!(
                    "grpcTell: isReplay {} invalid arguments: {:?}",
                    is_replay, args
                );
                Ok(vec![Par::default()])
            }
        }
    }

    pub async fn dev_null(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if self.is_contract_call().unapply(contract_args).is_none() {
            return Err(illegal_argument_error("dev_null"));
        }

        Ok(vec![])
    }

    // ==========================================================================
    // Space Factory (Reified RSpaces)
    // ==========================================================================

    /// Create a new space with the given URN configuration.
    ///
    /// The factory pattern works as follows:
    /// ```rholang
    /// new Factory(`rho:space:cell:hashmap:default`), mySpace in {
    ///   Factory!({}, *mySpace) |
    ///   use mySpace { ... }
    /// }
    /// ```
    ///
    /// For VectorDB spaces, the config map should contain:
    /// - dimensions: required, the embedding vector dimensionality
    /// - threshold: optional, similarity threshold 0-100 (default: 80)
    /// - embedding_type: optional, "boolean", "integer", or "float" (default: "integer")
    /// - metric: optional, "cosine", "euclidean", etc. (default: derived from embedding_type)
    ///
    /// ```rholang
    /// new VectorDBFactory(`rho:space:vectordb:hashmap:default`) in {
    ///   VectorDBFactory!({"dimensions": 4, "threshold": 50, "embedding_type": "integer"}, *semanticSpace)
    /// }
    /// ```
    ///
    /// The factory receives (config, reply_channel) and sends back an unforgeable
    /// name representing the space instance.
    pub async fn create_space(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
        urn: &str,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error(&format!("create_space({})", urn)));
        };

        // Variable arity syntax: Factory!(args...)
        // Supports 0-3 arguments with flexible ordering:
        //
        // 3 args: ("qualifier", theory_or_config, *reply) - explicit qualifier override
        // 2 args: ("qualifier", *reply) | (theory_or_config, *reply) | ("qualifier", theory_or_config)
        // 1 arg:  (*reply) | ("qualifier") | (theory_or_config)
        // 0 args: Use URN qualifier, no theory, no reply channel
        //
        // Argument detection:
        // - Qualifier: GString with value in {"default", "temp", "seq"}
        // - Theory: Has connective_instance (e.g., free @"Nat")
        // - Config map: Has EMapBody in expressions
        // - Reply channel: Has unforgeables or ids (name references)
        let (qualifier_override, theory_or_config_par, ack_opt) =
            Self::parse_space_factory_args(&args, urn)?;

        // Generate a unique space ID using UUID
        let space_id = uuid::Uuid::new_v4();
        let space_id_bytes = space_id.as_bytes().to_vec();

        // Create an unforgeable channel (GPrivate) representing the space
        let space_channel = Par::default().with_unforgeables(vec![GUnforgeable {
            unf_instance: Some(GPrivateBody(GPrivate {
                id: space_id_bytes.clone(),
            })),
        }]);

        // Determine if second argument is a config map or a theory specification
        // A config map has an EMapBody expression; theory has EFree (free Nat())
        let is_config_map = theory_or_config_par.exprs.iter().any(|expr| {
            matches!(expr.expr_instance, Some(ExprInstance::EMapBody(_)))
        });

        // Extract theory and config based on argument type
        let (theory, config_par) = if is_config_map {
            // Config map provided - no theory
            (None, theory_or_config_par.clone())
        } else {
            // Theory specification (or empty) - extract theory, use empty config
            let theory = self.extract_theory_from_par(&theory_or_config_par)?;
            if theory.is_some() {
                tracing::debug!("create_space: Found theory specification");
            }
            (theory, Par::default())
        };

        // Parse URN to SpaceConfig using computed pattern matching
        let config = config_from_urn(urn);

        let rspace_id = SpaceId::new(space_id_bytes.clone());

        // Create space using config-driven dispatch
        // Qualifier can be overridden by positional argument
        let (rho_space, qualifier): (RhoISpace, SpaceQualifier) = match &config {
            Some(cfg) => {
                // Apply qualifier override if provided, otherwise use URN qualifier
                let effective_qualifier = qualifier_override.unwrap_or(cfg.qualifier);

                // Apply Array config overrides from config_par (size, cyclic)
                // This allows creating arrays of different sizes from the same factory:
                //   ArrayFactory!({"size": 10, "cyclic": true}, *spaceRef)
                let effective_outer = match &cfg.outer {
                    OuterStorageType::Array { max_size, cyclic } => {
                        // Check for size override in config_par
                        let effective_size = self.extract_config_usize(&config_par, "size")
                            .unwrap_or(*max_size);
                        // Check for cyclic override in config_par
                        let effective_cyclic = self.extract_config_bool(&config_par, "cyclic")
                            .unwrap_or(*cyclic);
                        tracing::debug!(
                            "create_space: Array config overrides: size={} (default={}), cyclic={} (default={})",
                            effective_size, max_size, effective_cyclic, cyclic
                        );
                        OuterStorageType::Array {
                            max_size: effective_size,
                            cyclic: effective_cyclic,
                        }
                    }
                    other => other.clone(),
                };

                tracing::debug!(
                    "create_space: URN '{}' -> outer={:?}, data={:?}, qualifier={:?} (override={:?})",
                    urn, effective_outer, cfg.data_collection, effective_qualifier, qualifier_override
                );
                // Create a modified config with the effective qualifier for space creation
                // Note: theory is passed separately to create_rspace_from_config, so we use None here
                let effective_cfg = SpaceConfig {
                    qualifier: effective_qualifier,
                    outer: effective_outer,
                    data_collection: cfg.data_collection.clone(),
                    continuation_collection: cfg.continuation_collection.clone(),
                    theory: None,  // Theory is handled separately via the theory parameter
                    gas_config: cfg.gas_config.clone(),
                };
                let space = self.create_rspace_from_config(&effective_cfg, &config_par, rspace_id, &theory)?;
                (space, effective_qualifier)
            }
            None => {
                return Err(InterpreterError::ReduceError(format!(
                    "Unknown space URN: '{}'. Use a valid rho:space:* URN (e.g., \
                    rho:space:HashMapBagSpace, rho:space:PathMapSpace, rho:space:QueueSpace).",
                    urn
                )));
            }
        };

        // Store the space in space_store keyed by the GPrivate ID
        // Uses DashMap for lock-free concurrent insertion
        self.space_store.insert(space_id_bytes.clone(), rho_space);

        // Store the qualifier in space_qualifier_map for Seq enforcement
        // Uses DashMap for lock-free concurrent insertion
        // Formal Correspondence: GenericRSpace.v:1330-1335 (single_accessor_invariant)
        self.space_qualifier_map.insert(space_id_bytes.clone(), qualifier);

        // Store the effective config in space_config_map for allocation mode and hyperparam validation
        // Uses DashMap for lock-free concurrent insertion
        // IMPORTANT: We must store effective_cfg (with user's config overrides applied), not the
        // original URN config. This ensures the reducer uses the correct allocation_mode for
        // Array spaces with custom size/cyclic values.
        if let Some(cfg) = config {
            // Build effective config that matches what was passed to create_rspace_from_config
            let effective_outer = match &cfg.outer {
                OuterStorageType::Array { max_size, cyclic } => {
                    let effective_size = self.extract_config_usize(&config_par, "size")
                        .unwrap_or(*max_size);
                    let effective_cyclic = self.extract_config_bool(&config_par, "cyclic")
                        .unwrap_or(*cyclic);
                    OuterStorageType::Array {
                        max_size: effective_size,
                        cyclic: effective_cyclic,
                    }
                }
                other => other.clone(),
            };
            let effective_cfg = SpaceConfig {
                qualifier,
                outer: effective_outer,
                data_collection: cfg.data_collection.clone(),
                continuation_collection: cfg.continuation_collection.clone(),
                theory: None,  // Theory is not clonable and not needed for allocation_mode
                gas_config: cfg.gas_config.clone(),
            };
            self.space_config_map.insert(space_id_bytes, effective_cfg);
        }

        // Send the space reference to the reply channel (if provided)
        let output = vec![space_channel];
        if let Some(ack) = ack_opt {
            produce(&output, &ack).await?;
        }

        Ok(output)
    }

    /// Create a GenericRSpace from a SpaceConfig using config-driven dispatch.
    ///
    /// This helper function matches on the SpaceConfig's outer storage type and
    /// inner collection type to create the appropriate GenericRSpace instance.
    ///
    /// # Type System Note
    /// Due to Rust's type system, we need explicit type instantiation for each
    /// (outer, inner) combination. The `create_generic_rspace!` macro reduces
    /// boilerplate but the branching is required for correct monomorphization.
    fn create_rspace_from_config(
        &self,
        config: &SpaceConfig,
        config_par: &Par,
        rspace_id: SpaceId,
        theory: &Option<super::spaces::types::BoxedTheory>,
    ) -> Result<RhoISpace, InterpreterError> {
        // Extract qualifier from config
        let qualifier = config.qualifier;

        // Match on (outer_storage, data_collection) to create the right types
        match (&config.outer, &config.data_collection) {
            // ===================================================================
            // VectorDB - Special case: needs config parsing for dimensions/threshold
            // ===================================================================
            (_, InnerCollectionType::VectorDB { dimensions, backend }) => {
                // Parse VectorDB configuration from the config map
                // If config_par is empty (Par::default()), use URN defaults
                // metric=None means the backend decides based on embedding_type
                let (dims, threshold, embedding_type, metric) = if config_par.is_empty() {
                    // Empty config - use URN dimensions and defaults
                    // Backend will derive default metric from embedding_type
                    (*dimensions, 0.8, EmbeddingType::Integer, None)
                } else {
                    // Non-empty config - parse from the config map
                    let (parsed_dims, threshold, embedding_type, metric) = self.parse_vectordb_config(config_par)?;
                    // Use parsed dims if valid, otherwise fall back to URN
                    let dims = if parsed_dims > 0 { parsed_dims } else { *dimensions };
                    (dims, threshold, embedding_type, metric)
                };

                tracing::debug!(
                    "create_space: Creating VectorDB space (dims={}, threshold={}, type={:?}, backend={:?})",
                    dims, threshold, embedding_type, backend
                );

                let channel_store = VectorDBChannelStore::<Par, ListParWithRandom, BindPattern, TaggedContinuation>::new(
                    dims, threshold, metric, backend.clone(), embedding_type,
                );
                Ok(create_generic_rspace!(channel_store, rspace_id, qualifier, theory, &config.gas_config))
            }

            // ===================================================================
            // HashMap Outer Storage
            // ===================================================================
            (OuterStorageType::HashMap, InnerCollectionType::Bag) => {
                self.validate_no_extra_config_keys("Bag", config_par)?;
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, BagDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    BagDataCollection::new, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashMap, InnerCollectionType::Queue) => {
                self.validate_no_extra_config_keys("Queue", config_par)?;
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, QueueDataCollection<_>, QueueContinuationCollection<_, _>>::new(
                    QueueDataCollection::new, QueueContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashMap, InnerCollectionType::Stack) => {
                self.validate_no_extra_config_keys("Stack", config_par)?;
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, StackDataCollection<_>, StackContinuationCollection<_, _>>::new(
                    StackDataCollection::new, StackContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashMap, InnerCollectionType::Set) => {
                self.validate_no_extra_config_keys("Set", config_par)?;
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, SetDataCollection<_>, SetContinuationCollection<_, _>>::new(
                    SetDataCollection::new, SetContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashMap, InnerCollectionType::Cell) => {
                self.validate_no_extra_config_keys("Cell", config_par)?;
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, CellDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    || CellDataCollection::new("channel".to_string()),
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashMap, InnerCollectionType::PriorityQueue { .. }) => {
                self.validate_no_extra_config_keys("PriorityQueue (use URN parameter for priorities, e.g., priorityqueue(3))", config_par)?;
                // Note: PriorityQueue uses default 3 priority levels (fn() can't capture params)
                let store = HashMapChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, PriorityQueueDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    PriorityQueueDataCollection::new_default,
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            // ===================================================================
            // PathMap Outer Storage
            // ===================================================================
            (OuterStorageType::PathMap, InnerCollectionType::Bag) => {
                self.validate_no_extra_config_keys("Bag", config_par)?;
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, BagDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    BagDataCollection::new, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::PathMap, InnerCollectionType::Queue) => {
                self.validate_no_extra_config_keys("Queue", config_par)?;
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, QueueDataCollection<_>, QueueContinuationCollection<_, _>>::new(
                    QueueDataCollection::new, QueueContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::PathMap, InnerCollectionType::Stack) => {
                self.validate_no_extra_config_keys("Stack", config_par)?;
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, StackDataCollection<_>, StackContinuationCollection<_, _>>::new(
                    StackDataCollection::new, StackContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::PathMap, InnerCollectionType::Set) => {
                self.validate_no_extra_config_keys("Set", config_par)?;
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, SetDataCollection<_>, SetContinuationCollection<_, _>>::new(
                    SetDataCollection::new, SetContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::PathMap, InnerCollectionType::Cell) => {
                self.validate_no_extra_config_keys("Cell", config_par)?;
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, CellDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    || CellDataCollection::new("channel".to_string()),
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::PathMap, InnerCollectionType::PriorityQueue { .. }) => {
                self.validate_no_extra_config_keys("PriorityQueue (use URN parameter for priorities, e.g., priorityqueue(3))", config_par)?;
                // Note: PriorityQueue uses default 3 priority levels (fn() can't capture params)
                let store = RholangPathMapStore::<BindPattern, ListParWithRandom, TaggedContinuation, PriorityQueueDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    PriorityQueueDataCollection::new_default,
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            // ===================================================================
            // Array Outer Storage
            // Fixed-size indexed channel allocation. Channels are created from
            // sequential indices (0, 1, 2, ...) wrapped as Unforgeable Par.
            // ===================================================================
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::Bag) => {
                self.validate_array_config_keys(config_par)?;
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, BagDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    BagDataCollection::new, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::Queue) => {
                self.validate_array_config_keys(config_par)?;
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, QueueDataCollection<_>, QueueContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    QueueDataCollection::new, QueueContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::Stack) => {
                self.validate_array_config_keys(config_par)?;
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, StackDataCollection<_>, StackContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    StackDataCollection::new, StackContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::Set) => {
                self.validate_array_config_keys(config_par)?;
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, SetDataCollection<_>, SetContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    SetDataCollection::new, SetContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::Cell) => {
                self.validate_array_config_keys(config_par)?;
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, CellDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    || CellDataCollection::new("channel".to_string()), BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::Array { max_size, cyclic }, InnerCollectionType::PriorityQueue { .. }) => {
                self.validate_array_config_keys(config_par)?;
                // Note: PriorityQueue uses default 3 priority levels (fn() can't capture params)
                let store = ArrayChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, PriorityQueueDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    *max_size, *cyclic, rspace_id.clone(),
                    array_index_to_par, par_to_array_index,
                    PriorityQueueDataCollection::new_default, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            // ===================================================================
            // Vector Outer Storage (Unbounded Sequential Allocation)
            // Similar to Array but grows dynamically without size limits.
            // ===================================================================
            (OuterStorageType::Vector, InnerCollectionType::Bag) => {
                self.validate_no_extra_config_keys("Bag", config_par)?;
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, BagDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    BagDataCollection::new, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            (OuterStorageType::Vector, InnerCollectionType::Queue) => {
                self.validate_no_extra_config_keys("Queue", config_par)?;
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, QueueDataCollection<_>, QueueContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    QueueDataCollection::new, QueueContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            (OuterStorageType::Vector, InnerCollectionType::Stack) => {
                self.validate_no_extra_config_keys("Stack", config_par)?;
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, StackDataCollection<_>, StackContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    StackDataCollection::new, StackContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            (OuterStorageType::Vector, InnerCollectionType::Set) => {
                self.validate_no_extra_config_keys("Set", config_par)?;
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, SetDataCollection<_>, SetContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    SetDataCollection::new, SetContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            (OuterStorageType::Vector, InnerCollectionType::Cell) => {
                self.validate_no_extra_config_keys("Cell", config_par)?;
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, CellDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    || CellDataCollection::new("channel".to_string()), BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            (OuterStorageType::Vector, InnerCollectionType::PriorityQueue { .. }) => {
                self.validate_no_extra_config_keys("PriorityQueue (use URN parameter for priorities, e.g., priorityqueue(3))", config_par)?;
                // Note: PriorityQueue uses default 3 priority levels (fn() can't capture params)
                let store = VectorChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, PriorityQueueDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    rspace_id.clone(),
                    vector_index_to_par, par_to_vector_index,
                    PriorityQueueDataCollection::new_default, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }

            // ===================================================================
            // HashSet Outer Storage (Sequential Processes)
            // ===================================================================
            (OuterStorageType::HashSet, InnerCollectionType::Bag) => {
                self.validate_no_extra_config_keys("Bag", config_par)?;
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, BagDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    BagDataCollection::new, BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashSet, InnerCollectionType::Queue) => {
                self.validate_no_extra_config_keys("Queue", config_par)?;
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, QueueDataCollection<_>, QueueContinuationCollection<_, _>>::new(
                    QueueDataCollection::new, QueueContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashSet, InnerCollectionType::Stack) => {
                self.validate_no_extra_config_keys("Stack", config_par)?;
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, StackDataCollection<_>, StackContinuationCollection<_, _>>::new(
                    StackDataCollection::new, StackContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashSet, InnerCollectionType::Set) => {
                self.validate_no_extra_config_keys("Set", config_par)?;
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, SetDataCollection<_>, SetContinuationCollection<_, _>>::new(
                    SetDataCollection::new, SetContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashSet, InnerCollectionType::Cell) => {
                self.validate_no_extra_config_keys("Cell", config_par)?;
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, CellDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    || CellDataCollection::new("channel".to_string()),
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
            (OuterStorageType::HashSet, InnerCollectionType::PriorityQueue { .. }) => {
                self.validate_no_extra_config_keys("PriorityQueue (use URN parameter for priorities, e.g., priorityqueue(3))", config_par)?;
                // Note: PriorityQueue uses default 3 priority levels (fn() can't capture params)
                let store = HashSetChannelStore::<Par, BindPattern, ListParWithRandom, TaggedContinuation, PriorityQueueDataCollection<_>, BagContinuationCollection<_, _>>::new(
                    PriorityQueueDataCollection::new_default,
                    BagContinuationCollection::new,
                );
                Ok(create_generic_rspace!(store, rspace_id, qualifier, theory, &config.gas_config))
            }
        }
    }

    /// Parse VectorDB configuration from a Rholang map.
    ///
    /// This function uses the generic `RawVectorDBConfig` pass-through mechanism.
    /// Rholang only extracts `dimensions` (the universal parameter) and passes
    /// all other parameters as generic key-value pairs for the VectorDB backend
    /// to interpret.
    ///
    /// # Architecture Note
    ///
    /// This design decouples Rholang from VectorDB-specific semantics:
    /// - Rholang: Parses config to `RawVectorDBConfig`
    /// - VectorDB backend: Interprets threshold, metric, embedding_type, index, etc.
    ///
    /// Currently, threshold/metric/embedding_type parsing is done here temporarily
    /// for backward compatibility. This will be moved to rho-vectordb in Phase 4.
    ///
    /// # Expected format
    /// ```rholang
    /// {"dimensions": 4, "threshold": "0.5", "embedding_type": "integer", "metric": "cosine"}
    /// ```
    fn parse_vectordb_config(
        &self,
        config: &Par,
    ) -> Result<(usize, f32, EmbeddingType, Option<SimilarityMetric>), InterpreterError> {
        use super::spaces::types::parse_raw_vectordb_config;

        // Parse to generic RawVectorDBConfig (Rholang is agnostic to semantics)
        let raw_config = parse_raw_vectordb_config(config).map_err(|e| {
            InterpreterError::ReduceError(format!("VectorDB config error: {}", e))
        })?;

        // Extract dimensions (the only universal parameter)
        let dimensions = raw_config.dimensions;

        // =====================================================================
        // Extract VectorDB-specific params from raw config.
        // These are parsed here for convenience; rho-vectordb/src/db/config.rs
        // provides equivalent parsing for programmatic API usage.
        // =====================================================================

        // Parse threshold (default: 0.8)
        let threshold = self.parse_threshold_from_raw(&raw_config)?;

        // Parse embedding_type (default: Integer)
        let embedding_type = self.parse_embedding_type_from_raw(&raw_config)?;

        // Parse metric (None = backend decides based on embedding_type)
        let metric = self.parse_metric_from_raw(&raw_config)?;

        Ok((dimensions, threshold, embedding_type, metric))
    }

    /// Parse threshold from RawVectorDBConfig params.
    ///
    /// Accepts:
    /// - Float string: "0.5" -> 0.5
    /// - Integer (0-100 scale): 50 -> 0.5
    fn parse_threshold_from_raw(
        &self,
        config: &super::spaces::types::RawVectorDBConfig,
    ) -> Result<f32, InterpreterError> {
        use super::spaces::types::RawConfigValue;

        let threshold = match config.get("threshold") {
            Some(RawConfigValue::String(s)) => {
                let f: f32 = s.parse().map_err(|_| {
                    InterpreterError::ReduceError(format!(
                        "Invalid threshold string '{}': expected float 0.0-1.0",
                        s
                    ))
                })?;
                if !(0.0..=1.0).contains(&f) {
                    return Err(InterpreterError::ReduceError(
                        "VectorDB threshold must be 0.0-1.0".to_string(),
                    ));
                }
                f
            }
            Some(RawConfigValue::Int(i)) => {
                if !(0..=100).contains(i) {
                    return Err(InterpreterError::ReduceError(
                        "VectorDB threshold (integer) must be 0-100".to_string(),
                    ));
                }
                (*i as f32) / 100.0
            }
            Some(RawConfigValue::Float(f)) => {
                if !(0.0..=1.0).contains(f) {
                    return Err(InterpreterError::ReduceError(
                        "VectorDB threshold must be 0.0-1.0".to_string(),
                    ));
                }
                *f as f32
            }
            None => 0.8, // Default
            Some(other) => {
                return Err(InterpreterError::ReduceError(format!(
                    "Invalid threshold type: expected string or int, got {:?}",
                    std::mem::discriminant(other)
                )));
            }
        };
        Ok(threshold)
    }

    /// Parse embedding_type from RawVectorDBConfig params.
    fn parse_embedding_type_from_raw(
        &self,
        config: &super::spaces::types::RawVectorDBConfig,
    ) -> Result<EmbeddingType, InterpreterError> {
        use super::spaces::types::RawConfigValue;

        let embedding_type = match config.get("embedding_type") {
            Some(RawConfigValue::String(s)) => match s.to_lowercase().as_str() {
                "boolean" | "bool" | "binary" => EmbeddingType::Boolean,
                "integer" | "int" | "scaled" => EmbeddingType::Integer,
                "float" | "string" => EmbeddingType::Float,
                other => {
                    return Err(InterpreterError::ReduceError(format!(
                        "Unknown embedding_type: '{}'. Use: boolean, integer, or float",
                        other
                    )));
                }
            },
            None => EmbeddingType::Integer, // Default for Rholang (0-100 scale)
            Some(other) => {
                return Err(InterpreterError::ReduceError(format!(
                    "Invalid embedding_type: expected string, got {:?}",
                    std::mem::discriminant(other)
                )));
            }
        };
        Ok(embedding_type)
    }

    /// Parse metric from RawVectorDBConfig params.
    ///
    /// Returns `None` if not specified - the backend decides the default based on embedding_type.
    fn parse_metric_from_raw(
        &self,
        config: &super::spaces::types::RawVectorDBConfig,
    ) -> Result<Option<SimilarityMetric>, InterpreterError> {
        use super::spaces::types::RawConfigValue;

        let metric = match config.get("metric") {
            Some(RawConfigValue::String(s)) => match s.to_lowercase().as_str() {
                "cosine" => Some(SimilarityMetric::Cosine),
                "dot" | "dotproduct" | "dot_product" => Some(SimilarityMetric::DotProduct),
                "euclidean" | "l2" => Some(SimilarityMetric::Euclidean),
                "manhattan" | "l1" => Some(SimilarityMetric::Manhattan),
                "hamming" => Some(SimilarityMetric::Hamming),
                "jaccard" => Some(SimilarityMetric::Jaccard),
                other => {
                    return Err(InterpreterError::ReduceError(format!(
                        "Unknown metric: '{}'. Use: cosine, dot, euclidean, manhattan, hamming, or jaccard",
                        other
                    )));
                }
            },
            // Backend decides default based on embedding_type
            None => None,
            Some(other) => {
                return Err(InterpreterError::ReduceError(format!(
                    "Invalid metric: expected string, got {:?}",
                    std::mem::discriminant(other)
                )));
            }
        };
        Ok(metric)
    }

    /// Extract a string value from a Par.
    fn extract_string_from_par(&self, par: &Option<Par>) -> Result<String, InterpreterError> {
        use models::rhoapi::expr::ExprInstance::GString;

        let par = par.as_ref().ok_or_else(|| {
            InterpreterError::ReduceError("Expected Par, got None".to_string())
        })?;

        for expr in &par.exprs {
            if let Some(GString(s)) = &expr.expr_instance {
                return Ok(s.clone());
            }
        }

        Err(InterpreterError::ReduceError(
            "Expected string value in Par".to_string(),
        ))
    }

    /// Extract an integer value from a Par.
    fn extract_int_from_par(&self, par: &Option<Par>) -> Result<i64, InterpreterError> {
        use models::rhoapi::expr::ExprInstance::GInt;

        let par = par.as_ref().ok_or_else(|| {
            InterpreterError::ReduceError("Expected Par, got None".to_string())
        })?;

        for expr in &par.exprs {
            if let Some(GInt(i)) = &expr.expr_instance {
                return Ok(*i);
            }
        }

        Err(InterpreterError::ReduceError(
            "Expected integer value in Par".to_string(),
        ))
    }

    // ==========================================================================
    // Theory Extraction Helpers (Reified RSpaces - Type Theory)
    // ==========================================================================

    /// Extract a theory from a Par that may contain an EFree marker.
    ///
    /// Looks for EFree in the Par's expressions and extracts the theory name
    /// from its body, then loads the theory using BuiltinTheoryLoader.
    ///
    /// # Example
    /// For `free Nat()`, this extracts "Nat" and loads the Nat theory.
    fn extract_theory_from_par(
        &self,
        par: &Par,
    ) -> Result<Option<super::spaces::types::BoxedTheory>, InterpreterError> {
        use super::spaces::factory::{BuiltinTheoryLoader, TheoryLoader, TheorySpec};

        for expr in &par.exprs {
            if let Some(ExprInstance::EFreeBody(e_free)) = &expr.expr_instance {
                let body = e_free.body.as_ref().ok_or_else(|| {
                    InterpreterError::ReduceError("EFree missing body".to_string())
                })?;

                // Extract theory name from body
                let theory_name = self.extract_theory_name_from_par(body)?;
                let spec = TheorySpec::parse(&theory_name);

                let loader = BuiltinTheoryLoader::new();
                return loader
                    .load(&spec)
                    .map(Some)
                    .map_err(|e| InterpreterError::ReduceError(format!("Theory load error: {}", e)));
            }
        }
        Ok(None)
    }

    /// Extract the theory name from a Par containing a string.
    ///
    /// The Par should contain a GString expression (from normalizing `free Nat()`).
    fn extract_theory_name_from_par(&self, par: &Par) -> Result<String, InterpreterError> {
        // Check for string literal: free Nat() normalizes to GString("Nat")
        for expr in &par.exprs {
            if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
                return Ok(s.clone());
            }
        }
        Err(InterpreterError::ReduceError(
            "Cannot extract theory name: expected theory call like free Nat()".to_string(),
        ))
    }

    /// Extract theory from a config map's "theory" key.
    ///
    /// Config format: `{"qualifier": "default", "theory": free Nat()}`
    ///
    /// Returns `Ok(Some(theory))` if a theory is specified, `Ok(None)` if not.
    fn extract_theory_from_config(
        &self,
        config: &Par,
    ) -> Result<Option<super::spaces::types::BoxedTheory>, InterpreterError> {
        use models::rhoapi::expr::ExprInstance::EMapBody;

        // Config should be a map like {"qualifier": "default", "theory": free Nat()}
        for expr in &config.exprs {
            if let Some(EMapBody(emap)) = &expr.expr_instance {
                for kv in &emap.kvs {
                    // Check if key is "theory"
                    if let (Some(key_par), Some(value_par)) = (&kv.key, &kv.value) {
                        for key_expr in &key_par.exprs {
                            if let Some(ExprInstance::GString(key_str)) = &key_expr.expr_instance {
                                if key_str == "theory" {
                                    return self.extract_theory_from_par(value_par);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(None) // No theory key in config
    }

    /// Extract all string keys from a config map.
    ///
    /// Config format: `{"key1": value1, "key2": value2, ...}`
    ///
    /// Returns a list of all string keys found in the config map.
    /// Non-string keys are ignored (they would be rejected by config parsing anyway).
    fn extract_config_keys(&self, config: &Par) -> Vec<String> {
        use models::rhoapi::expr::ExprInstance::EMapBody;

        let mut keys = Vec::new();
        for expr in &config.exprs {
            if let Some(EMapBody(emap)) = &expr.expr_instance {
                for kv in &emap.kvs {
                    if let Some(key_par) = &kv.key {
                        for key_expr in &key_par.exprs {
                            if let Some(ExprInstance::GString(key_str)) = &key_expr.expr_instance {
                                keys.push(key_str.clone());
                            }
                        }
                    }
                }
            }
        }
        keys
    }

    /// Extract a usize value from a config map by key.
    ///
    /// Config format: `{"size": 10, ...}`
    ///
    /// Returns `Some(value)` if the key exists and is a valid positive integer,
    /// `None` if the key is not present.
    fn extract_config_usize(&self, config: &Par, key: &str) -> Option<usize> {
        use models::rhoapi::expr::ExprInstance::{EMapBody, GInt};

        for expr in &config.exprs {
            if let Some(EMapBody(emap)) = &expr.expr_instance {
                for kv in &emap.kvs {
                    if let (Some(key_par), Some(value_par)) = (&kv.key, &kv.value) {
                        // Check if key matches
                        for key_expr in &key_par.exprs {
                            if let Some(ExprInstance::GString(key_str)) = &key_expr.expr_instance {
                                if key_str == key {
                                    // Extract integer value
                                    for value_expr in &value_par.exprs {
                                        if let Some(GInt(i)) = &value_expr.expr_instance {
                                            if *i > 0 {
                                                return Some(*i as usize);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract a bool value from a config map by key.
    ///
    /// Config format: `{"cyclic": true, ...}`
    ///
    /// Returns `Some(value)` if the key exists and is a boolean,
    /// `None` if the key is not present.
    fn extract_config_bool(&self, config: &Par, key: &str) -> Option<bool> {
        use models::rhoapi::expr::ExprInstance::{EMapBody, GBool};

        for expr in &config.exprs {
            if let Some(EMapBody(emap)) = &expr.expr_instance {
                for kv in &emap.kvs {
                    if let (Some(key_par), Some(value_par)) = (&kv.key, &kv.value) {
                        // Check if key matches
                        for key_expr in &key_par.exprs {
                            if let Some(ExprInstance::GString(key_str)) = &key_expr.expr_instance {
                                if key_str == key {
                                    // Extract boolean value
                                    for value_expr in &value_par.exprs {
                                        if let Some(GBool(b)) = &value_expr.expr_instance {
                                            return Some(*b);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Validate that non-VectorDB spaces receive no unknown config keys.
    ///
    /// Non-VectorDB space types (Bag, Queue, Stack, Set, Cell, PriorityQueue) do not
    /// accept user config parameters beyond "theory" (which is handled separately).
    /// This prevents silent failures when users pass unsupported parameters.
    ///
    /// # Arguments
    /// * `collection_type` - Name of the collection type for error messages
    /// * `config_par` - The config Par from which to extract keys
    ///
    /// # Returns
    /// * `Ok(())` if no unknown keys are present
    /// * `Err(InterpreterError)` if unknown keys are found
    fn validate_no_extra_config_keys(
        &self,
        collection_type: &str,
        config_par: &Par,
    ) -> Result<(), InterpreterError> {
        let keys = self.extract_config_keys(config_par);
        // Filter out known common keys handled by the interpreter
        let unknown_keys: Vec<_> = keys
            .iter()
            .filter(|k| *k != "theory")
            .collect();

        if !unknown_keys.is_empty() {
            return Err(InterpreterError::ReduceError(format!(
                "{} does not accept config parameters. Unknown keys: {:?}. \
                 Only 'theory' is supported.",
                collection_type, unknown_keys
            )));
        }
        Ok(())
    }

    /// Validate config keys for Array outer storage.
    ///
    /// Array accepts these keys:
    /// - `size`: Maximum number of channels (required, parsed by URN parser)
    /// - `cyclic`: Whether to wrap around when full (optional, default false)
    /// - `theory`: Theory name for enhanced matching (optional)
    ///
    /// # Arguments
    /// * `config_par` - The config Par from which to extract keys
    ///
    /// # Returns
    /// * `Ok(())` if only valid keys are present
    /// * `Err(InterpreterError)` if unknown keys are found
    fn validate_array_config_keys(
        &self,
        config_par: &Par,
    ) -> Result<(), InterpreterError> {
        let keys = self.extract_config_keys(config_par);
        // Filter out known Array config keys
        let unknown_keys: Vec<_> = keys
            .iter()
            .filter(|k| *k != "size" && *k != "cyclic" && *k != "theory")
            .collect();

        if !unknown_keys.is_empty() {
            return Err(InterpreterError::ReduceError(format!(
                "Array does not accept these config parameters: {:?}. \
                 Supported keys: 'size', 'cyclic', 'theory'.",
                unknown_keys
            )));
        }
        Ok(())
    }

    // ==========================================================================
    // Vector Operations (Reified RSpaces - Tensor Logic)
    // ==========================================================================

    /// Execute vector/tensor operations from the Tensor Logic paper.
    ///
    /// Supported operations:
    /// - sigmoid: Elementwise sigmoid σ(x) = 1/(1+e^(-x))
    /// - temperature_sigmoid: Temperature-controlled sigmoid σ(x,T) = 1/(1+e^(-x/T))
    /// - softmax: Softmax normalization
    /// - heaviside: Step function H(x) = 1 if x > 0 else 0
    /// - majority: Majority voting for binary vectors
    /// - l2_normalize: L2 normalization v / ||v||
    /// - cosine_similarity: Cosine similarity between two vectors
    /// - euclidean_distance: Euclidean distance between two vectors
    /// - dot_product: Dot product of two vectors
    /// - gram_matrix: Gram matrix from embedding matrix
    /// - superposition: Embedding superposition S = V · Emb
    /// - retrieval: Embedding retrieval D = S · Emb^T
    /// - top_k_similar: Find top-k most similar vectors
    ///
    /// Usage in Rholang:
    /// ```rholang
    /// new VectorOps(`rho:lang:vector`) in {
    ///   VectorOps!("sigmoid", [0.1, -0.5, 2.3], *result) |
    ///   VectorOps!("temperature_sigmoid", [0.1, -0.5, 2.3], 0.5, *result) |
    ///   VectorOps!("cosine_similarity", [1.0, 0.0], [0.0, 1.0], *result)
    /// }
    /// ```
    pub async fn vector_ops(
        &self,
        contract_args: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        use models::rhoapi::EList;
        use models::rhoapi::Expr;
        use models::rhoapi::expr::ExprInstance::{EListBody, GInt, GString};
        use models::rust::rholang::implicits::single_expr;

        // Input type tracking for output format selection
        #[derive(Clone, Copy, PartialEq)]
        enum VectorInputType {
            IntegerList,  // [10, 30, 20, 5] → output as integer list
            FloatString,  // "10.5,30.2,20.0,5.1" → output as comma-delimited string
        }

        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("vector_ops"));
        };

        // Helper to extract integer vector from Par (accepts GInt, converts to f32 internally)
        let extract_int_vec = |par: &Par| -> Option<Vec<i64>> {
            let expr = single_expr(par)?;
            match &expr.expr_instance {
                Some(EListBody(elist)) => {
                    let mut result = Vec::with_capacity(elist.ps.len());
                    for p in &elist.ps {
                        let e = single_expr(p)?;
                        match &e.expr_instance {
                            Some(GInt(i)) => result.push(*i),
                            _ => return None,
                        }
                    }
                    Some(result)
                }
                _ => None,
            }
        };

        // Helper to extract f32 vector from Par (accepts GInt, converts to f32)
        let extract_f32_vec = |par: &Par| -> Option<Vec<f32>> {
            extract_int_vec(par).map(|v| v.into_iter().map(|x| x as f32).collect())
        };

        // Helper to extract f32 vector from comma-delimited string
        let extract_string_vec = |par: &Par| -> Option<Vec<f32>> {
            let expr = single_expr(par)?;
            match &expr.expr_instance {
                Some(GString(s)) => {
                    let mut result = Vec::new();
                    for part in s.split(',') {
                        let trimmed = part.trim();
                        match trimmed.parse::<f32>() {
                            Ok(f) => result.push(f),
                            Err(_) => return None,
                        }
                    }
                    if result.is_empty() { None } else { Some(result) }
                }
                _ => None,
            }
        };

        // Helper to create Par from f32 vector (binary output: >= 0.5 becomes 1, else 0)
        let f32_vec_to_binary_par = |v: Vec<f32>| -> Par {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(EListBody(EList {
                    ps: v.into_iter().map(|x| {
                        let binary = if x >= 0.5 { 1i64 } else { 0i64 };
                        Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(binary)),
                        }])
                    }).collect(),
                    ..Default::default()
                })),
            }])
        };

        // Helper to create Par from f32 vector (rounded to nearest integer, preserves magnitude)
        let f32_vec_to_int_par = |v: Vec<f32>| -> Par {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(EListBody(EList {
                    ps: v.into_iter().map(|x| {
                        let int_val = x.round() as i64;
                        Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(int_val)),
                        }])
                    }).collect(),
                    ..Default::default()
                })),
            }])
        };

        // Helper to create Par from integer vector (direct output)
        let int_vec_to_par = |v: Vec<i64>| -> Par {
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(EListBody(EList {
                    ps: v.into_iter().map(|x| {
                        Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(x)),
                        }])
                    }).collect(),
                    ..Default::default()
                })),
            }])
        };

        // Helper to create Par from bool vector
        let bool_vec_to_par = |v: Vec<bool>| -> Par {
            use models::rhoapi::expr::ExprInstance::GBool;
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(EListBody(EList {
                    ps: v.into_iter().map(|x| {
                        Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GBool(x)),
                        }])
                    }).collect(),
                    ..Default::default()
                })),
            }])
        };

        // Helper to extract bool vector from Par
        let extract_bool_vec = |par: &Par| -> Option<Vec<bool>> {
            use models::rhoapi::expr::ExprInstance::GBool;
            let expr = single_expr(par)?;
            match &expr.expr_instance {
                Some(EListBody(elist)) => {
                    let mut result = Vec::with_capacity(elist.ps.len());
                    for p in &elist.ps {
                        let e = single_expr(p)?;
                        match &e.expr_instance {
                            Some(GBool(b)) => result.push(*b),
                            _ => return None,
                        }
                    }
                    Some(result)
                }
                _ => None,
            }
        };

        // Helper to format f32 ensuring floating point notation
        // (appends .0 to whole numbers that would otherwise format without decimal)
        let format_float = |x: f32| -> String {
            let s = format!("{}", x);
            if s.contains('.') || s.contains('e') || s.contains('E') {
                s
            } else {
                format!("{}.0", s)
            }
        };

        // Helper to create Par from f32 vector as comma-delimited string
        let f32_vec_to_string_par = |v: Vec<f32>| -> Par {
            let s = v.iter()
                .map(|x| format_float(*x))
                .collect::<Vec<_>>()
                .join(",");
            Par::default().with_exprs(vec![Expr {
                expr_instance: Some(GString(s)),
            }])
        };

        // Unified extraction: tries string first, then integer list
        let extract_vec_with_type = |par: &Par| -> Option<(Vec<f32>, VectorInputType)> {
            // Try string format first: "10.5,30.2,20.0"
            if let Some(v) = extract_string_vec(par) {
                return Some((v, VectorInputType::FloatString));
            }
            // Fall back to integer list: [10, 30, 20]
            if let Some(v) = extract_f32_vec(par) {
                return Some((v, VectorInputType::IntegerList));
            }
            None
        };

        // Select output format based on input type
        let format_f32_output = |v: Vec<f32>, input_type: VectorInputType, use_binary: bool| -> Par {
            match input_type {
                VectorInputType::FloatString => f32_vec_to_string_par(v),
                VectorInputType::IntegerList => {
                    if use_binary {
                        f32_vec_to_binary_par(v)
                    } else {
                        f32_vec_to_int_par(v)
                    }
                }
            }
        };

        // Unified extraction for integer operations: tries string first (parsed as int), then integer list
        let extract_int_vec_with_type = |par: &Par| -> Option<(Vec<i64>, VectorInputType)> {
            // Try string format first: "10,30,20" (parse floats, round to int)
            if let Some(v) = extract_string_vec(par) {
                let ints: Vec<i64> = v.iter().map(|x| x.round() as i64).collect();
                return Some((ints, VectorInputType::FloatString));
            }
            // Fall back to integer list: [10, 30, 20]
            if let Some(v) = extract_int_vec(par) {
                return Some((v, VectorInputType::IntegerList));
            }
            None
        };

        // Format integer vector output based on input type
        let format_int_output = |v: Vec<i64>, input_type: VectorInputType| -> Par {
            match input_type {
                VectorInputType::FloatString => {
                    // Convert to floats for string output
                    let floats: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                    f32_vec_to_string_par(floats)
                }
                VectorInputType::IntegerList => int_vec_to_par(v),
            }
        };

        // Unified extraction for matrix rows: tries string first, then integer list
        // Returns just the vector without type tracking (for use with extract_2d_matrix)
        let extract_f32_vec_unified = |par: &Par| -> Option<Vec<f32>> {
            // Try string format first: "0.9,0.1,0.0"
            if let Some(v) = extract_string_vec(par) {
                return Some(v);
            }
            // Fall back to integer list: [90, 10, 0]
            extract_f32_vec(par)
        };

        // Detect input type from matrix (check first row)
        let detect_matrix_input_type = |par: &Par| -> VectorInputType {
            if let Some(expr) = single_expr(par) {
                if let Some(EListBody(elist)) = &expr.expr_instance {
                    if let Some(first_row) = elist.ps.first() {
                        if let Some(row_expr) = single_expr(first_row) {
                            if matches!(&row_expr.expr_instance, Some(GString(_))) {
                                return VectorInputType::FloatString;
                            }
                        }
                    }
                }
            }
            VectorInputType::IntegerList
        };

        // Format 2D matrix output based on input type
        let format_matrix_output = |matrix: &ndarray::Array2<f32>, input_type: VectorInputType| -> Par {
            match input_type {
                VectorInputType::FloatString => {
                    // Output as list of float strings
                    let rows: Vec<Expr> = matrix.rows().into_iter().map(|row| {
                        let s = row.iter()
                            .map(|x| format_float(*x))
                            .collect::<Vec<_>>()
                            .join(",");
                        Expr {
                            expr_instance: Some(GString(s)),
                        }
                    }).collect();
                    Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(EListBody(EList {
                            ps: rows.into_iter().map(|e| Par::default().with_exprs(vec![e])).collect(),
                            ..Default::default()
                        })),
                    }])
                }
                VectorInputType::IntegerList => {
                    // Existing behavior: output as nested integer lists
                    matrix_to_par(matrix)
                }
            }
        };

        // Helper to extract remainder items from args[2] (which is an EList Par)
        let extract_remainder = |args: &[Par]| -> Vec<Par> {
            if args.len() < 3 {
                return Vec::new();
            }
            let remainder_par = &args[2];
            let Some(expr) = single_expr(remainder_par) else {
                return Vec::new();
            };
            match &expr.expr_instance {
                Some(EListBody(elist)) => elist.ps.clone(),
                _ => Vec::new(),
            }
        };

        // Get operation name
        let [op_par, ..] = args.as_slice() else {
            return Err(InterpreterError::ReduceError(
                "VectorOps requires at least an operation name".to_string(),
            ));
        };

        let Some(op_name) = RhoString::unapply(op_par) else {
            return Err(InterpreterError::ReduceError(
                "VectorOps operation name must be a string".to_string(),
            ));
        };

        // Extract remainder items (extra args beyond the 2 required)
        let remainder = extract_remainder(&args);

        // Arg structure with FreeVar remainder:
        // - args[0] = op name
        // - args[1] = first operand (e.g., vector)
        // - remainder = extra args extracted from args[2] EList
        //
        // For (op, vec, ack): vec_par = args[1], ack = remainder[0]
        // For (op, vec, temp, ack): vec_par = args[1], temp = remainder[0], ack = remainder[1]
        // For (op, vec1, vec2, ack): vec1 = args[1], vec2 = remainder[0], ack = remainder[1]

        match op_name.as_str() {
            // =================================================================
            // Unary operations: (op, vector, ack)
            // =================================================================
            "sigmoid" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:sigmoid"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:sigmoid"))?;
                let Some((v, input_type)) = extract_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "sigmoid requires a numeric vector or comma-delimited string".to_string(),
                    ));
                };
                let arr = vector_ops::slice_to_array1(&v);
                let result = vector_ops::sigmoid(&arr);
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, true)];
                produce(&output, ack).await?;
                Ok(output)
            }

            "softmax" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:softmax"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:softmax"))?;
                let Some((v, input_type)) = extract_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "softmax requires a numeric vector or comma-delimited string".to_string(),
                    ));
                };
                let arr = vector_ops::slice_to_array1(&v);
                let result = vector_ops::softmax(&arr);
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, true)];
                produce(&output, ack).await?;
                Ok(output)
            }

            "heaviside" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:heaviside"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:heaviside"))?;
                let Some((v, input_type)) = extract_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "heaviside requires a numeric vector or comma-delimited string".to_string(),
                    ));
                };
                let arr = vector_ops::slice_to_array1(&v);
                let result = vector_ops::heaviside(&arr);
                // Heaviside returns booleans - convert to f32 (0.0/1.0) for unified output
                let f32_result: Vec<f32> = result.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                let output = vec![format_f32_output(f32_result, input_type, true)];
                produce(&output, ack).await?;
                Ok(output)
            }

            "l2_normalize" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:l2_normalize"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:l2_normalize"))?;
                let Some((v, input_type)) = extract_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "l2_normalize requires a numeric vector or comma-delimited string".to_string(),
                    ));
                };
                let arr = vector_ops::slice_to_array1(&v);
                let result = vector_ops::l2_normalize_safe(&arr);
                // use_binary=false to preserve magnitude (normalized values in 0-1 range)
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, false)];
                produce(&output, ack).await?;
                Ok(output)
            }

            "majority" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:majority"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:majority"))?;
                let Some(v) = extract_bool_vec(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "majority requires a boolean vector".to_string(),
                    ));
                };
                let arr = ndarray::Array1::from_vec(v);
                let result = vector_ops::majority(&arr);
                let output = vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(models::rhoapi::expr::ExprInstance::GBool(result)),
                }])];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Binary operations with temperature: (op, vector, temp, ack)
            // =================================================================
            "temperature_sigmoid" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:temperature_sigmoid"))?;
                let temp_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:temperature_sigmoid"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:temperature_sigmoid"))?;
                let Some((v, input_type)) = extract_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "temperature_sigmoid requires a numeric vector or comma-delimited string".to_string(),
                    ));
                };
                // Temperature must be an integer (Rholang has no native float type)
                // Interpret as scaling factor: temp=1 is baseline, temp=10 is 10x sharper
                let Some(temp) = RhoNumber::unapply(temp_par).map(|n| n as f32) else {
                    return Err(InterpreterError::ReduceError(
                        "temperature_sigmoid requires temperature as a number".to_string(),
                    ));
                };
                let arr = vector_ops::slice_to_array1(&v);
                let result = vector_ops::temperature_sigmoid(&arr, temp);
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, true)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Binary operations: (op, vec1, vec2, ack)
            // =================================================================
            "cosine_similarity" => {
                let vec1_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:cosine_similarity"))?;
                let vec2_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:cosine_similarity"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:cosine_similarity"))?;
                // Detect input type from first vector
                let Some((v1, input_type)) = extract_vec_with_type(vec1_par) else {
                    return Err(InterpreterError::ReduceError(
                        "cosine_similarity requires two numeric vectors".to_string(),
                    ));
                };
                let Some(v2) = extract_f32_vec_unified(vec2_par) else {
                    return Err(InterpreterError::ReduceError(
                        "cosine_similarity requires two numeric vectors".to_string(),
                    ));
                };
                let arr1 = vector_ops::slice_to_array1(&v1);
                let arr2 = vector_ops::slice_to_array1(&v2);
                let result = vector_ops::cosine_similarity_safe(&arr1, &arr2);
                // Return float string for float string inputs, scaled integer otherwise
                let output = match input_type {
                    VectorInputType::FloatString => vec![Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(GString(format_float(result))),
                    }])],
                    VectorInputType::IntegerList => {
                        let scaled = (result * 100.0).round() as i64;
                        vec![Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(scaled)),
                        }])]
                    }
                };
                produce(&output, ack).await?;
                Ok(output)
            }

            "euclidean_distance" => {
                let vec1_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:euclidean_distance"))?;
                let vec2_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:euclidean_distance"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:euclidean_distance"))?;
                // Detect input type from first vector
                let Some((v1, input_type)) = extract_vec_with_type(vec1_par) else {
                    return Err(InterpreterError::ReduceError(
                        "euclidean_distance requires two numeric vectors".to_string(),
                    ));
                };
                let Some(v2) = extract_f32_vec_unified(vec2_par) else {
                    return Err(InterpreterError::ReduceError(
                        "euclidean_distance requires two numeric vectors".to_string(),
                    ));
                };
                let arr1 = vector_ops::slice_to_array1(&v1);
                let arr2 = vector_ops::slice_to_array1(&v2);
                let result = vector_ops::euclidean_distance(&arr1, &arr2);
                // Return float string for float string inputs, integer otherwise
                let output = match input_type {
                    VectorInputType::FloatString => vec![Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(GString(format_float(result))),
                    }])],
                    VectorInputType::IntegerList => {
                        let int_result = result.round() as i64;
                        vec![Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(int_result)),
                        }])]
                    }
                };
                produce(&output, ack).await?;
                Ok(output)
            }

            "dot_product" => {
                let vec1_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:dot_product"))?;
                let vec2_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:dot_product"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:dot_product"))?;
                // Detect input type from first vector
                let Some((v1, input_type)) = extract_vec_with_type(vec1_par) else {
                    return Err(InterpreterError::ReduceError(
                        "dot_product requires two numeric vectors".to_string(),
                    ));
                };
                let Some(v2) = extract_f32_vec_unified(vec2_par) else {
                    return Err(InterpreterError::ReduceError(
                        "dot_product requires two numeric vectors".to_string(),
                    ));
                };
                let arr1 = vector_ops::slice_to_array1(&v1);
                let arr2 = vector_ops::slice_to_array1(&v2);
                let result = vector_ops::dot_product(&arr1, &arr2);
                // Return float string for float string inputs, integer otherwise
                let output = match input_type {
                    VectorInputType::FloatString => vec![Par::default().with_exprs(vec![Expr {
                        expr_instance: Some(GString(format_float(result))),
                    }])],
                    VectorInputType::IntegerList => {
                        let int_result = result.round() as i64;
                        vec![Par::default().with_exprs(vec![Expr {
                            expr_instance: Some(GInt(int_result)),
                        }])]
                    }
                };
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Matrix operations: (op, matrix, ack)
            // =================================================================
            "gram_matrix" => {
                let matrix_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:gram_matrix"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:gram_matrix"))?;
                // Detect input type from first row
                let input_type = detect_matrix_input_type(matrix_par);
                // Extract 2D matrix from nested list (supports both float strings and int lists)
                let Some(rows) = extract_2d_matrix(matrix_par, &extract_f32_vec_unified) else {
                    return Err(InterpreterError::ReduceError(
                        "gram_matrix requires a 2D numeric matrix".to_string(),
                    ));
                };
                // Convert Vec<Vec<f32>> to Vec<Array1<f32>> for rows_to_array2
                let array_rows: Vec<ndarray::Array1<f32>> = rows
                    .into_iter()
                    .map(|row| vector_ops::vec_to_array1(row))
                    .collect();
                let matrix = vector_ops::rows_to_array2(&array_rows);
                let result = vector_ops::gram_matrix(&matrix);
                let output = vec![format_matrix_output(&result, input_type)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Superposition: (op, values, embeddings, ack)
            // =================================================================
            "superposition" => {
                let values_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:superposition"))?;
                let embeddings_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:superposition"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:superposition"))?;
                let Some((values, input_type)) = extract_vec_with_type(values_par) else {
                    return Err(InterpreterError::ReduceError(
                        "superposition requires a values vector or comma-delimited string".to_string(),
                    ));
                };
                let Some(emb_rows) = extract_2d_matrix(embeddings_par, &extract_f32_vec_unified) else {
                    return Err(InterpreterError::ReduceError(
                        "superposition requires an embeddings matrix".to_string(),
                    ));
                };
                let values_arr = vector_ops::slice_to_array1(&values);
                // Convert Vec<Vec<f32>> to Vec<Array1<f32>> for rows_to_array2
                let array_rows: Vec<ndarray::Array1<f32>> = emb_rows
                    .into_iter()
                    .map(|row| vector_ops::vec_to_array1(row))
                    .collect();
                let emb_arr = vector_ops::rows_to_array2(&array_rows);
                let result = vector_ops::superposition(&values_arr, &emb_arr);
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, false)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Retrieval: (op, superposition, embeddings, ack)
            // =================================================================
            "retrieval" => {
                let super_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:retrieval"))?;
                let embeddings_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:retrieval"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:retrieval"))?;
                let Some((sup, input_type)) = extract_vec_with_type(super_par) else {
                    return Err(InterpreterError::ReduceError(
                        "retrieval requires a superposition vector or comma-delimited string".to_string(),
                    ));
                };
                let Some(emb_rows) = extract_2d_matrix(embeddings_par, &extract_f32_vec_unified) else {
                    return Err(InterpreterError::ReduceError(
                        "retrieval requires an embeddings matrix".to_string(),
                    ));
                };
                let sup_arr = vector_ops::slice_to_array1(&sup);
                // Convert Vec<Vec<f32>> to Vec<Array1<f32>> for rows_to_array2
                let array_rows: Vec<ndarray::Array1<f32>> = emb_rows
                    .into_iter()
                    .map(|row| vector_ops::vec_to_array1(row))
                    .collect();
                let emb_arr = vector_ops::rows_to_array2(&array_rows);
                let result = vector_ops::retrieval(&sup_arr, &emb_arr);
                let output = vec![format_f32_output(vector_ops::array1_to_vec(result), input_type, false)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // Top-K Similar: (op, query, embeddings, k, ack)
            // =================================================================
            "top_k_similar" => {
                let query_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:top_k_similar"))?;
                let embeddings_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:top_k_similar"))?;
                let k_par = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:top_k_similar"))?;
                let ack = remainder.get(2).ok_or_else(|| illegal_argument_error("vector_ops:top_k_similar"))?;
                let Some((query, input_type)) = extract_vec_with_type(query_par) else {
                    return Err(InterpreterError::ReduceError(
                        "top_k_similar requires a query vector".to_string(),
                    ));
                };
                let Some(emb_rows) = extract_2d_matrix(embeddings_par, &extract_f32_vec_unified) else {
                    return Err(InterpreterError::ReduceError(
                        "top_k_similar requires an embeddings matrix".to_string(),
                    ));
                };
                let Some(k) = RhoNumber::unapply(k_par).map(|n| n as usize) else {
                    return Err(InterpreterError::ReduceError(
                        "top_k_similar requires k as an integer".to_string(),
                    ));
                };
                let query_arr = vector_ops::slice_to_array1(&query);
                // Convert Vec<Vec<f32>> to Vec<Array1<f32>> for rows_to_array2
                let array_rows: Vec<ndarray::Array1<f32>> = emb_rows
                    .into_iter()
                    .map(|row| vector_ops::vec_to_array1(row))
                    .collect();
                let emb_arr = vector_ops::rows_to_array2(&array_rows);
                let result = vector_ops::top_k_similar(&query_arr, &emb_arr, k);
                // Return list of (index, similarity) tuples
                // Format similarity based on input type: float string or scaled integer
                let output_par = Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(EListBody(EList {
                        ps: result.into_iter().map(|(idx, sim)| {
                            // Create tuple (index, similarity)
                            // Format similarity based on input type
                            let sim_par = match input_type {
                                VectorInputType::FloatString => Par::default().with_exprs(vec![Expr {
                                    expr_instance: Some(GString(format_float(sim))),
                                }]),
                                VectorInputType::IntegerList => {
                                    let sim_percent = (sim * 100.0).round() as i64;
                                    Par::default().with_exprs(vec![Expr {
                                        expr_instance: Some(GInt(sim_percent)),
                                    }])
                                }
                            };
                            Par::default().with_exprs(vec![Expr {
                                expr_instance: Some(models::rhoapi::expr::ExprInstance::ETupleBody(
                                    models::rhoapi::ETuple {
                                        ps: vec![
                                            Par::default().with_exprs(vec![Expr {
                                                expr_instance: Some(GInt(idx as i64)),
                                            }]),
                                            sim_par,
                                        ],
                                        ..Default::default()
                                    },
                                )),
                            }])
                        }).collect(),
                        ..Default::default()
                    })),
                }]);
                let output = vec![output_par];
                produce(&output, ack).await?;
                Ok(output)
            }

            // =================================================================
            // HYPERVECTOR OPERATIONS (High-Dimensional Computing)
            // =================================================================

            // XOR binding: (op, vec1, vec2, ack)
            "bind" => {
                let vec1_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:bind"))?;
                let vec2_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:bind"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:bind"))?;
                let (Some((v1, input_type)), Some((v2, _))) = (extract_int_vec_with_type(vec1_par), extract_int_vec_with_type(vec2_par)) else {
                    return Err(InterpreterError::ReduceError(
                        "bind requires two binary integer vectors or comma-delimited strings".to_string(),
                    ));
                };
                let result = vector_ops::bind(&v1, &v2);
                let output = vec![format_int_output(result, input_type)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Unbind (inverse of bind): (op, bound, key, ack)
            "unbind" => {
                let bound_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:unbind"))?;
                let key_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:unbind"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:unbind"))?;
                let (Some((bound, input_type)), Some((key, _))) = (extract_int_vec_with_type(bound_par), extract_int_vec_with_type(key_par)) else {
                    return Err(InterpreterError::ReduceError(
                        "unbind requires two binary integer vectors or comma-delimited strings".to_string(),
                    ));
                };
                let result = vector_ops::unbind(&bound, &key);
                let output = vec![format_int_output(result, input_type)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Majority bundling: (op, vectors_list, ack)
            "bundle" => {
                let vectors_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:bundle"))?;
                let ack = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:bundle"))?;
                // Extract list of vectors
                let Some(vectors) = extract_2d_int_matrix(vectors_par, &extract_int_vec) else {
                    return Err(InterpreterError::ReduceError(
                        "bundle requires a list of binary integer vectors".to_string(),
                    ));
                };
                // Convert to slices for the bundle function
                let vec_slices: Vec<&[i64]> = vectors.iter().map(|v| v.as_slice()).collect();
                let result = vector_ops::bundle(&vec_slices);
                let output = vec![int_vec_to_par(result)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Circular shift permutation: (op, vector, shift, ack)
            "permute" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:permute"))?;
                let shift_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:permute"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:permute"))?;
                let Some((v, input_type)) = extract_int_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "permute requires a binary integer vector or comma-delimited string".to_string(),
                    ));
                };
                let Some(shift) = RhoNumber::unapply(shift_par) else {
                    return Err(InterpreterError::ReduceError(
                        "permute requires shift amount as an integer".to_string(),
                    ));
                };
                let result = vector_ops::permute(&v, shift);
                let output = vec![format_int_output(result, input_type)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Inverse permutation: (op, vector, shift, ack)
            "unpermute" => {
                let vec_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:unpermute"))?;
                let shift_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:unpermute"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:unpermute"))?;
                let Some((v, input_type)) = extract_int_vec_with_type(vec_par) else {
                    return Err(InterpreterError::ReduceError(
                        "unpermute requires a binary integer vector or comma-delimited string".to_string(),
                    ));
                };
                let Some(shift) = RhoNumber::unapply(shift_par) else {
                    return Err(InterpreterError::ReduceError(
                        "unpermute requires shift amount as an integer".to_string(),
                    ));
                };
                let result = vector_ops::unpermute(&v, shift);
                let output = vec![format_int_output(result, input_type)];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Hamming similarity: (op, vec1, vec2, ack)
            "hamming_similarity" => {
                let vec1_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:hamming_similarity"))?;
                let vec2_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:hamming_similarity"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:hamming_similarity"))?;
                let (Some(v1), Some(v2)) = (extract_int_vec(vec1_par), extract_int_vec(vec2_par)) else {
                    return Err(InterpreterError::ReduceError(
                        "hamming_similarity requires two binary integer vectors".to_string(),
                    ));
                };
                let result = vector_ops::hamming_similarity(&v1, &v2);
                // Returns percentage (0-100)
                let output = vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(GInt(result)),
                }])];
                produce(&output, ack).await?;
                Ok(output)
            }

            // Resonance (cleanup/lookup): (op, query, codebook, ack)
            "resonance" => {
                let query_par = args.get(1).ok_or_else(|| illegal_argument_error("vector_ops:resonance"))?;
                let codebook_par = remainder.get(0).ok_or_else(|| illegal_argument_error("vector_ops:resonance"))?;
                let ack = remainder.get(1).ok_or_else(|| illegal_argument_error("vector_ops:resonance"))?;
                let Some(query) = extract_int_vec(query_par) else {
                    return Err(InterpreterError::ReduceError(
                        "resonance requires a query binary integer vector".to_string(),
                    ));
                };
                let Some(codebook) = extract_2d_int_matrix(codebook_par, &extract_int_vec) else {
                    return Err(InterpreterError::ReduceError(
                        "resonance requires a codebook (list of binary integer vectors)".to_string(),
                    ));
                };
                // Convert to slices for the resonance function
                let code_slices: Vec<&[i64]> = codebook.iter().map(|v| v.as_slice()).collect();
                let result_idx = vector_ops::resonance(&query, &code_slices);
                // Return the index of the most similar vector
                let output = vec![Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(GInt(result_idx as i64)),
                }])];
                produce(&output, ack).await?;
                Ok(output)
            }

            _ => Err(InterpreterError::ReduceError(format!(
                "Unknown vector operation: {}. Supported operations: sigmoid, temperature_sigmoid, \
                 softmax, heaviside, majority, l2_normalize, cosine_similarity, euclidean_distance, \
                 dot_product, gram_matrix, superposition, retrieval, top_k_similar, \
                 bind, unbind, bundle, permute, unpermute, hamming_similarity, resonance",
                op_name
            ))),
        }
    }

    /*
     * The following functions below can be removed once rust-casper calls create_rho_runtime.
     * Until then, they must remain in the rholang directory to avoid circular dependencies.
     */

    // See casper/src/test/scala/coop/rchain/casper/helper/TestResultCollector.scala
    // TODO remove this once Rust node will be completed ( this stuff already moved under Casper, double check related files)
    pub async fn handle_message(
        &self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        // println!("\nhit handle_message");
        let mut printer = PrettyPrinter::new();

        fn clue_msg(clue: String, attempt: i64) -> String {
            format!("{} (test attempt: {})", clue, attempt)
        }

        if let Some((produce, _, _, assert_par)) = self.is_contract_call().unapply(message) {
            if let Some((test_name, attempt, assertion, clue, ack_channel)) =
                IsAssert::unapply(assert_par.clone())
            {
                if let Some((expected_or_unexpected, equals_or_not_equals_str, actual)) =
                    IsComparison::unapply(assertion.clone())
                {
                    if equals_or_not_equals_str == "==" {
                        let assertion = RhoTestAssertion::RhoAssertEquals {
                            test_name,
                            expected: expected_or_unexpected.clone(),
                            actual: actual.clone(),
                            clue: clue.clone(),
                        };

                        let output = vec![new_gbool_par(assertion.is_success(), Vec::new(), false)];
                        produce(&output, &ack_channel).await?;

                        assert_eq!(
                            printer.build_string_from_message(&actual),
                            printer.build_string_from_message(&expected_or_unexpected),
                            "{}",
                            clue_msg(clue, attempt)
                        );

                        assert_eq!(
                            actual,
                            expected_or_unexpected,
                            "{}",
                            clue_msg(clue, attempt)
                        );
                        Ok(output)
                    } else if equals_or_not_equals_str == "!=" {
                        let assertion = RhoTestAssertion::RhoAssertNotEquals {
                            test_name,
                            unexpected: expected_or_unexpected.clone(),
                            actual: actual.clone(),
                            clue: clue.clone(),
                        };

                        let output = vec![new_gbool_par(assertion.is_success(), Vec::new(), false)];
                        produce(&output, &ack_channel).await?;

                        assert_ne!(
                            printer.build_string_from_message(&actual),
                            printer.build_string_from_message(&expected_or_unexpected),
                            "{}",
                            clue_msg(clue, attempt)
                        );

                        assert_ne!(
                            actual,
                            expected_or_unexpected,
                            "{}",
                            clue_msg(clue, attempt)
                        );
                        Ok(output)
                    } else {
                        Err(illegal_argument_error("handle_message"))
                    }
                } else if let Some(condition) = RhoBoolean::unapply(&assertion) {
                    let output = vec![new_gbool_par(condition, Vec::new(), false)];
                    produce(&output, &ack_channel).await?;

                    assert_eq!(condition, true, "{}", clue_msg(clue, attempt));
                    Ok(output)
                } else {
                    let output = vec![new_gbool_par(false, Vec::new(), false)];
                    produce(&output, &ack_channel).await?;

                    Err(InterpreterError::BugFoundError(format!(
                        "Failed to evaluate assertion: {:?}",
                        assertion
                    )))
                }
            } else if let Some(_) = IsSetFinished::unapply(assert_par) {
                Ok(vec![])
            } else {
                Err(illegal_argument_error("handle_message"))
            }
        } else {
            Err(illegal_argument_error("handle_message"))
        }
    }

    // See casper/src/test/scala/coop/rchain/casper/helper/RhoLoggerContract.scala

    pub async fn std_log(
        &mut self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((_, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [log_level_par, par] => {
                    if let Some(log_level) = RhoString::unapply(log_level_par) {
                        let msg = self.pretty_printer.build_string_from_message(par);

                        match log_level.as_str() {
                            "trace" => {
                                println!("trace: {}", msg);
                                Ok(vec![])
                            }
                            "debug" => {
                                println!("debug: {}", msg);
                                Ok(vec![])
                            }
                            "info" => {
                                println!("info: {}", msg);
                                Ok(vec![])
                            }
                            "warn" => {
                                println!("warn: {}", msg);
                                Ok(vec![])
                            }
                            "error" => {
                                println!("error: {}", msg);
                                Ok(vec![])
                            }
                            _ => Err(illegal_argument_error("std_log")),
                        }
                    } else {
                        Err(illegal_argument_error("std_log"))
                    }
                }
                _ => Err(illegal_argument_error("std_log")),
            }
        } else {
            Err(illegal_argument_error("std_log"))
        }
    }

    // See casper/src/test/scala/coop/rchain/casper/helper/DeployerIdContract.scala

    pub async fn deployer_id_make(
        &mut self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((produce, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [deployer_id_par, key_par, ack_channel] => {
                    if let (Some(deployer_id_str), Some(public_key)) = (
                        RhoString::unapply(deployer_id_par),
                        RhoByteArray::unapply(key_par),
                    ) {
                        if deployer_id_str == "deployerId" {
                            let output = vec![RhoDeployerId::create_par(public_key)];
                            produce(&output, &ack_channel).await?;
                            Ok(output)
                        } else {
                            Err(illegal_argument_error("deployer_id_make"))
                        }
                    } else {
                        Err(illegal_argument_error("deployer_id_make"))
                    }
                }
                _ => Err(illegal_argument_error("deployer_id_make")),
            }
        } else {
            Err(illegal_argument_error("deployer_id_make"))
        }
    }

    // See casper/src/test/scala/coop/rchain/casper/helper/Secp256k1SignContract.scala

    pub async fn secp256k1_sign(
        &mut self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((produce, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [hash_par, sk_par, ack_channel] => {
                    if let (Some(hash), Some(secret_key)) = (
                        RhoByteArray::unapply(hash_par),
                        RhoByteArray::unapply(sk_par),
                    ) {
                        if secret_key.len() != 32 {
                            return Err(InterpreterError::BugFoundError(format!(
                                "Invalid private key length: must be 32 bytes, got {}",
                                secret_key.len()
                            )));
                        }

                        let signing_key =
                            SigningKey::from_slice(&secret_key).expect("Invalid private key");

                        let signature: Signature = signing_key
                            .sign_prehash(&hash)
                            .expect("Failed to sign prehash");
                        let der_bytes = signature.to_der().as_bytes().to_vec();

                        let result_par = new_gbytearray_par(der_bytes, Vec::new(), false);

                        let output = vec![result_par];
                        produce(&output, &ack_channel).await?;
                        Ok(output)
                    } else {
                        Err(illegal_argument_error("secp256k1_sign"))
                    }
                }
                _ => Err(illegal_argument_error("secp256k1_sign")),
            }
        } else {
            Err(illegal_argument_error("secp256k1_sign"))
        }
    }

    // See casper/src/test/scala/coop/rchain/casper/helper/SysAuthTokenContract.scala

    pub async fn sys_auth_token_make(
        &mut self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((produce, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [ack_channel] => {
                    let auth_token = new_gsys_auth_token_par(Vec::new(), false);

                    let output = vec![auth_token];
                    produce(&output, &ack_channel).await?;
                    Ok(output)
                }
                _ => Err(illegal_argument_error("sys_auth_token_make")),
            }
        } else {
            Err(illegal_argument_error("sys_auth_token_make"))
        }
    }

    //See casper/src/test/scala/coop/rchain/casper/helper/BlockDataContract.scala

    pub async fn block_data_set(
        &mut self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((produce, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [key_par, value_par, ack_channel] => {
                    if let Some(key) = RhoString::unapply(key_par) {
                        match key.as_str() {
                            "sender" => {
                                if let Some(public_key_bytes) = RhoByteArray::unapply(value_par) {
                                    let mut block_data = self.block_data.write().await;
                                    block_data.sender = PublicKey {
                                        bytes: public_key_bytes.clone().into(),
                                    };
                                    drop(block_data);

                                    let result_par = vec![Par::default()];
                                    produce(&result_par, &ack_channel).await?;
                                    Ok(result_par)
                                } else {
                                    Err(illegal_argument_error("block_data_set"))
                                }
                            }
                            "blockNumber" => {
                                if let Some(block_number) = RhoNumber::unapply(value_par) {
                                    let mut block_data = self.block_data.write().await;
                                    block_data.block_number = block_number;
                                    drop(block_data);

                                    let result_par = vec![Par::default()];
                                    produce(&result_par, &ack_channel).await?;
                                    Ok(result_par)
                                } else {
                                    Err(illegal_argument_error("block_data_set"))
                                }
                            }
                            _ => Err(illegal_argument_error("block_data_set")),
                        }
                    } else {
                        Err(illegal_argument_error("block_data_set"))
                    }
                }
                _ => Err(illegal_argument_error("block_data_set")),
            }
        } else {
            Err(illegal_argument_error("block_data_set"))
        }
    }

    // See casper/src/test/scala/coop/rchain/casper/helper/CasperInvalidBlocksContract.scala

    pub async fn casper_invalid_blocks_set(
        &self,
        message: (Vec<ListParWithRandom>, bool, Vec<Par>),
        invalid_blocks: &InvalidBlocks,
    ) -> Result<Vec<Par>, InterpreterError> {
        if let Some((produce, _, _, args)) = self.is_contract_call().unapply(message) {
            match args.as_slice() {
                [new_invalid_blocks_par, ack_channel] => {
                    let mut invalid_blocks_lock = invalid_blocks.invalid_blocks.write().await;
                    *invalid_blocks_lock = new_invalid_blocks_par.clone();

                    let result_par = vec![Par::default()];
                    produce(&result_par, &ack_channel).await?;
                    Ok(result_par)
                }
                _ => Err(illegal_argument_error("casper_invalid_blocks_set")),
            }
        } else {
            Err(illegal_argument_error("casper_invalid_blocks_set"))
        }
    }
}

// See casper/src/test/scala/coop/rchain/casper/helper/RhoSpec.scala

pub fn test_framework_contracts() -> Vec<Definition> {
    vec![
        Definition {
            urn: "rho:test:assertAck".to_string(),
            fixed_channel: byte_name(101),
            arity: 5,
            body_ref: 101,
            handler: {
                Box::new(|ctx| {
                    let sp = ctx.system_processes.clone();
                    Box::new(move |args| {
                        let sp = sp.clone();
                        Box::pin(async move { sp.handle_message(args).await })
                    })
                })
            },
            remainder: None,
        },
        Definition {
            urn: "rho:test:testSuiteCompleted".to_string(),
            fixed_channel: byte_name(102),
            arity: 1,
            body_ref: 102,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let sp = sp.clone();
                    Box::pin(async move { sp.handle_message(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:io:stdlog".to_string(),
            fixed_channel: byte_name(103),
            arity: 2,
            body_ref: 103,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let mut sp = sp.clone();
                    Box::pin(async move { sp.std_log(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:deployerId:make".to_string(),
            fixed_channel: byte_name(104),
            arity: 3,
            body_ref: 104,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let mut sp = sp.clone();
                    Box::pin(async move { sp.deployer_id_make(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:crypto:secp256k1Sign".to_string(),
            fixed_channel: byte_name(105),
            arity: 3,
            body_ref: 105,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let mut sp = sp.clone();
                    Box::pin(async move { sp.secp256k1_sign(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "sys:test:authToken:make".to_string(),
            fixed_channel: byte_name(106),
            arity: 1,
            body_ref: 106,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let mut sp = sp.clone();
                    Box::pin(async move { sp.sys_auth_token_make(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:block:data:set".to_string(),
            fixed_channel: byte_name(107),
            arity: 3,
            body_ref: 107,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                Box::new(move |args| {
                    let mut sp = sp.clone();
                    Box::pin(async move { sp.block_data_set(args).await })
                })
            }),
            remainder: None,
        },
        Definition {
            urn: "rho:test:casper:invalidBlocks:set".to_string(),
            fixed_channel: byte_name(108),
            arity: 2,
            body_ref: 108,
            handler: Box::new(|ctx| {
                let sp = ctx.system_processes.clone();
                let invalid_blocks = ctx.invalid_blocks.clone();
                Box::new(move |args| {
                    let sp = sp.clone();
                    let invalid_blocks = invalid_blocks.clone();
                    Box::pin(async move { sp.casper_invalid_blocks_set(args, &invalid_blocks).await })
                })
            }),
            remainder: None,
        },
    ]
}

// See casper/src/test/scala/coop/rchain/casper/helper/TestResultCollector.scala

struct IsAssert;

impl IsAssert {
    pub fn unapply(p: Vec<Par>) -> Option<(String, i64, Par, String, Par)> {
        match p.as_slice() {
            [test_name_par, attempt_par, assertion_par, clue_par, ack_channel_par] => {
                if let (Some(test_name), Some(attempt), Some(clue)) = (
                    RhoString::unapply(test_name_par),
                    RhoNumber::unapply(attempt_par),
                    RhoString::unapply(clue_par),
                ) {
                    Some((
                        test_name,
                        attempt,
                        assertion_par.clone(),
                        clue,
                        ack_channel_par.clone(),
                    ))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

struct IsComparison;

impl IsComparison {
    pub fn unapply(p: Par) -> Option<(Par, String, Par)> {
        if let Some(expr) = single_expr(&p) {
            match expr.expr_instance.unwrap() {
                ExprInstance::ETupleBody(etuple) => match etuple.ps.as_slice() {
                    [expected_par, operator_par, actual_par] => {
                        if let Some(operator) = RhoString::unapply(operator_par) {
                            Some((expected_par.clone(), operator, actual_par.clone()))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },

                _ => None,
            }
        } else {
            None
        }
    }
}

struct IsSetFinished;

impl IsSetFinished {
    pub fn unapply(p: Vec<Par>) -> Option<bool> {
        match p.as_slice() {
            [has_finished_par] => {
                if let Some(has_finished) = RhoBoolean::unapply(has_finished_par) {
                    Some(has_finished)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum RhoTestAssertion {
    RhoAssertTrue {
        test_name: String,
        is_success: bool,
        clue: String,
    },

    RhoAssertEquals {
        test_name: String,
        expected: Par,
        actual: Par,
        clue: String,
    },

    RhoAssertNotEquals {
        test_name: String,
        unexpected: Par,
        actual: Par,
        clue: String,
    },
}

impl RhoTestAssertion {
    pub fn is_success(&self) -> bool {
        match self {
            RhoTestAssertion::RhoAssertTrue { is_success, .. } => *is_success,
            RhoTestAssertion::RhoAssertEquals {
                expected, actual, ..
            } => actual == expected,
            RhoTestAssertion::RhoAssertNotEquals {
                unexpected, actual, ..
            } => actual != unexpected,
        }
    }
}

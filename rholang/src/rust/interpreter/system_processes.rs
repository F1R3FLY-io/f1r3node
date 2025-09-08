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
use super::util::rev_address::RevAddress;
use crypto::rust::hash::blake2b256::Blake2b256;
use crypto::rust::hash::keccak256::Keccak256;
use crypto::rust::hash::sha_256::Sha256Hasher;
use crypto::rust::public_key::PublicKey;
use crypto::rust::signatures::ed25519::Ed25519;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;
use k256::{
    ecdsa::{signature::Signer, Signature, SigningKey},
    elliptic_curve::generic_array::GenericArray,
};
use models::rhoapi::expr::ExprInstance;
use models::rhoapi::g_unforgeable::UnfInstance::GPrivateBody;
use models::rhoapi::{Bundle, GPrivate, GUnforgeable, ListParWithRandom, Par, Var};
use models::rust::casper::protocol::casper_message::BlockMessage;
use models::rust::rholang::implicits::single_expr;
use models::rust::utils::{new_gbool_par, new_gbytearray_par, new_gsys_auth_token_par};
use models::Byte;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/SystemProcesses.scala
// NOTE: Not implementing Logger
pub type RhoSysFunction = Box<
    dyn Fn(
        (Vec<ListParWithRandom>, bool, Vec<Par>),
    ) -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>>>>,
>;
pub type RhoDispatchMap = Arc<RwLock<HashMap<i64, RhoSysFunction>>>;
pub type Name = Par;
pub type Arity = i32;
pub type Remainder = Option<Var>;
pub type BodyRef = i64;
pub type Contract = dyn Fn(Vec<ListParWithRandom>) -> ();

#[derive(Clone)]
pub struct InvalidBlocks {
    pub invalid_blocks: Arc<RwLock<Par>>,
}

impl InvalidBlocks {
    pub fn new() -> Self {
        InvalidBlocks {
            invalid_blocks: Arc::new(RwLock::new(Par::default())),
        }
    }

    pub fn set_params(&self, invalid_blocks: Par) -> () {
        let mut lock: RwLockWriteGuard<Par> = self.invalid_blocks.write().unwrap();

        *lock = invalid_blocks;
    }
}

pub fn byte_name(b: Byte) -> Par {
    Par::default().with_unforgeables(vec![GUnforgeable {
        unf_instance: Some(GPrivateBody(GPrivate { id: vec![b] })),
    }])
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
}

pub fn non_deterministic_ops() -> HashSet<i64> {
    HashSet::from([
        BodyRefs::GPT4,
        BodyRefs::DALLE3,
        BodyRefs::TEXT_TO_AUDIO,
        BodyRefs::RANDOM,
    ])
}

#[derive(Clone)]
pub struct ProcessContext {
    pub space: RhoISpace,
    pub dispatcher: RhoDispatch,
    pub block_data: Arc<RwLock<BlockData>>,
    pub invalid_blocks: InvalidBlocks,
    pub system_processes: SystemProcesses,
}

impl ProcessContext {
    pub fn create(
        space: RhoISpace,
        dispatcher: RhoDispatch,
        block_data: Arc<RwLock<BlockData>>,
        invalid_blocks: InvalidBlocks,
        openai_service: Arc<Mutex<OpenAIService>>,
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
            ),
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
                -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>>>>,
        >,
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
                    -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>>>>,
            >,
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
                -> Pin<Box<dyn Future<Output = Result<Vec<Par>, InterpreterError>>>>,
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
    pub block_data: Arc<RwLock<BlockData>>,
    openai_service: Arc<Mutex<OpenAIService>>,
    pretty_printer: PrettyPrinter,
}

impl SystemProcesses {
    fn create(
        dispatcher: RhoDispatch,
        space: RhoISpace,
        block_data: Arc<RwLock<BlockData>>,
        openai_service: Arc<Mutex<OpenAIService>>,
    ) -> Self {
        SystemProcesses {
            dispatcher,
            space,
            block_data,
            openai_service,
            pretty_printer: PrettyPrinter::new(),
        }
    }

    fn is_contract_call(&self) -> ContractCall {
        ContractCall {
            space: self.space.clone(),
            dispatcher: self.dispatcher.clone(),
        }
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
        algorithm: Box<dyn Fn(Vec<u8>) -> Vec<u8>>,
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
        &mut self,
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
                .map(|public_key| RevAddress::from_public_key(&PublicKey { bytes: public_key }))
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
        block_data: Arc<RwLock<BlockData>>,
    ) -> Result<Vec<Par>, InterpreterError> {
        let Some((produce, _, _, args)) = self.is_contract_call().unapply(contract_args) else {
            return Err(illegal_argument_error("get_block_data"));
        };

        let [ack] = args.as_slice() else {
            return Err(illegal_argument_error("get_block_data"));
        };

        let data = block_data.read().unwrap();
        let output = vec![
            Par::default().with_exprs(vec![RhoNumber::create_expr(data.block_number)]),
            Par::default().with_exprs(vec![RhoNumber::create_expr(data.time_stamp)]),
            RhoByteArray::create_par(data.sender.bytes.clone()),
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

        let invalid_blocks = invalid_blocks.invalid_blocks.read().unwrap().clone();
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

        let mut openai_service = self.openai_service.lock().unwrap();
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

        let mut openai_service = self.openai_service.lock().unwrap();
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

        let mut openai_service = self.openai_service.lock().unwrap();
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

    /*

    override def grpcTell: Contract[F] = {
        case isContractCall(_, true, previous, args) =>
          // args could be:
          // - clientHost, clientPort, folderId, ack
          // - clientHost, clientPort, folderId, error, ack if failed previously

          // so using the last element as ack
          println("grpcTell (replay): args: " + args)
          F.delay(previous)

        case isContractCall(
            _,
            false,
            _,
            Seq(
              RhoType.String(clientHost),
              RhoType.Number(clientPort),
              RhoType.String(notificationPayload)
            )
            ) =>
          //TODO: remove
          println(
            "grpcTell: clientHost: " + clientHost + ", clientPort: " + clientPort + ", notificationPayload: " + notificationPayload
          )
          (for {
            _ <- GrpcClient.initClientAndTell(clientHost, clientPort, notificationPayload).recover {
                  case e => println("GrpcClient crashed: " + e.getMessage)
                }
            output = Seq(RhoType.Nil())
          } yield output).onError {
            case e =>
              println("grpcTell: error: " + e.getMessage)
              e.raiseError
          }
        case isContractCall(_, isReplay, _, args) =>
          println("grpcTell: isReplay " + isReplay + " invalid arguments: " + args)
          F.delay(Seq(RhoType.Nil()))
      }

     */

    /*
     * The following functions below can be removed once rust-casper calls create_rho_runtime.
     * Until then, they must remain in the rholang directory to avoid circular dependencies.
     */

    // See casper/src/test/scala/coop/rchain/casper/helper/TestResultCollector.scala

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

                        let key_bytes = GenericArray::clone_from_slice(&secret_key);
                        let signing_key =
                            SigningKey::from_bytes(&key_bytes).expect("Invalid private key");

                        let signature: Signature = signing_key.sign(&hash);

                        let result_par = new_gbytearray_par(
                            signature.to_der().as_bytes().to_vec(),
                            Vec::new(),
                            false,
                        );

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
                                    let mut block_data = self.block_data.write().unwrap();
                                    block_data.sender = PublicKey {
                                        bytes: public_key_bytes.clone(),
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
                                    let mut block_data = self.block_data.write().unwrap();
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
                    let mut invalid_blocks_lock = invalid_blocks.invalid_blocks.write().unwrap();
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

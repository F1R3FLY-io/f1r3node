// See casper/src/main/scala/coop/rchain/casper/Validate.scala

use crate::rust::{
    block_status::{BlockError, InvalidBlock, ValidBlock},
    casper::CasperSnapshot,
    ValidBlockProcessing,
};
use block_storage::rust::key_value_block_store::KeyValueBlockStore;
use bytes::Bytes;
use crypto::rust::hash::blake2b256::Blake2b256;
use crypto::rust::signatures::secp256k1::Secp256k1;
use crypto::rust::signatures::signatures_alg::SignaturesAlg;
use models::casper::Signature as ProtoSignature;
use models::rust::{
    block_metadata::BlockMetadata,
    casper::pretty_printer::PrettyPrinter,
    casper::protocol::casper_message::{ApprovedBlock, BlockMessage},
};
use rspace_plus_plus::rspace::history::Either;
use shared::rust::{dag::dag_ops, store::key_value_store::KvStoreError};

use crate::rust::util::proto_util;

use crate::rust::errors::CasperError;
use crate::rust::estimator::Estimator;
use crate::rust::util::rholang::runtime_manager::RuntimeManager;
use models::rust::block_hash::BlockHash;
use models::rust::validator::Validator;
use prost::{bytes, Message};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

pub type PublicKey = Vec<u8>;
pub type Data = Vec<u8>;
pub type Signature = Vec<u8>;

const DRIFT: i64 = 15000; // 15 seconds

pub struct Validate;

impl Validate {
    //TODO: It should be simplified once we remove &self from the verify function.
    fn signature_verifiers() -> HashMap<String, Box<dyn Fn(&Data, &Signature, &PublicKey) -> bool>>
    {
        let mut map: HashMap<String, Box<dyn Fn(&Data, &Signature, &PublicKey) -> bool>> =
            HashMap::new();
        map.insert(
            "secp256k1".to_string(),
            Box::new(|data: &Vec<u8>, signature: &Vec<u8>, pub_key: &Vec<u8>| {
                let secp256k1 = Secp256k1;
                secp256k1.verify(data, signature, pub_key)
            }) as Box<dyn Fn(&Data, &Signature, &PublicKey) -> bool>,
        );
        map
    }

    pub fn signature(d: &Data, sig: &ProtoSignature) -> bool {
        Self::signature_verifiers()
            .get(&sig.algorithm)
            .map_or(false, |verify| {
                verify(d, &sig.sig.to_vec(), &sig.public_key.to_vec())
            })
    }

    fn ignore(b: &BlockMessage, reason: &str) -> String {
        format!(
            "Ignoring block {} because {}",
            PrettyPrinter::build_string_bytes(&b.block_hash),
            reason
        )
    }

    pub fn approved_block(approved_block: &ApprovedBlock) -> bool {
        let candidate_bytes_digest =
            Blake2b256::hash(approved_block.clone().candidate.to_proto().encode_to_vec());
        let required_signatures = approved_block.candidate.required_sigs;

        let signature_verifiers = Self::signature_verifiers();

        let signatures: HashSet<Bytes> = approved_block
            .sigs
            .iter()
            .filter_map(|signature| {
                signature_verifiers
                    .get(&signature.algorithm)
                    .and_then(|verify_sig| {
                        if verify_sig(
                            &candidate_bytes_digest,
                            &signature.sig.to_vec(),
                            &signature.public_key.to_vec(),
                        ) {
                            Some(signature.public_key.clone())
                        } else {
                            None
                        }
                    })
            })
            .collect();

        let log_msg = match signatures.is_empty() {
            true => "ApprovedBlock is self-signed by ceremony master.".to_string(),
            false => {
                let sigs_str = signatures
                    .iter()
                    .map(|pk| {
                        let hex_str = hex::encode(pk);
                        format!("<{}...>", &hex_str[..10])
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("ApprovedBlock is signed by: {}", sigs_str)
            }
        };

        log::info!("{}", log_msg);
        let enough_sigs = signatures.len() >= required_signatures as usize;

        if !enough_sigs {
            log::warn!(
                "Received invalid ApprovedBlock message not containing enough valid signatures."
            );
        }

        enough_sigs
    }

    pub fn block_signature(b: &BlockMessage) -> bool {
        let result = Self::signature_verifiers()
            .get(&b.sig_algorithm)
            .map(|verify| {
                match verify(&b.block_hash.to_vec(), &b.sig.to_vec(), &b.sender.to_vec()) {
                    true => true,
                    false => {
                        log::warn!("{}", Self::ignore(b, "signature is invalid."));
                        false
                    }
                }
            });

        result.unwrap_or_else(|| {
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!("signature algorithm {} is unsupported.", b.sig_algorithm)
                )
            );
            false
        })
    }

    pub fn block_sender_has_weight(
        b: &BlockMessage,
        genesis: &BlockMessage,
        block_store: &mut KeyValueBlockStore,
    ) -> Result<bool, KvStoreError> {
        if b == genesis {
            Ok(true)
        } else {
            proto_util::weight_from_sender(block_store, b).map(|weight| {
                if weight > 0 {
                    true
                } else {
                    log::warn!(
                        "{}",
                        Self::ignore(
                            b,
                            &format!(
                                "block creator {} has 0 weight.",
                                PrettyPrinter::build_string_bytes(&b.sender)
                            )
                        )
                    );
                    false
                }
            })
        }
    }

    pub fn format_of_fields(b: &BlockMessage) -> bool {
        if b.block_hash.is_empty() {
            log::warn!("{}", Self::ignore(b, "block hash is empty."));
            false
        } else if b.sig.is_empty() {
            log::warn!("{}", Self::ignore(b, "block signature is empty."));
            false
        } else if b.sig_algorithm.is_empty() {
            log::warn!("{}", Self::ignore(b, "block signature algorithm is empty."));
            false
        } else if b.shard_id.is_empty() {
            log::warn!("{}", Self::ignore(b, "block shard identifier is empty."));
            false
        } else if b.body.state.post_state_hash.is_empty() {
            log::warn!("{}", Self::ignore(b, "block post state hash is empty."));
            false
        } else {
            true
        }
    }

    pub fn version(b: &BlockMessage, version: i64) -> bool {
        let block_version = b.header.version;
        if block_version == version {
            true
        } else {
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!(
                        "received block version {} is the expected version {}.",
                        block_version, version
                    )
                )
            );
            false
        }
    }

    //TODO: Scala message -> Double check ordering of validity checks
    pub async fn block_summary(
        block: &BlockMessage,
        genesis: &BlockMessage,
        s: &mut CasperSnapshot,
        shard_id: &str,
        expiration_threshold: i32,
        estimator: &Estimator,
        block_store: &mut KeyValueBlockStore,
    ) -> ValidBlockProcessing {
        match Self::block_hash(block) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::timestamp(block, block_store) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::shard_identifier(block, shard_id) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::deploys_shard_identifier(block, shard_id) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::repeat_deploy(block, s, block_store, expiration_threshold) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::block_number(block, s) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::future_transaction(block) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::transaction_expiration(block, expiration_threshold) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::justification_follows(block, block_store) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::parents(block, genesis, s, estimator).await {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::sequence_number(block, s) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }
        match Self::justification_regressions(block, s) {
            Either::Left(err) => return Either::Left(err),
            Either::Right(_) => {}
        }

        // Equivalent to Scala's "} yield s).value" - return ValidBlock if all validations passed
        Either::Right(ValidBlock::Valid)
    }

    /// Validate no deploy with the same sig has been produced in the chain
    /// Agnostic of non-parent justifications
    pub fn repeat_deploy(
        block: &BlockMessage,
        s: &mut CasperSnapshot,
        block_store: &mut KeyValueBlockStore,
        expiration_threshold: i32,
    ) -> ValidBlockProcessing {
        let deploy_key_set: HashSet<Bytes> = block
            .body
            .deploys
            .iter()
            .map(|deploy| deploy.deploy.sig.clone())
            .collect();

        let block_metadata = BlockMetadata::from_block(block, false, None, None);

        let init_parents = match proto_util::get_parents_metadata(&s.dag, &block_metadata) {
            Ok(parents) => parents,
            Err(e) => return Either::Left(BlockError::BlockException(CasperError::from(e))),
        };

        // Calculate max block number and earliest acceptable block number
        let max_block_number = proto_util::max_block_number_metadata(&init_parents);
        let earliest_block_number = max_block_number + 1 - expiration_threshold as i64;

        let traversed_blocks = dag_ops::bf_traverse(init_parents, |block_metadata| {
            proto_util::get_parent_metadatas_above_block_number(
                &block_metadata,
                earliest_block_number,
                &s.dag,
            )
            .unwrap_or_default()
        });

        let maybe_duplicated_block_metadata = traversed_blocks.into_iter().find(|block_metadata| {
            let block = block_store.get_unsafe(&block_metadata.block_hash);
            let block_deploys = proto_util::deploys(&block);
            block_deploys
                .iter()
                .any(|deploy| deploy_key_set.contains(&deploy.deploy.sig))
        });

        let maybe_error = maybe_duplicated_block_metadata.map(|duplicated_block_metadata| {
      let duplicated_block = block_store.get_unsafe(&duplicated_block_metadata.block_hash);
      let current_block_hash_string = PrettyPrinter::build_string_bytes(&block.block_hash);
      let block_hash_string = PrettyPrinter::build_string_bytes(&duplicated_block.block_hash);

      let duplicated_deploys = proto_util::deploys(&duplicated_block);
      let duplicated_deploy = duplicated_deploys
        .iter()
        .map(|processed_deploy| &processed_deploy.deploy)
        .find(|deploy| deploy_key_set.contains(&deploy.sig))
        .expect("Duplicated deploy should exist");

      let term = &duplicated_deploy.data.term;
      let deployer_string = PrettyPrinter::build_string_bytes(&duplicated_deploy.pk.bytes);
      let timestamp_string = duplicated_deploy.data.time_stamp.to_string();

      let message = format!(
        "found deploy [{}] (user {}, millisecond timestamp {})] with the same sig in the block {} as current block {}",
        term,
        &deployer_string,
        timestamp_string,
        block_hash_string,
        current_block_hash_string
      );

      log::warn!("{}", Self::ignore(block, &message));
      BlockError::Invalid(InvalidBlock::InvalidRepeatDeploy)
    });

        maybe_error.map_or(Either::Right(ValidBlock::Valid), Either::Left)
    }

    // This is not a slashable offence
    pub fn timestamp(
        b: &BlockMessage,
        block_store: &mut KeyValueBlockStore,
    ) -> ValidBlockProcessing {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let timestamp = b.header.timestamp;

        let before_future = current_time + DRIFT >= timestamp;

        let latest_parent_timestamp =
            proto_util::parent_hashes(b)
                .iter()
                .fold(0i64, |latest_timestamp, parent_hash| {
                    let parent = block_store.get_unsafe(parent_hash);
                    let timestamp = parent.header.timestamp;
                    latest_timestamp.max(timestamp)
                });
        let after_latest_parent = timestamp >= latest_parent_timestamp;

        if before_future && after_latest_parent {
            Either::Right(ValidBlock::Valid)
        } else {
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!(
                        "block timestamp {} is not between latest parent block time and current time.",
                        timestamp
                    )
                )
            );
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidTimestamp))
        }
    }

    /// Agnostic of non-parent justifications
    pub fn block_number(b: &BlockMessage, s: &mut CasperSnapshot) -> ValidBlockProcessing {
        let parents: Vec<BlockMetadata> = match proto_util::parent_hashes(b)
            .iter()
            .map(|parent_hash| match s.dag.lookup(parent_hash) {
                Ok(Some(parent_metadata)) => Ok(parent_metadata),
                Ok(None) => Err(KvStoreError::KeyNotFound(format!(
                    "Block dag store was missing {}",
                    PrettyPrinter::build_string_bytes(parent_hash)
                ))),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<BlockMetadata>, KvStoreError>>()
        {
            Ok(parents) => parents,
            Err(e) => return Either::Left(BlockError::BlockException(CasperError::from(e))),
        };

        let max_block_number = parents
            .iter()
            .fold(-1, |acc, parent| acc.max(parent.block_number));

        let number = proto_util::block_number(b);
        let result = max_block_number + 1 == number;

        if result {
            Either::Right(ValidBlock::Valid)
        } else {
            let log_message = if parents.is_empty() {
                format!(
                    "block number {} is not zero, but block has no parents.",
                    number
                )
            } else {
                format!(
                    "block number {} is not one more than maximum parent number {}.",
                    number, max_block_number
                )
            };

            log::warn!("{}", Self::ignore(b, &log_message));
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockNumber))
        }
    }

    pub fn future_transaction(b: &BlockMessage) -> ValidBlockProcessing {
        let block_number = proto_util::block_number(b);

        let processed_deploys = proto_util::deploys(b);
        let deploys: Vec<_> = processed_deploys
            .iter()
            .map(|processed_deploy| &processed_deploy.deploy)
            .collect();

        let maybe_future_deploy = deploys
            .iter()
            .find(|&deploy| deploy.data.valid_after_block_number >= block_number);

        let maybe_error = maybe_future_deploy.map(|future_deploy| {
            let message = format!(
                "block contains an future deploy with valid after block number of {}: {}",
                future_deploy.data.valid_after_block_number, future_deploy.data.term
            );

            log::warn!("{}", Self::ignore(b, &message));
            BlockError::Invalid(InvalidBlock::ContainsFutureDeploy)
        });

        maybe_error.map_or(Either::Right(ValidBlock::Valid), Either::Left)
    }

    pub fn transaction_expiration(
        b: &BlockMessage,
        expiration_threshold: i32,
    ) -> ValidBlockProcessing {
        let earliest_acceptable_valid_after_block_number =
            proto_util::block_number(b) - expiration_threshold as i64;

        let processed_deploys = proto_util::deploys(b);
        let deploys: Vec<_> = processed_deploys
            .iter()
            .map(|processed_deploy| &processed_deploy.deploy)
            .collect();

        let maybe_expired_deploy = deploys.iter().find(|&deploy| {
            deploy.data.valid_after_block_number <= earliest_acceptable_valid_after_block_number
        });

        let maybe_error = maybe_expired_deploy.map(|expired_deploy| {
            let message = format!(
                "block contains an expired deploy with valid after block number of {}: {}",
                expired_deploy.data.valid_after_block_number, expired_deploy.data.term
            );

            log::warn!("{}", Self::ignore(b, &message));
            BlockError::Invalid(InvalidBlock::ContainsExpiredDeploy)
        });

        maybe_error.map_or(Either::Right(ValidBlock::Valid), Either::Left)
    }

    /// Works with either efficient justifications or full explicit justifications.
    /// Specifically, with efficient justifications, if a block B doesn't update its
    /// creator justification, this check will fail as expected. The exception is when
    /// B's creator justification is the genesis block.
    pub fn sequence_number(b: &BlockMessage, s: &mut CasperSnapshot) -> ValidBlockProcessing {
        let creator_justification_seq_number =
            match proto_util::creator_justification_block_message(b) {
                Some(justification) => match s.dag.lookup(&justification.latest_block_hash) {
                    Ok(Some(block_metadata)) => block_metadata.sequence_number as i64,
                    Ok(None) => {
                        return Either::Left(BlockError::BlockException(CasperError::from(
                            KvStoreError::KeyNotFound(format!(
                                "Latest block hash {} is missing from block dag store.",
                                PrettyPrinter::build_string_bytes(&justification.latest_block_hash)
                            )),
                        )));
                    }
                    Err(e) => {
                        return Either::Left(BlockError::BlockException(CasperError::from(e)));
                    }
                },
                None => -1,
            };

        let number = b.seq_num as i64;
        let result = creator_justification_seq_number + 1 == number;

        if result {
            Either::Right(ValidBlock::Valid)
        } else {
            let message = format!(
                "seq number {} is not one more than creator justification number {}.",
                number, creator_justification_seq_number
            );

            log::warn!("{}", Self::ignore(b, &message));
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidSequenceNumber))
        }
    }

    // Agnostic of justifications
    pub fn shard_identifier(b: &BlockMessage, shard_id: &str) -> ValidBlockProcessing {
        if b.shard_id == shard_id {
            Either::Right(ValidBlock::Valid)
        } else {
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!(
                        "got shard identifier {} while {} was expected.",
                        b.shard_id, shard_id
                    )
                )
            );
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidShardId))
        }
    }

    // Validator should only process deploys from its own shard
    pub fn deploys_shard_identifier(b: &BlockMessage, shard_id: &str) -> ValidBlockProcessing {
        if b.body
            .deploys
            .iter()
            .all(|deploy| deploy.deploy.data.shard_id == shard_id)
        {
            Either::Right(ValidBlock::Valid)
        } else {
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!("not for all deploys shard identifier is {}.", shard_id)
                )
            );
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidShardId))
        }
    }

    // TODO: Scala message -> Double check this validation isn't shadowed by the blockSignature validation
    pub fn block_hash(b: &BlockMessage) -> ValidBlockProcessing {
        let block_hash_computed = proto_util::hash_block(b);
        if b.block_hash == block_hash_computed {
            Either::Right(ValidBlock::Valid)
        } else {
            let computed_hash_string = PrettyPrinter::build_string_bytes(&block_hash_computed);
            let hash_string = PrettyPrinter::build_string_bytes(&b.block_hash);
            log::warn!(
                "{}",
                Self::ignore(
                    b,
                    &format!(
                        "block hash {} does not match to computed value {}.",
                        hash_string, computed_hash_string
                    )
                )
            );
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidBlockHash))
        }
    }

    /// Works only with fully explicit justifications.
    pub async fn parents(
        b: &BlockMessage,
        genesis: &BlockMessage,
        s: &mut CasperSnapshot,
        estimator: &Estimator,
    ) -> ValidBlockProcessing {
        let maybe_parent_hashes = proto_util::parent_hashes(b);
        let parent_hashes = match maybe_parent_hashes {
            hashes if hashes.is_empty() => vec![genesis.block_hash.clone()],
            hashes => hashes,
        };

        let latest_messages_hashes = proto_util::to_latest_message_hashes(b.justifications.clone());
        let tip_hashes = match estimator
            .tips_with_latest_messages(&mut s.dag, genesis, latest_messages_hashes.clone())
            .await
        {
            Ok(tip_hashes) => tip_hashes,
            Err(e) => return Either::Left(BlockError::from(e)),
        };
        let computed_parent_hashes = tip_hashes.tips;

        if parent_hashes == computed_parent_hashes {
            Either::Right(ValidBlock::Valid)
        } else {
            let parents_string: String = parent_hashes
                .iter()
                .map(|hash| PrettyPrinter::build_string_bytes(hash))
                .collect::<Vec<String>>()
                .join(",");
            let estimate_string: String = computed_parent_hashes
                .iter()
                .map(|hash| PrettyPrinter::build_string_bytes(hash))
                .collect::<Vec<String>>()
                .join(",");
            let justification_string: String = latest_messages_hashes
                .values()
                .map(|hash| PrettyPrinter::build_string_bytes(hash))
                .collect::<Vec<String>>()
                .join(",");

            let message = format!(
                "block parents {} did not match estimate {} based on justification {}.",
                parents_string, estimate_string, justification_string
            );

            log::warn!("{}", Self::ignore(b, &message));
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents))
        }
    }

    /// This check must come before Validate.parents
    pub fn justification_follows(
        b: &BlockMessage,
        block_store: &mut KeyValueBlockStore,
    ) -> ValidBlockProcessing {
        let justified_validators: HashSet<Bytes> = b
            .justifications
            .iter()
            .map(|justification| justification.validator.clone())
            .collect();

        let parent_hashes = proto_util::parent_hashes(b);
        let main_parent_hash = match parent_hashes.first() {
            Some(hash) => hash,
            None => return Either::Left(BlockError::Invalid(InvalidBlock::InvalidParents)),
        };

        let main_parent = block_store.get_unsafe(main_parent_hash);
        let bonded_validators: HashSet<Bytes> = proto_util::bonds(&main_parent)
            .iter()
            .map(|bond| bond.validator.clone())
            .collect();

        if bonded_validators == justified_validators {
            Either::Right(ValidBlock::Valid)
        } else {
            let justified_validators_pp: HashSet<String> = justified_validators
                .iter()
                .map(|validator| PrettyPrinter::build_string_bytes(validator))
                .collect();
            let bonded_validators_pp: HashSet<String> = bonded_validators
                .iter()
                .map(|validator| PrettyPrinter::build_string_bytes(validator))
                .collect();

            let message = format!(
                "the justified validators, {:?}, do not match the bonded validators, {:?}.",
                justified_validators_pp, bonded_validators_pp
            );

            log::warn!("{}", Self::ignore(b, &message));
            Either::Left(BlockError::Invalid(InvalidBlock::InvalidFollows))
        }
    }

    /// Justification regression check.
    /// Compares justifications that has been already used by sender and recorded in the DAG with
    /// justifications used by the same sender in new block `b` and assures that there is no
    /// regression.
    ///
    /// When we switch between equivocation forks for a slashed validator, we will potentially get a
    /// justification regression that is valid. We cannot ignore this as the creator only drops the
    /// justification block created by the equivocator on the following block.
    /// Hence, we ignore justification regressions involving the block's sender and
    /// let checkEquivocations handle it instead.
    // TODO double check this logic
    pub fn justification_regressions(
        b: &BlockMessage,
        s: &mut CasperSnapshot,
    ) -> ValidBlockProcessing {
        match s.dag.latest_message(&b.sender) {
            Ok(None) => {
                // `b` is first message from sender of `b`, so regression is not possible
                Either::Right(ValidBlock::Valid)
            }
            Ok(Some(cur_senders_block)) => {
                // Latest Message from sender of `b` is present in the DAG
                // Here we comparing view on the network by sender from the standpoint of
                // his previous block created (current Latest Message of sender)
                // and new block `b` (potential new Latest Message of sender)
                let new_sender_block = b;
                let new_lms =
                    proto_util::to_latest_message_hashes(new_sender_block.justifications.clone());
                let cur_lms =
                    proto_util::to_latest_message_hashes(cur_senders_block.justifications.clone());

                // We let checkEquivocations handle when sender uses old self-justification
                let new_lms_no_self: HashMap<Validator, BlockHash> = new_lms
                    .into_iter()
                    .filter(|(validator, _)| validator != &b.sender)
                    .collect();

                // Check each Latest Message for regression (block seq num goes backwards)
                let mut remaining_lms: Vec<(Validator, BlockHash)> =
                    new_lms_no_self.into_iter().collect();

                let log_warn =
                    |current_hash: &BlockHash, regressive_hash: &BlockHash, sender: &Validator| {
                        let msg = format!(
                            "block {} by {} has a lower sequence number than {}.",
                            PrettyPrinter::build_string_bytes(regressive_hash),
                            PrettyPrinter::build_string_bytes(sender),
                            PrettyPrinter::build_string_bytes(current_hash)
                        );
                        log::warn!("{}", Self::ignore(b, &msg));
                    };

                loop {
                    match remaining_lms.as_slice() {
                        // No more Latest Messages to check
                        [] => break,
                        // Check if sender of LatestMessage does justification regression
                        [new_lm, tail @ ..] => {
                            let (sender, new_justification_hash) = new_lm;
                            let no_sender_in_cur_lms = !cur_lms.contains_key(sender);

                            if no_sender_in_cur_lms {
                                // If there is no justification to compare with - regression is not possible
                                remaining_lms = tail.to_vec();
                                continue;
                            }

                            let cur_justification_hash = &cur_lms[sender];

                            // Compare and check for regression
                            let new_justification =
                                match s.dag.lookup_unsafe(new_justification_hash) {
                                    Ok(metadata) => metadata,
                                    Err(e) => {
                                        return Either::Left(BlockError::BlockException(
                                            CasperError::from(e),
                                        ))
                                    }
                                };
                            let cur_justification =
                                match s.dag.lookup_unsafe(cur_justification_hash) {
                                    Ok(metadata) => metadata,
                                    Err(e) => {
                                        return Either::Left(BlockError::BlockException(
                                            CasperError::from(e),
                                        ))
                                    }
                                };

                            let regression_detected = {
                                let regression = !new_justification.invalid
                                    && new_justification.sequence_number
                                        < cur_justification.sequence_number;

                                if regression {
                                    log_warn(
                                        cur_justification_hash,
                                        new_justification_hash,
                                        sender,
                                    );
                                }

                                regression
                            };

                            // Exit when regression detected, or continue to check remaining Latest Messages
                            if regression_detected {
                                return Either::Left(BlockError::Invalid(
                                    InvalidBlock::JustificationRegression,
                                ));
                            } else {
                                remaining_lms = tail.to_vec();
                            }
                        }
                    }
                }

                Either::Right(ValidBlock::Valid)
            }
            Err(e) => Either::Left(BlockError::BlockException(CasperError::from(e))),
        }
    }

    /// If block contains an invalid justification block B and the creator of B is still bonded,
    /// return a RejectableBlock. Otherwise, return an IncludeableBlock.
    pub fn neglected_invalid_block(
        block: &BlockMessage,
        s: &mut CasperSnapshot,
    ) -> ValidBlockProcessing {
        let mut invalid_justifications = Vec::new();
        for justification in &block.justifications {
            let latest_block_opt = match s.dag.lookup(&justification.latest_block_hash) {
                Ok(opt) => opt,
                Err(e) => return Either::Left(BlockError::BlockException(CasperError::from(e))),
            };
            if latest_block_opt.map_or(false, |block_metadata| block_metadata.invalid) {
                invalid_justifications.push(justification);
            }
        }

        let bonds = proto_util::bonds(block);
        let neglected_invalid_justification = invalid_justifications.iter().any(|justification| {
            let slashed_validator_bond = bonds
                .iter()
                .find(|bond| bond.validator == justification.validator);

            match slashed_validator_bond {
                Some(bond) => bond.stake > 0,
                None => false,
            }
        });

        if neglected_invalid_justification {
            Either::Left(BlockError::Invalid(InvalidBlock::NeglectedInvalidBlock))
        } else {
            Either::Right(ValidBlock::Valid)
        }
    }

    pub async fn bonds_cache(
        b: &BlockMessage,
        runtime_manager: &RuntimeManager,
    ) -> ValidBlockProcessing {
        let bonds = proto_util::bonds(b);
        let tuplespace_hash = proto_util::post_state_hash(b);

        match runtime_manager.compute_bonds(&tuplespace_hash).await {
            Ok(computed_bonds) => {
                let bonds_set: HashSet<_> = bonds
                    .iter()
                    .map(|bond| (&bond.validator, bond.stake))
                    .collect();
                let computed_bonds_set: HashSet<_> = computed_bonds
                    .iter()
                    .map(|bond| (&bond.validator, bond.stake))
                    .collect();

                if bonds_set == computed_bonds_set {
                    Either::Right(ValidBlock::Valid)
                } else {
                    log::warn!("Bonds in proof of stake contract do not match block's bond cache.");
                    Either::Left(BlockError::Invalid(InvalidBlock::InvalidBondsCache))
                }
            }
            Err(ex) => {
                log::warn!("Failed to compute bonds from tuplespace hash: {}", ex);
                Either::Left(BlockError::BlockException(ex))
            }
        }
    }

    /// All of deploys must have greater or equal phloPrice than minPhloPrice
    pub fn phlo_price(b: &BlockMessage, min_phlo_price: i64) -> ValidBlockProcessing {
        if b.body
            .deploys
            .iter()
            .all(|deploy| deploy.deploy.data.phlo_price >= min_phlo_price)
        {
            Either::Right(ValidBlock::Valid)
        } else {
            Either::Left(BlockError::Invalid(InvalidBlock::LowDeployCost))
        }
    }
}

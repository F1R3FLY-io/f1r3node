// See casper/src/main/scala/coop/rchain/casper/SafetyOracle.scala

use crate::rust::safety::clique_oracle::CliqueOracle;
use block_storage::rust::dag::block_dag_key_value_storage::KeyValueDagRepresentation;
use models::rust::block_hash::BlockHash;
use shared::rust::store::key_value_store::KvStoreError;
/*
 * Implementation inspired by Ethereum's CBC casper simulator's clique oracle implementation.
 *
 * https://github.com/ethereum/cbc-casper/blob/0.2.0/casper/safety_oracles/clique_oracle.py
 *
 * "If nodes in an e-clique see each other agreeing on e and can't see each other disagreeing on e,
 * then there does not exist any new message from inside the clique that will cause them to assign
 * lower scores to e. Further, if the clique has more than half of the validators by weight,
 * then no messages external to the clique can raise the scores these validators assign to
 * a competing [candidate] to be higher than the score they assign to e."
 *
 * - From https://github.com/ethereum/research/blob/master/papers/CasperTFG/CasperTFG.pdf
 *
 * That is unless there are equivocations.
 * The fault tolerance threshold is a subjective value that the user sets to "secretly" state that they
 * tolerate up to fault_tolerance_threshold fraction of the total weight to equivocate.
 *
 * In the extreme case when your normalized fault tolerance threshold is 1,
 * all validators must be part of the clique that supports the candidate in order to state that it is finalized.
 */

pub trait SafetyOracle {
    /**
     * The normalizedFaultTolerance must be greater than the fault tolerance threshold t in order
     * for a candidate to be safe.
     *
     * @param candidateBlockHash Block hash of candidate block to detect safety on
     * @return normalizedFaultTolerance float between -1 and 1, where -1 means potentially orphaned
     */
    async fn normalized_fault_tolerance(
        block_dag: &KeyValueDagRepresentation,
        candidate_block_hash: &BlockHash,
    ) -> Result<f32, KvStoreError>;
}

pub const MIN_FAULT_TOLERANCE: f32 = -1.0;
pub const MAX_FAULT_TOLERANCE: f32 = 1.0;

pub struct CliqueOracleImpl;

impl SafetyOracle for CliqueOracleImpl {
    async fn normalized_fault_tolerance(
        block_dag: &KeyValueDagRepresentation,
        candidate_block_hash: &BlockHash,
    ) -> Result<f32, KvStoreError> {
        CliqueOracle::normalized_fault_tolerance(&candidate_block_hash, &block_dag).await
    }
}

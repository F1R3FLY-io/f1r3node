// See rspace/src/main/scala/coop/rchain/rspace/IReplaySpace.scala

use super::errors::RSpaceError;
use super::hashing::blake2b256_hash::Blake2b256Hash;
use super::rspace_interface::ISpace;
use super::trace::Log;

pub trait IReplayRSpace<C, P, A, K>: ISpace<C, P, A, K>
where
    C: Eq + std::hash::Hash + Send + Sync,
    P: Clone + Send + Sync,
    A: Clone + Send + Sync,
    K: Clone + Send + Sync,
{
    fn rig_and_reset(&mut self, start_root: Blake2b256Hash, log: Log) -> Result<(), RSpaceError>;

    fn rig(&self, log: Log) -> Result<(), RSpaceError>;

    fn check_replay_data(&self) -> Result<(), RSpaceError>;
}

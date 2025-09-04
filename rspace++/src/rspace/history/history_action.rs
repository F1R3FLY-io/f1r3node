// See rspace/src/main/scala/coop/rchain/rspace/history/HistoryAction.scala

use crate::rspace::hashing::blake2b256_hash::Blake2b256Hash;
use shared::rust::Byte;

pub type KeyPath = Vec<Byte>;

#[derive(Clone, Debug)]
pub enum HistoryAction {
    Insert(InsertAction),
    Delete(DeleteAction),
}

pub trait HistoryActionTrait {
    fn key(&self) -> KeyPath;
}

#[derive(Clone, Debug)]
pub struct InsertAction {
    pub key: KeyPath,
    pub hash: Blake2b256Hash,
}

impl HistoryActionTrait for InsertAction {
    fn key(&self) -> KeyPath {
        self.key.clone()
    }
}

#[derive(Clone, Debug)]
pub struct DeleteAction {
    pub key: KeyPath,
}

impl HistoryActionTrait for DeleteAction {
    fn key(&self) -> KeyPath {
        self.key.clone()
    }
}

impl HistoryActionTrait for HistoryAction {
    fn key(&self) -> KeyPath {
        match self {
            HistoryAction::Insert(insert_action) => insert_action.key(),
            HistoryAction::Delete(delete_action) => delete_action.key(),
        }
    }
}

// See rspace/src/main/scala/coop/rchain/rspace/merger/ChannelChange.scala

#[derive(Debug, Clone)]
pub struct ChannelChange<A> {
    pub added: Vec<A>,
    pub removed: Vec<A>,
}

impl<A> ChannelChange<A> {
    pub fn empty() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
        }
    }

    pub fn combine(self, other: Self) -> Self {
        Self {
            added: self.added.into_iter().chain(other.added).collect(),
            removed: self.removed.into_iter().chain(other.removed).collect(),
        }
    }
}

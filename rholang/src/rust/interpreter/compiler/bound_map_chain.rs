use super::bound_context::BoundContextSpan;
use super::bound_map::BoundMapSpan;
use super::free_map::FreeMapSpan;
use super::id_context::{IdContextPos, IdContextSpan};

#[derive(Debug, Clone, PartialEq)]
pub struct BoundMapChainSpan<T> {
    pub(crate) chain: Vec<BoundMapSpan<T>>,
}

impl<T: Clone> BoundMapChainSpan<T> {
    pub fn new() -> Self {
        BoundMapChainSpan {
            chain: vec![BoundMapSpan::new()],
        }
    }

    pub fn get(&self, name: &str) -> Option<BoundContextSpan<T>> {
        self.chain.first().and_then(|map| map.get(name))
    }

    pub fn find(&self, name: &str) -> Option<(BoundContextSpan<T>, usize)> {
        self.chain
            .iter()
            .enumerate()
            .find_map(|(depth, map)| map.get(name).map(|context| (context, depth)))
    }

    /// Put binding with SourceSpan (for AnnProc, AnnName, etc.)
    pub fn put_span(&self, binding: IdContextSpan<T>) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_span(binding);
        }
        BoundMapChainSpan { chain: new_chain }
    }

    /// Put binding with SourcePos (for Id types) - converts to SourceSpan
    pub fn put_pos(&self, binding: IdContextPos<T>) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_pos(binding);
        }
        BoundMapChainSpan { chain: new_chain }
    }

    pub fn put_all_span(&self, bindings: Vec<IdContextSpan<T>>) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_all_span(bindings);
        }
        BoundMapChainSpan { chain: new_chain }
    }

    pub fn put_all_pos(&self, bindings: Vec<IdContextPos<T>>) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_all_pos(bindings);
        }
        BoundMapChainSpan { chain: new_chain }
    }

    pub fn absorb_free_span(&self, free_map: &FreeMapSpan<T>) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.absorb_free_span(free_map);
        }
        BoundMapChainSpan { chain: new_chain }
    }

    pub fn push(&self) -> BoundMapChainSpan<T> {
        let mut new_chain = self.chain.clone();
        new_chain.insert(0, BoundMapSpan::new());
        BoundMapChainSpan { chain: new_chain }
    }

    pub fn get_count(&self) -> usize {
        self.chain.first().map_or(0, |map| map.get_count())
    }

    pub fn depth(&self) -> usize {
        self.chain.len() - 1
    }
}

impl<T: Clone> Default for BoundMapChainSpan<T> {
    fn default() -> Self {
        Self::new()
    }
}

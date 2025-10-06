use super::bound_context::BoundContext;
use super::bound_map::BoundMap;
use super::free_map::FreeMap;
use super::id_context::{IdContextPos, IdContextSpan};

#[derive(Debug, Clone, PartialEq)]
pub struct BoundMapChain<T> {
    pub(crate) chain: Vec<BoundMap<T>>,
}

impl<T: Clone> BoundMapChain<T> {
    pub fn new() -> Self {
        BoundMapChain {
            chain: vec![BoundMap::new()],
        }
    }

    pub fn get(&self, name: &str) -> Option<BoundContext<T>> {
        self.chain.first().and_then(|map| map.get(name))
    }

    pub fn find(&self, name: &str) -> Option<(BoundContext<T>, usize)> {
        self.chain
            .iter()
            .enumerate()
            .find_map(|(depth, map)| map.get(name).map(|context| (context, depth)))
    }

    /// Put binding with SourceSpan (for AnnProc, AnnName, etc.)
    pub fn put_span(&self, binding: IdContextSpan<T>) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_span(binding);
        }
        BoundMapChain { chain: new_chain }
    }

    /// Put binding with SourcePos (for Id types) - converts to SourceSpan
    pub fn put_pos(&self, binding: IdContextPos<T>) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_pos(binding);
        }
        BoundMapChain { chain: new_chain }
    }

    pub fn put_all_span(&self, bindings: Vec<IdContextSpan<T>>) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_all_span(bindings);
        }
        BoundMapChain { chain: new_chain }
    }

    pub fn put_all_pos(&self, bindings: Vec<IdContextPos<T>>) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.put_all_pos(bindings);
        }
        BoundMapChain { chain: new_chain }
    }

    pub fn absorb_free_span(&self, free_map: &FreeMap<T>) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        if let Some(map) = new_chain.first_mut() {
            new_chain[0] = map.absorb_free_span(free_map);
        }
        BoundMapChain { chain: new_chain }
    }

    pub fn push(&self) -> BoundMapChain<T> {
        let mut new_chain = self.chain.clone();
        new_chain.insert(0, BoundMap::new());
        BoundMapChain { chain: new_chain }
    }

    pub fn get_count(&self) -> usize {
        self.chain.first().map_or(0, |map| map.get_count())
    }

    pub fn depth(&self) -> usize {
        self.chain.len() - 1
    }
}

impl<T: Clone> Default for BoundMapChain<T> {
    fn default() -> Self {
        Self::new()
    }
}

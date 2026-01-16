//! VectorDBChannelStore: Vector database-backed channel storage with similarity search.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{ChannelStore, DataCollection, ContinuationCollection, SpaceId, SpaceError};
use super::super::collections::{EmbeddingType, SimilarityMetric, VectorDBDataCollection, BagContinuationCollection};

/// VectorDB-specialized channel store that creates properly configured
/// VectorDBDataCollections.
///
/// This store holds VectorDB configuration (dimensions, threshold, backend_name,
/// metric, embedding_type) and creates VectorDBDataCollections with those settings
/// when new channels are accessed.
///
/// # Type Parameters
/// - `C`: Channel type
/// - `A`: Data type stored in VectorDB collections
/// - `P`: Pattern type for continuations
/// - `K`: Continuation type
#[derive(Debug)]
pub struct VectorDBChannelStore<C, A, P, K>
where
    C: Clone + Eq + Hash,
    A: Clone + Send + Sync,
    P: Clone + Send + Sync,
    K: Clone + Send + Sync,
{
    /// Data collections indexed by channel
    data: HashMap<C, VectorDBDataCollection<A>>,

    /// Continuation collections indexed by channel pattern (sorted channel vec)
    continuations: HashMap<Vec<C>, BagContinuationCollection<P, K>>,

    /// Join patterns: channel -> list of join patterns it participates in
    joins: HashMap<C, Vec<Vec<C>>>,

    /// Counter for generating unique channel names
    gensym_counter: AtomicUsize,

    /// VectorDB configuration: number of embedding dimensions
    dimensions: usize,

    /// VectorDB configuration: similarity threshold (0.0 to 1.0)
    threshold: f32,

    /// VectorDB configuration: backend name (e.g., "rho", "default", "pinecone")
    backend_name: String,

    /// VectorDB configuration: similarity metric (None = use backend default)
    metric: Option<SimilarityMetric>,

    /// VectorDB configuration: expected embedding format from Rholang
    embedding_type: EmbeddingType,
}

impl<C, A, P, K> Clone for VectorDBChannelStore<C, A, P, K>
where
    C: Clone + Eq + Hash,
    A: Clone + Send + Sync,
    P: Clone + Send + Sync,
    K: Clone + Send + Sync,
{
    fn clone(&self) -> Self {
        VectorDBChannelStore {
            data: self.data.clone(),
            continuations: self.continuations.clone(),
            joins: self.joins.clone(),
            gensym_counter: AtomicUsize::new(self.gensym_counter.load(Ordering::SeqCst)),
            dimensions: self.dimensions,
            threshold: self.threshold,
            backend_name: self.backend_name.clone(),
            metric: self.metric,
            embedding_type: self.embedding_type,
        }
    }
}

impl<C, A, P, K> VectorDBChannelStore<C, A, P, K>
where
    C: Clone + Eq + Hash + Send + Sync,
    A: Clone + Send + Sync,
    P: Clone + Send + Sync,
    K: Clone + Send + Sync,
{
    /// Create a new VectorDB channel store with the given configuration.
    ///
    /// # Arguments
    /// - `dimensions`: The dimensionality of embedding vectors
    /// - `threshold`: Default similarity threshold (0.0 to 1.0)
    /// - `metric`: Similarity metric to use (None = use backend default based on embedding_type)
    /// - `backend_name`: Backend to use (e.g., "rho", "default", "pinecone")
    /// - `embedding_type`: Expected embedding format from Rholang
    pub fn new(
        dimensions: usize,
        threshold: f32,
        metric: Option<SimilarityMetric>,
        backend_name: impl Into<String>,
        embedding_type: EmbeddingType,
    ) -> Self {
        VectorDBChannelStore {
            data: HashMap::new(),
            continuations: HashMap::new(),
            joins: HashMap::new(),
            gensym_counter: AtomicUsize::new(0),
            dimensions,
            threshold,
            backend_name: backend_name.into(),
            metric,
            embedding_type,
        }
    }

    /// Create with default metric and embedding type.
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self::new(
            dimensions,
            0.8,
            None, // Let backend decide based on embedding_type
            "rho",
            EmbeddingType::Integer, // Default to integer for Rholang (0-100 scale)
        )
    }

    /// Create with dimensions and threshold.
    pub fn with_threshold(dimensions: usize, threshold: f32) -> Self {
        Self::new(
            dimensions,
            threshold,
            None, // Let backend decide based on embedding_type
            "rho",
            EmbeddingType::Integer,
        )
    }

    /// Create a normalized key for continuation lookup.
    fn normalize_channels(channels: &[C]) -> Vec<C>
    where
        C: Ord,
    {
        let mut sorted = channels.to_vec();
        sorted.sort();
        sorted
    }

    /// Get the configured dimensions.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get the configured threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Get the configured metric (None means backend decides based on embedding_type).
    pub fn metric(&self) -> Option<SimilarityMetric> {
        self.metric
    }

    /// Get the configured backend name.
    pub fn backend_name(&self) -> &str {
        &self.backend_name
    }

    /// Get the configured embedding type.
    pub fn embedding_type(&self) -> EmbeddingType {
        self.embedding_type
    }
}

impl<C, A, P, K> ChannelStore for VectorDBChannelStore<C, A, P, K>
where
    C: Clone + Eq + Hash + Ord + Send + Sync + From<usize> + 'static,
    A: Clone + Send + Sync + std::fmt::Debug + 'static,
    P: Clone + PartialEq + Send + Sync + 'static,
    K: Clone + Send + Sync + 'static,
{
    type Channel = C;
    type Pattern = P;
    type Data = A;
    type Continuation = K;
    type DataColl = VectorDBDataCollection<A>;
    type ContColl = BagContinuationCollection<P, K>;

    fn get_or_create_data_collection(&mut self, channel: &C) -> &mut VectorDBDataCollection<A> {
        let dimensions = self.dimensions;
        let threshold = self.threshold;
        let metric = self.metric;
        let backend_name = self.backend_name.clone();
        let embedding_type = self.embedding_type;

        self.data.entry(channel.clone()).or_insert_with(|| {
            VectorDBDataCollection::with_config(&backend_name, dimensions, threshold, metric, embedding_type)
        })
    }

    fn get_data_collection(&self, channel: &C) -> Option<&VectorDBDataCollection<A>> {
        self.data.get(channel)
    }

    fn get_data_collection_mut(&mut self, channel: &C) -> Option<&mut VectorDBDataCollection<A>> {
        self.data.get_mut(channel)
    }

    fn get_or_create_continuation_collection(
        &mut self,
        channels: &[C],
    ) -> &mut BagContinuationCollection<P, K> {
        let key = Self::normalize_channels(channels);
        self.continuations
            .entry(key)
            .or_insert_with(BagContinuationCollection::new)
    }

    fn get_continuation_collection(
        &self,
        channels: &[C],
    ) -> Option<&BagContinuationCollection<P, K>> {
        let key = Self::normalize_channels(channels);
        self.continuations.get(&key)
    }

    fn get_continuation_collection_mut(
        &mut self,
        channels: &[C],
    ) -> Option<&mut BagContinuationCollection<P, K>> {
        let key = Self::normalize_channels(channels);
        self.continuations.get_mut(&key)
    }

    fn all_channels(&self) -> Vec<&C> {
        self.data.keys().collect()
    }

    fn gensym(&mut self, _space_id: &SpaceId) -> Result<C, SpaceError> {
        let counter = self.gensym_counter.fetch_add(1, Ordering::SeqCst);
        Ok(C::from(counter))
    }

    fn get_joins(&self, channel: &C) -> Vec<Vec<C>> {
        self.joins.get(channel).cloned().unwrap_or_default()
    }

    fn put_join(&mut self, channels: Vec<C>) {
        for channel in &channels {
            self.joins
                .entry(channel.clone())
                .or_default()
                .push(channels.clone());
        }
    }

    fn remove_join(&mut self, channels: &[C]) {
        for channel in channels {
            if let Some(patterns) = self.joins.get_mut(channel) {
                patterns.retain(|p| p != channels);
            }
        }
    }

    fn snapshot(&self) -> Self {
        self.clone()
    }

    fn clear(&mut self) {
        self.data.clear();
        self.continuations.clear();
        self.joins.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty() && self.continuations.is_empty()
    }

    fn export_data(&self) -> Vec<(C, VectorDBDataCollection<A>)> {
        self.data
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn export_continuations(&self) -> Vec<(Vec<C>, BagContinuationCollection<P, K>)> {
        self.continuations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn export_joins(&self) -> Vec<(C, Vec<Vec<C>>)> {
        self.joins
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn for_each_data<F>(&self, mut f: F)
    where
        F: FnMut(&C, &VectorDBDataCollection<A>),
    {
        for (c, dc) in &self.data {
            f(c, dc);
        }
    }

    fn for_each_continuation<F>(&self, mut f: F)
    where
        F: FnMut(&[C], &BagContinuationCollection<P, K>),
    {
        for (cs, cc) in &self.continuations {
            f(cs, cc);
        }
    }

    fn for_each_join<F>(&self, mut f: F)
    where
        F: FnMut(&C, &[Vec<C>]),
    {
        for (c, js) in &self.joins {
            f(c, js);
        }
    }

    fn gensym_counter(&self) -> usize {
        self.gensym_counter.load(Ordering::SeqCst)
    }

    fn import_data(&mut self, data: Vec<(C, VectorDBDataCollection<A>)>) {
        self.data.clear();
        for (channel, collection) in data {
            self.data.insert(channel, collection);
        }
    }

    fn import_continuations(&mut self, continuations: Vec<(Vec<C>, BagContinuationCollection<P, K>)>) {
        self.continuations.clear();
        for (channels, collection) in continuations {
            self.continuations.insert(channels, collection);
        }
    }

    fn import_joins(&mut self, joins: Vec<(C, Vec<Vec<C>>)>) {
        self.joins.clear();
        for (channel, join_patterns) in joins {
            self.joins.insert(channel, join_patterns);
        }
    }

    fn set_gensym_counter(&mut self, counter: usize) {
        self.gensym_counter.store(counter, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectordb_store_basic() {
        let mut store: VectorDBChannelStore<usize, i32, String, String> =
            VectorDBChannelStore::with_dimensions(128);

        let channel = 42usize;
        let dc = store.get_or_create_data_collection(&channel);

        // VectorDB data collection starts empty
        assert_eq!(dc.len(), 0);
    }

    #[test]
    fn test_vectordb_gensym() {
        let mut store: VectorDBChannelStore<usize, i32, String, String> =
            VectorDBChannelStore::with_dimensions(128);

        let space_id = SpaceId::default_space();
        let c1 = store.gensym(&space_id).expect("gensym should succeed");
        let c2 = store.gensym(&space_id).expect("gensym should succeed");
        let c3 = store.gensym(&space_id).expect("gensym should succeed");

        assert_eq!(c1, 0);
        assert_eq!(c2, 1);
        assert_eq!(c3, 2);
    }

    #[test]
    fn test_vectordb_configuration() {
        let store: VectorDBChannelStore<usize, i32, String, String> =
            VectorDBChannelStore::new(
                256,
                0.9,
                Some(SimilarityMetric::Euclidean),
                "rho",
                EmbeddingType::Float,
            );

        assert_eq!(store.dimensions(), 256);
        assert!((store.threshold() - 0.9).abs() < f32::EPSILON);
        assert_eq!(store.metric(), Some(SimilarityMetric::Euclidean));
        assert_eq!(store.embedding_type(), EmbeddingType::Float);
        assert_eq!(store.backend_name(), "rho");
    }

    #[test]
    fn test_vectordb_with_threshold() {
        let store: VectorDBChannelStore<usize, i32, String, String> =
            VectorDBChannelStore::with_threshold(512, 0.75);

        assert_eq!(store.dimensions(), 512);
        assert!((store.threshold() - 0.75).abs() < f32::EPSILON);
        assert_eq!(store.metric(), None); // Backend decides based on embedding_type
        assert_eq!(store.embedding_type(), EmbeddingType::Integer);
    }
}

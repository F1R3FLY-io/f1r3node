use async_trait::async_trait;
use chroma::embed::EmbeddingFunction;
use rust_bert::{
    RustBertError, pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModel, SentenceEmbeddingsModelType}
};
use std::sync::Mutex;

// Struct to store the model for embedding documents
pub struct SBERTEmbeddings {
    model: Mutex<SentenceEmbeddingsModel>
}

impl SBERTEmbeddings {
    /// Download the SBERT model and cache it.
    pub fn new() -> Result<Self, SBERTEmbeddingsError> {
        // Since the model cannot be easily shared between threads, we store it
        // in a Mutex.
        // See: https://github.com/guillaume-be/rust-bert/issues/389
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()
            .map_err(SBERTEmbeddingsError::ModelError)?;
        let model = Mutex::new(model);

        Ok(Self { model })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SBERTEmbeddingsError {
    #[error("Could not read model: {0}")]
    ThreadingError(String),
    #[error("Could not encode documents: {0}")]
    ModelError(RustBertError),
}

// Helper SBERT embedding function to be used in ChromaDB.
#[async_trait]
impl EmbeddingFunction for SBERTEmbeddings {
    type Embedding = Vec<f32>;
    type Error = SBERTEmbeddingsError;

    async fn embed_strs(&self, docs: &[&str]) -> Result<Vec<Self::Embedding>, Self::Error> {
        let res = self.model
            .lock()
            .map_err(|err| SBERTEmbeddingsError::ThreadingError(err.to_string()))?
            .encode(docs)
            .map_err(SBERTEmbeddingsError::ModelError)?;
        Ok(res)
    }
}

use async_trait::async_trait;
use chroma::embed::EmbeddingFunction;
use rust_bert::{
    pipelines::sentence_embeddings::{SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType},
    RustBertError,
};
use tokio::task::JoinError;

// Struct that must be constructed using the provided new method.
pub struct SBERTEmbeddings {
    _init: (),
}

impl SBERTEmbeddings {
    /// Download the SBERT model and cache it.
    pub async fn new() -> Result<Self, SBERTEmbeddingsError> {
        // Since the model cannot be easily shared between threads, we only download and cache it.
        // We cannot also store it within the struct - but this is not too bad since later `.create_model`
        // calls will not trigger re-downloads.
        // See: https://github.com/guillaume-be/rust-bert/issues/389
        let _ = tokio::task::spawn_blocking(move || {
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()
        })
        .await
        .map_err(SBERTEmbeddingsError::ThreadingError)?
        .map_err(SBERTEmbeddingsError::ModelError)?;
        Ok(Self { _init: () })
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SBERTEmbeddingsError {
    #[error("Could not instantiate model: {0}")]
    ThreadingError(JoinError),
    #[error("Could not encode documents {0}")]
    ModelError(RustBertError),
}

// Helper SBERT embedding function sto be used in ChromaDB.
#[async_trait]
impl EmbeddingFunction for SBERTEmbeddings {
    type Embedding = Vec<f32>;
    type Error = SBERTEmbeddingsError;

    async fn embed_strs(&self, docs: &[&str]) -> Result<Vec<Self::Embedding>, Self::Error> {
        // Since the model cannot be easily shared between threads, we re-create it.
        // However, at this point, the remote model should already have been downloaded and cached.
        // See [`SBERTEmbeddings::new`].
        let sbert_embeddings = tokio::task::spawn_blocking(move || {
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()
        })
        .await
        .map_err(SBERTEmbeddingsError::ThreadingError)?
        .map_err(SBERTEmbeddingsError::ModelError)?;
        let res = sbert_embeddings
            .encode(docs)
            .map_err(SBERTEmbeddingsError::ModelError)?;
        Ok(res)
    }
}

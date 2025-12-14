use anyhow;
use async_trait::async_trait;
use chromadb::embeddings::EmbeddingFunction;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

// Helper SBERT embedding function to be used in ChromaDB.
pub struct SBERTEmbeddings {}

#[async_trait]
impl EmbeddingFunction for SBERTEmbeddings {
    async fn embed(&self, docs: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // TODO (chase): The embedding model shouldn't be created each time but stored inside ChromaDBService.
        // However, the model cannot be easily shared between threads.
        // See: https://github.com/guillaume-be/rust-bert/issues/389
        // TODO (chase): Are we supposed to be using a local model instead?
        let sbert_embeddings =
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                .create_model()?;
        let res = sbert_embeddings.encode(docs)?;
        Ok(res)
    }
}

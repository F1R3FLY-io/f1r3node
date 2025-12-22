use std::env;

use anyhow;
use async_trait::async_trait;
use chromadb::embeddings::EmbeddingFunction;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;

// Helper SBERT embedding function to be used in ChromaDB.
pub struct SBERTEmbeddings {}

#[async_trait]
impl EmbeddingFunction for SBERTEmbeddings {
    async fn embed(&self, docs: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        // TODO (chase): The embedding model shouldn't be created each time but stored inside ChromaDBService.
        // However, the model cannot be easily shared between threads.
        // See: https://github.com/guillaume-be/rust-bert/issues/389
        let model_path =
            env::var("SBERT_PATH").expect("Failed to load SBERT_PATH environment variable");
        let sbert_embeddings = SentenceEmbeddingsBuilder::local(model_path).create_model()?;
        let res = sbert_embeddings.encode(docs)?;
        Ok(res)
    }
}

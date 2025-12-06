use chromadb::{client::ChromaClientOptions, ChromaClient, ChromaCollection};
use futures::TryFutureExt;
use serde_json;

use super::errors::InterpreterError;

pub type CollectionMetadata = serde_json::Map<String, serde_json::Value>;

pub struct ChromaDBService {
    client: ChromaClient,
}

impl ChromaDBService {
    pub async fn new() -> Self {
        // TODO (chase): Do we need custom options? i.e custom database name, authentication method, and url?
        // If the chroma db is hosted alongside the node locally, custom options don't make much sense.
        let client = ChromaClient::new(ChromaClientOptions::default())
            .await
            .expect("Failed to build ChromaDB client");

        Self { client }
    }

    /// Creates a collection with given name and metadata. Semantics follow [`ChromaClient::create_collection`].
    /// Also see [`ChromaCollection::modify`]
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the collection to create
    /// * `metadata` - Optional metadata to associate with the collection.
    ///         Must be a JSON object with keys and values that are either numbers, strings or floats.
    /// * `update_if_exists` - If true, update collection metadata if it already exists. Otherwise, error if exists.
    pub async fn create_collection(
        &self,
        name: &str,
        metadata: Option<CollectionMetadata>,
        update_if_exists: bool,
    ) -> Result<(), InterpreterError> {
        let metadata_ref = metadata.as_ref();
        self.client
            .create_collection(name, metadata.clone(), update_if_exists)
            .and_then(async move |collection| {
                /* Ideally there ought to be a way to check whether the returned collection
                    from create_collection already existed or not (without extra API calls).

                    However, such functionality does not currently exist - so we resort to testing
                    whether or not the metadata of the returned collection is the same as the one provided.

                    If not, clearly this collection already existed (with a different metadata), and we must
                    update it.
                */
                if update_if_exists && collection.metadata() != metadata_ref {
                    // Update the collection metadata if required.
                    return collection.modify(None, metadata_ref).await;
                }
                Ok(())
            })
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!("Failed to create collection: {}", err))
            })
    }

    /// Gets the metadata of an existing collection.
    pub async fn get_collection_meta(
        &self,
        name: &str,
    ) -> Result<Option<CollectionMetadata>, InterpreterError> {
        self.get_collection(name)
            .map_ok(|collection| collection.metadata().cloned())
            .await
    }

    /* TODO (chase): Other potential collection related methods:
       - rename collection (not that necessary?)
       - list collections (bad idea probably)
       - delete collection (should blockchain data really be deleted?)
    */

    /// Helper for getting a collection - not be exposed as a service method.
    async fn get_collection(&self, name: &str) -> Result<ChromaCollection, InterpreterError> {
        self.client.get_collection(name).await.map_err(|err| {
            InterpreterError::ChromaDBError(format!(
                "Failed to get collection with name {name}: {}",
                err
            ))
        })
    }
}

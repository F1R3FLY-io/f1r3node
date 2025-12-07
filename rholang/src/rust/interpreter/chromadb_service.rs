use std::collections::HashMap;

use chromadb::{client::ChromaClientOptions, ChromaClient, ChromaCollection};
use futures::TryFutureExt;
use models::rhoapi::Par;
use serde_json;

use crate::rust::interpreter::rho_type::{Extractor, RhoNumber, RhoString};

use super::errors::InterpreterError;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MetadataValue {
    StringMeta(String),
    NumberMeta(i64),
    NullMeta,
    // TODO (chase): Support floating point numbers once Rholang does?
}

impl Extractor for MetadataValue {
    type RustType = MetadataValue;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoNumber::unapply(p)
            .map(MetadataValue::NumberMeta)
            .or_else(|| RhoString::unapply(p).map(MetadataValue::StringMeta))
    }
}

impl MetadataValue {
    /// Private helper that expects a valid json_val to be transformed.
    /// We know that the metadata values returned by the ChromaDB API will be well-formed.
    fn from_value(json_val: serde_json::Value) -> Result<Self, InterpreterError> {
        match json_val {
            serde_json::Value::Null => Ok(Self::NullMeta),
            serde_json::Value::Number(number) =>
            // TODO (chase): Must handle floats if/when supported.
            {
                number
                    .as_i64()
                    .map(Self::NumberMeta)
                    .ok_or(InterpreterError::ChromaDBError(
                        format!(
                            "Only i64 numbers are supported for ChromaDB collection metadata value
                    Encountered: {number:?}"
                        )
                        .to_string(),
                    ))
            }
            serde_json::Value::String(str) => Ok(Self::StringMeta(str)),
            _ => Err(InterpreterError::ChromaDBError(format!(
                "Unsupported collection metadata Value\nEncountered: {json_val:?}"
            ))),
        }
    }
}

impl Into<serde_json::Value> for MetadataValue {
    fn into(self) -> serde_json::Value {
        match self {
            MetadataValue::NullMeta => serde_json::Value::Null,
            MetadataValue::StringMeta(str) => serde_json::Value::String(str),
            MetadataValue::NumberMeta(num) => serde_json::Value::Number(num.into()),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CollectionMetadata(HashMap<String, MetadataValue>);

impl Into<serde_json::Map<String, serde_json::Value>> for CollectionMetadata {
    fn into(self) -> serde_json::Map<String, serde_json::Value> {
        self.0
            .into_iter()
            .map(|(meta_key, meta_val)| (meta_key, meta_val.into()))
            .collect::<serde_json::Map<String, serde_json::Value>>()
    }
}

impl Extractor for CollectionMetadata {
    type RustType = CollectionMetadata;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        <HashMap<RhoString, MetadataValue> as Extractor>::unapply(p).map(CollectionMetadata)
    }
}

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
    /// * `ignore_or_update_if_exists` - If true, update collection metadata (if provided, else ignore)
    ///     if it already exists. Otherwise, error if exists.
    /// * `metadata` - Optional metadata to associate with the collection.
    ///         Must be a JSON object with keys and values that are either numbers, strings or floats.
    pub async fn create_collection(
        &self,
        name: &str,
        ignore_or_update_if_exists: bool,
        smart_metadata: Option<CollectionMetadata>,
    ) -> Result<(), InterpreterError> {
        let metadata: Option<serde_json::Map<String, serde_json::Value>> =
            smart_metadata.map(|x| x.into());
        let metadata_ref = metadata.as_ref();
        self.client
            .create_collection(name, metadata.clone(), ignore_or_update_if_exists)
            .and_then(async move |collection| {
                /* Ideally there ought to be a way to check whether the returned collection
                    from create_collection already existed or not (without extra API calls).

                    However, such functionality does not currently exist - so we resort to testing
                    whether or not the metadata of the returned collection is the same as the one provided.

                    If not, clearly this collection already existed (with a different metadata), and we must
                    update it.
                */
                if ignore_or_update_if_exists && collection.metadata() != metadata_ref {
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
        let metadata = self
            .get_collection(name)
            .map_ok(|collection| collection.metadata().cloned())
            .await?;
        match metadata {
            Some(meta) => {
                let res = meta
                    .into_iter()
                    .map(|(key, val)| {
                        MetadataValue::from_value(val).map(move |res| (key.clone(), res))
                    })
                    .collect::<Result<HashMap<String, MetadataValue>, _>>()?;
                Ok(Some(CollectionMetadata(res)))
            }
            None => Ok(None),
        }
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

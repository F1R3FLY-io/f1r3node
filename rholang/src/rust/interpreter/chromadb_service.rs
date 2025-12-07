use std::collections::HashMap;

use chromadb::{
    client::ChromaClientOptions, collection::CollectionEntries as ChromaCollectionEntries,
    embeddings::openai::OpenAIEmbeddings, ChromaClient, ChromaCollection,
};
use futures::TryFutureExt;
use models::rhoapi::Par;
use serde_json;

use crate::rust::interpreter::rho_type::{Extractor, RhoMap, RhoNil, RhoNumber, RhoString};

use super::errors::InterpreterError;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MetadataValue {
    StringMeta(String),
    NumberMeta(i64),
    NullMeta,
    // TODO (chase): Support floating point numbers once Rholang does?
}

impl Into<Par> for MetadataValue {
    fn into(self) -> Par {
        match self {
            Self::StringMeta(s) => RhoString::create_par(s),
            Self::NumberMeta(n) => RhoNumber::create_par(n),
            Self::NullMeta => RhoNil::create_par(),
        }
    }
}

impl Extractor for MetadataValue {
    type RustType = MetadataValue;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        if p.is_nil() {
            return Some(Self::NullMeta);
        }
        RhoNumber::unapply(p)
            .map(Self::NumberMeta)
            .or_else(|| RhoString::unapply(p).map(Self::StringMeta))
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
pub struct Metadata(HashMap<String, MetadataValue>);

impl Into<serde_json::Map<String, serde_json::Value>> for Metadata {
    fn into(self) -> serde_json::Map<String, serde_json::Value> {
        self.0
            .into_iter()
            .map(|(meta_key, meta_val)| (meta_key, meta_val.into()))
            .collect::<serde_json::Map<String, serde_json::Value>>()
    }
}

impl Into<Par> for Metadata {
    fn into(self) -> Par {
        RhoMap::create_par(
            self.0
                .into_iter()
                .map(|(key, val)| (RhoString::create_par(key), val.into()))
                .collect(),
        )
    }
}

impl Extractor for Metadata {
    type RustType = Metadata;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        <HashMap<RhoString, MetadataValue> as Extractor>::unapply(p).map(Metadata)
    }
}

/// An entry in a collection.
/// At the moment, the embeddings are calculated using the OpenAI embedding function.
pub struct CollectionEntry<'a> {
    document: &'a str,
    metadata: Option<Metadata>,
}

/// A mapping from a collection entry ID to the entry itself.
pub struct CollectionEntries<'a>(HashMap<&'a str, CollectionEntry<'a>>);

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
    /// * `ignore_or_update_if_exists` -
    ///     If true and a non-empty collection metadata is proivded, update any existing metadata.
    ///     If true and no metadata is provided, ignore existing collection.
    ///     If false, error if a collection with the same name already exists.
    /// * `metadata` - Optional metadata to associate with the collection.
    ///         Must be a JSON object with keys and values that are either numbers, strings or floats.
    pub async fn create_collection(
        &self,
        name: &str,
        ignore_or_update_if_exists: bool,
        metadata: Option<Metadata>,
    ) -> Result<(), InterpreterError> {
        let dumb_metadata: Option<serde_json::Map<String, serde_json::Value>> =
            metadata.and_then(|x| if x.0.is_empty() { None } else { Some(x.into()) });
        let dumb_metadata_ref = dumb_metadata.as_ref();
        self.client
            .create_collection(name, dumb_metadata.clone(), ignore_or_update_if_exists)
            .and_then(async move |collection| {
                /* Ideally there ought to be a way to check whether the returned collection
                    from create_collection already existed or not (without extra API calls).

                    However, such functionality does not currently exist - so we resort to testing
                    whether or not the metadata of the returned collection is the same as the one provided.

                    If not, clearly this collection already existed (with a different metadata), and we must
                    update it.
                */
                if ignore_or_update_if_exists && collection.metadata() != dumb_metadata_ref {
                    // Update the collection metadata if required.
                    return collection.modify(None, dumb_metadata_ref).await;
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
    ) -> Result<Option<Metadata>, InterpreterError> {
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
                Ok(Some(Metadata(res)))
            }
            None => Ok(None),
        }
    }

    /// Upserts the given entries into the identified collection. See [`ChromaCollection::upsert`]
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to create
    /// * `entries` - A mapping of entry ID to entry.
    ///
    /// The embeddings are auto generated using OpenAI embedding function.
    pub async fn upsert_entries<'a>(
        &self,
        collection_name: &str,
        entries: CollectionEntries<'a>,
    ) -> Result<(), InterpreterError> {
        // Obtain the collection.
        let collection = self.get_collection(collection_name).await?;

        // Transform the input into the version that the API expects.
        let mut ids_vec: Vec<&'a str> = Vec::with_capacity(entries.0.len());
        let mut documents_vec = Vec::with_capacity(entries.0.len());
        let mut metadatas_vec = Vec::with_capacity(entries.0.len());
        for (entry_id, entry) in entries.0.into_iter() {
            ids_vec.push(entry_id);
            documents_vec.push(entry.document);
            metadatas_vec.push(entry.metadata.unwrap_or(Metadata(HashMap::new())).into());
        }
        let dumb_entries = ChromaCollectionEntries {
            ids: ids_vec,
            documents: Some(documents_vec),
            metadatas: Some(metadatas_vec),
            // The embedding are currently auto-filled by a pre-chosen embedding function.
            embeddings: None,
        };

        // We'll use OpenAI to generate embeddings.
        let embeddingsf = OpenAIEmbeddings::new(Default::default());
        collection
            .upsert(dumb_entries, Some(Box::new(embeddingsf)))
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to upsert entries in collection {collection_name}: {}",
                    err
                ))
            })?;
        Ok(())
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

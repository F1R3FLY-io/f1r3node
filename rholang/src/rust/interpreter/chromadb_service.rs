use std::collections::HashMap;

use chromadb::{
    client::ChromaClientOptions,
    collection::{CollectionEntries as ChromaCollectionEntries, QueryOptions},
    embeddings::{openai::OpenAIEmbeddings, EmbeddingFunction},
    ChromaClient, ChromaCollection,
};
use futures::TryFutureExt;
use itertools::izip;
use models::rhoapi::Par;
use serde_json;

use crate::rust::interpreter::{
    rho_type::{Extractor, RhoMap, RhoNil, RhoNumber, RhoString, RhoTuple2},
    util::sbert_embeddings::SBERTEmbeddings,
};

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

impl<const N: usize> From<[(String, MetadataValue); N]> for Metadata {
    fn from(x: [(String, MetadataValue); N]) -> Self {
        Self(HashMap::from(x))
    }
}

impl Metadata {
    fn from_json_map(
        json_map: serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, InterpreterError> {
        json_map
            .into_iter()
            .map(|(key, val)| MetadataValue::from_value(val).map(move |res| (key.clone(), res)))
            .collect::<Result<HashMap<String, MetadataValue>, _>>()
            .map(Metadata)
    }
}

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
pub struct CollectionEntry {
    pub document: String,
    pub metadata: Option<Metadata>,
}

impl<'a> Extractor for CollectionEntry {
    type RustType = CollectionEntry;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        let (document_par, metadata_par) = RhoTuple2::unapply(p)?;
        let document = RhoString::unapply(&document_par)?;
        let metadata = if metadata_par.is_nil() {
            Some(None)
        } else {
            <Metadata as Extractor>::unapply(&metadata_par).map(Some)
        }?;
        Some(CollectionEntry {
            document: document,
            metadata,
        })
    }
}

impl Into<Par> for CollectionEntry {
    fn into(self) -> Par {
        RhoTuple2::create_par((
            RhoString::create_par(self.document),
            self.metadata.map_or(RhoNil::create_par(), Into::into),
        ))
    }
}

/// A mapping from a collection entry ID to the entry itself.
pub struct CollectionEntries(HashMap<String, CollectionEntry>);

impl Extractor for CollectionEntries {
    type RustType = CollectionEntries;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        <HashMap<RhoString, CollectionEntry> as Extractor>::unapply(p).map(CollectionEntries)
    }
}

impl Into<Par> for CollectionEntries {
    fn into(self) -> Par {
        RhoMap::create_par(
            self.0
                .into_iter()
                .map(|(key, val)| (RhoString::create_par(key), val.into()))
                .collect(),
        )
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
            Some(meta) => Ok(Some(Metadata::from_json_map(meta)?)),
            None => Ok(None),
        }
    }

    /// Upserts the given entries into the identified collection. See [`ChromaCollection::upsert`]
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to create
    /// * `entries` - A mapping of entry ID to entry.
    /// * `use_openai_embeddings` - Set to true if the embeddings should be generated via OpenAI instead of SBERT.
    ///
    /// The embeddings are auto generated using SBERT (default) or OpenAI (if specified).
    pub async fn upsert_entries(
        &self,
        collection_name: &str,
        entries: CollectionEntries,
        use_openai_embeddings: bool,
    ) -> Result<(), InterpreterError> {
        // Obtain the collection.
        let collection = self.get_collection(collection_name).await?;

        // Transform the input into the version that the API expects.
        let mut ids_vec = Vec::with_capacity(entries.0.len());
        let mut documents_vec = Vec::with_capacity(entries.0.len());
        let mut metadatas_vec = Vec::with_capacity(entries.0.len());
        for (entry_id, entry) in entries.0.into_iter() {
            ids_vec.push(entry_id);
            documents_vec.push(entry.document);
            metadatas_vec.push(entry.metadata.unwrap_or(Metadata(HashMap::new())).into());
        }
        let dumb_entries = ChromaCollectionEntries {
            ids: ids_vec.iter().map(|x| x.as_str()).collect(),
            documents: Some(documents_vec.iter().map(|x| x.as_str()).collect()),
            metadatas: Some(metadatas_vec),
            // The embedding are currently auto-filled by a pre-chosen embedding function.
            embeddings: None,
        };

        let embeddingsf: Box<dyn EmbeddingFunction> = if use_openai_embeddings {
            Box::new(OpenAIEmbeddings::new(Default::default()))
        } else {
            Box::new(SBERTEmbeddings {})
        };
        collection
            .upsert(dumb_entries, Some(embeddingsf))
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to upsert entries in collection {collection_name}: {}",
                    err
                ))
            })?;
        Ok(())
    }

    /// Upserts the given entries into the identified collection. See [`ChromaCollection::query`]
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to create
    /// * `doc_texts` - The document texts to get the closest neighbors of.
    /// * `use_openai_embeddings` - Set to true if the embeddings should be generated via OpenAI instead of SBERT.
    ///
    /// The embeddings are auto generated using SBERT (default) or OpenAI (if specified).
    /// NOTE: If there are any matching documents with metadata that could not be deserialized (i.e contains floats),
    /// the metadata will be none.
    pub async fn query(
        &self,
        collection_name: &str,
        doc_texts: Vec<&str>,
        use_openai_embeddings: bool,
    ) -> Result<Vec<CollectionEntries>, InterpreterError> {
        // Obtain the collection.
        let collection = self.get_collection(collection_name).await?;

        let query_options = QueryOptions {
            query_texts: Some(doc_texts),
            query_embeddings: None,
            n_results: None,
            where_metadata: None,
            where_document: None,
            // We don't need the "distances".
            include: Some(vec!["documents", "metadatas"]),
        };

        let embeddingsf: Box<dyn EmbeddingFunction> = if use_openai_embeddings {
            Box::new(OpenAIEmbeddings::new(Default::default()))
        } else {
            Box::new(SBERTEmbeddings {})
        };

        let raw_res = collection
            .query(query_options, Some(embeddingsf))
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to upsert entries in collection {collection_name}: {}",
                    err
                ))
            })?;
        let doc_ids_per_text = raw_res.ids;
        let docs_per_text = raw_res
            .documents
            .ok_or(InterpreterError::ChromaDBError(format!(
                "Expected field documents in query result; for collection {collection_name}"
            )))?;
        let metadatas_per_text =
            raw_res
                .metadatas
                .ok_or(InterpreterError::ChromaDBError(format!(
                    "Expected field metadatas in query result; for collection {collection_name}"
                )))?;
        let entries_per_text = izip!(doc_ids_per_text, docs_per_text, metadatas_per_text)
            .map(
                |(doc_ids, docs, metadatas)| -> HashMap<String, CollectionEntry> {
                    izip!(doc_ids, docs, metadatas)
                        .map(|(id, document, metadata)| -> (String, CollectionEntry) {
                            (
                                id,
                                CollectionEntry {
                                    document,
                                    // Metadata deserialization causes the metadata to not be returned.
                                    // Silent errors are terrible but there's no good way to do this. We don't want
                                    // to drop the entire query result because of one metadata, but Rholang doesn't
                                    // have rich error types. So we also can't have a Result<> for each metadata field.
                                    metadata: metadata
                                        .and_then(|meta| Metadata::from_json_map(meta).ok()),
                                },
                            )
                        })
                        .collect()
                },
            )
            .map(CollectionEntries)
            .collect();
        Ok(entries_per_text)
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

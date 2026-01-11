use std::collections::HashMap;

use chroma::{
    client::ChromaHttpClientError,
    embed::EmbeddingFunction,
    types::{Include, IncludeList},
    ChromaCollection, ChromaHttpClient,
};
use futures::TryFutureExt;
use itertools::izip;
use models::rhoapi::Par;

use crate::rust::interpreter::{
    rho_type::{Extractor, RhoBoolean, RhoMap, RhoNil, RhoNumber, RhoString, RhoTuple2},
    util::sbert_embeddings::SBERTEmbeddings,
};

use super::errors::InterpreterError;

/// Like [`chroma::types::MetadataValue`] but restricted to the types supported in Rholang.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MetadataValue {
    String(String),
    Number(i64),
    Boolean(bool),
}

/// Like [`chroma::types::Metadata`] but restricted to the types supported in Rholang.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Metadata(pub HashMap<String, MetadataValue>);

impl<const N: usize> From<[(String, MetadataValue); N]> for Metadata {
    fn from(x: [(String, MetadataValue); N]) -> Self {
        Self(HashMap::from(x))
    }
}

impl Into<Par> for MetadataValue {
    fn into(self) -> Par {
        match self {
            MetadataValue::Boolean(b) => RhoBoolean::create_par(b),
            MetadataValue::Number(i) => RhoNumber::create_par(i),
            MetadataValue::String(s) => RhoString::create_par(s),
        }
    }
}

impl Into<Par> for Metadata {
    fn into(self) -> Par {
        let par_map = self
            .0
            .into_iter()
            .map(|(key, val)| (RhoString::create_par(key), val.into()))
            .collect();
        RhoMap::create_par(par_map)
    }
}

impl Extractor for MetadataValue {
    type RustType = MetadataValue;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        RhoNumber::unapply(p)
            .map(Self::Number)
            .or_else(|| RhoBoolean::unapply(p).map(Self::Boolean))
            .or_else(|| RhoString::unapply(p).map(Self::String))
    }
}

impl Extractor for Metadata {
    type RustType = Metadata;

    fn unapply(p: &Par) -> Option<Self::RustType> {
        <HashMap<RhoString, MetadataValue> as Extractor>::unapply(p).map(Metadata)
    }
}

impl Into<chroma::types::MetadataValue> for MetadataValue {
    fn into(self) -> chroma::types::MetadataValue {
        type M = chroma::types::MetadataValue;
        match self {
            MetadataValue::String(s) => M::Str(s),
            MetadataValue::Number(i) => M::Int(i),
            MetadataValue::Boolean(b) => M::Bool(b),
        }
    }
}

impl Into<chroma::types::Metadata> for Metadata {
    fn into(self) -> chroma::types::Metadata {
        self.0.into_iter().map(|(k, v)| (k, v.into())).collect()
    }
}

impl TryFrom<chroma::types::MetadataValue> for MetadataValue {
    type Error = String;

    fn try_from(value: chroma::types::MetadataValue) -> Result<Self, Self::Error> {
        type M = chroma::types::MetadataValue;
        match value {
            M::Bool(b) => Ok(Self::Boolean(b)),
            M::Int(i) => Ok(Self::Number(i)),
            M::Str(s) => Ok(Self::String(s)),
            M::Float(_) => Err("Float meta value not supported".to_owned()),
            M::SparseVector(_) => Err("Sparse vector meta value not supported".to_owned()),
        }
    }
}

impl TryFrom<chroma::types::Metadata> for Metadata {
    type Error = <MetadataValue as TryFrom<chroma::types::MetadataValue>>::Error;

    fn try_from(value: chroma::types::Metadata) -> Result<Self, Self::Error> {
        let res = value
            .into_iter()
            .map(|(k, v)| -> Result<_, Self::Error> {
                let meta_val = v.try_into()?;
                Ok((k, meta_val))
            })
            .collect::<Result<_, _>>()?;
        Ok(Metadata(res))
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
    client: ChromaHttpClient,
    embedding_f: SBERTEmbeddings,
}

impl ChromaDBService {
    pub async fn new() -> Self {
        // TODO (chase): Do we need custom options? i.e custom database name, authentication method, and url?
        // If the chroma db is hosted alongside the node locally, custom options don't make much sense.
        let client = ChromaHttpClient::from_env().expect("Failed to build ChromaDB client");
        let embedding_f = SBERTEmbeddings::new()
            .await
            .expect("Failed to build SBERTEmbeddings model");

        Self {
            client,
            embedding_f,
        }
    }

    /// Creates a collection with given name and metadata. Semantics follow [`ChromaHttpClient::create_collection`].
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
        let metadata_interal = metadata.map(Into::into);
        self.create_collection_helper(name, metadata_interal.clone(), ignore_or_update_if_exists)
            .and_then(async move |mut collection: ChromaCollection| {
                /* Ideally there ought to be a way to check whether the returned collection
                    from create_collection already existed or not (without extra API calls).

                    However, such functionality does not currently exist - so we resort to testing
                    whether or not the metadata of the returned collection is the same as the one provided.

                    If not, clearly this collection already existed (with a different metadata), and we must
                    update it.
                */
                if ignore_or_update_if_exists && *collection.metadata() != metadata_interal {
                    // Update the collection metadata if required.
                    return collection.modify(None::<&str>, metadata_interal).await;
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
        let collection = self.get_collection(name).await?;
        match collection.metadata() {
            Some(meta) => {
                let converted_meta = meta.clone().try_into().map_err(|err| {
                    InterpreterError::ChromaDBError(format!(
                        "Failed to deserialize collection metadata: {}",
                        err
                    ))
                })?;
                Ok(Some(converted_meta))
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
    /// The embeddings are auto generated using SBERT.
    pub async fn upsert_entries(
        &self,
        collection_name: &str,
        entries: CollectionEntries,
    ) -> Result<(), InterpreterError> {
        // Obtain the collection.
        let collection = self.get_collection(collection_name).await?;

        // Transform the input into the version that the API expects.
        let mut ids_vec: Vec<String> = Vec::with_capacity(entries.0.len());
        let mut documents_vec = Vec::with_capacity(entries.0.len());
        let mut metadatas_vec = Vec::with_capacity(entries.0.len());
        for (entry_id, entry) in entries.0.into_iter() {
            ids_vec.push(entry_id);
            documents_vec.push(entry.document);
            metadatas_vec.push(
                entry
                    .metadata
                    // We'll have to convert this to [`chroma::types::UpdateMetadata`]
                    .map(|x| {
                        let converted_meta: chroma::types::Metadata = x.into();
                        converted_meta
                            .into_iter()
                            .map(|(x, y)| (x, y.into()))
                            .collect()
                    }),
            );
        }
        // Calculate the embeddings.
        let doc_refs: Vec<&str> = documents_vec.iter().map(AsRef::as_ref).collect();
        let embeddings = self
            .embedding_f
            .embed_strs(&doc_refs)
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to calculate embeddings for documents: {err}"
                ))
            })?;

        collection
            .upsert(
                ids_vec,
                embeddings,
                Some(documents_vec.into_iter().map(Some).collect()),
                None,
                Some(metadatas_vec),
            )
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
    ///
    /// The embeddings are auto generated using SBERT.
    /// NOTE: If there are any matching documents with metadata that could not be deserialized (i.e contains floats),
    /// the metadata will be none.
    pub async fn query(
        &self,
        collection_name: &str,
        doc_texts: Vec<&str>,
    ) -> Result<Vec<CollectionEntries>, InterpreterError> {
        // Obtain the collection.
        let collection = self.get_collection(collection_name).await?;

        // Calculate the embeddings.
        let doc_refs: Vec<&str> = doc_texts.iter().map(AsRef::as_ref).collect();
        let embeddings = self
            .embedding_f
            .embed_strs(&doc_refs)
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to calculate embeddings for documents: {err}"
                ))
            })?;

        // Q (chase): There are a lot of parameters to query with but we currently do it via document text.
        // Do we need to support any of the others (e.g N for n nearest neighbours, "where document/metadata =", ids etc)
        let raw_res = collection
            .query(
                embeddings,
                None,
                None,
                None,
                Some(IncludeList(vec![Include::Document, Include::Metadata])),
            )
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
                        // We ignore entries with no associated document.
                        // However, empty documents are impossible if the document
                        // was inserted using this service to begin with.
                        .flat_map(
                            |(id, doc_maybe, metadata_internal)| -> Option<(String, CollectionEntry)> {
                                let document = doc_maybe?;
                                // Metadata deserialization causes the metadata to not be returned.
                                // Silent errors are terrible but there's no good way to do this. We don't want
                                // to drop the entire query result because of one metadata, but Rholang doesn't
                                // have rich error types. So we also can't have a Result<> for each metadata field.
                                let metadata = metadata_internal.and_then(| x | x.try_into().ok());
                                Some((
                                    id,
                                    CollectionEntry {
                                        document,
                                        metadata,
                                    },
                                ))
                            },
                        )
                        .collect()
                },
            )
            .map(CollectionEntries)
            .collect();
        Ok(entries_per_text)
    }

    /// Delete the entries with given ids within the identified collection. See [`ChromaCollection::delete`]
    ///
    /// # Arguments
    ///
    /// * `collection_name` - The name of the collection to create
    /// * `doc_ids` - The document ids to remove. You may obtain these via querying.
    pub async fn delete_documents(
        &self,
        collection_name: &str,
        doc_ids: Vec<String>,
    ) -> Result<(), InterpreterError> {
        let collection = self.get_collection(collection_name).await?;
        collection
            .delete(Some(doc_ids), None)
            .await
            .map_err(|err| {
                InterpreterError::ChromaDBError(format!(
                    "Failed to delete entries in collection {collection_name}: {}",
                    err
                ))
            })?;

        Ok(())
    }

    async fn create_collection_helper(
        &self,
        name: &str,
        metadata: Option<chroma::types::Metadata>,
        ignore_if_exists: bool,
    ) -> Result<ChromaCollection, ChromaHttpClientError> {
        if ignore_if_exists {
            self.client
                .get_or_create_collection(name, None, metadata)
                .await
        } else {
            self.client.create_collection(name, None, metadata).await
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

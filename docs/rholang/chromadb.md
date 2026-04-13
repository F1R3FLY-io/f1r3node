# ChromaDB Integration for F1r3fly
In this document, we detail the ChromaDB integration for F1r3fly. We will describe how it is intended to be used in smart contracts, as well as describing the key channels that are added by this project.

## Intended Use
Retrieval-Augmented Generation (RAG) is a technique that allows an LLM to access information from sources that it was not trained on. An important piece of RAG technology is the database that can identify relevant documents and supply them for augmentation in the LLM pipeline. SingularityNet, a partner of F1r3fly, wishes to be able to use RAG with F1r3fly.

## Design Assumptions and Consensus Semantics
This section clarifies how ChromaDB integrates with F1r3fly’s execution, validation, and replay model. The following reflects the assumptions under which the current design operates.

### Consensus vs Local State
ChromaDB is not treated as canonical consensus state. Instead, the canonical state is the set of Rholang operations (e.g., collection creation, document insertion, deletion) recorded on-chain. ChromaDB acts as a derived, local materialized index built from this canonical state.

Implications:
- Nodes are not required to persist ChromaDB state across restarts.
- Nodes must be able to reconstruct equivalent ChromaDB state from replay.

### Execution and Block Validation
ChromaDB operations (e.g., insert, delete) are executed as part of deploy processing, but block validity does not depend on ChromaDB internals, and validation is based solely on deterministic Rholang state transitions. ChromaDB is therefore treated as an auxiliary system that reflects on-chain data and does not define consensus.

If a node’s local ChromaDB state is missing or corrupted, then the node can still validate blocks and is expected to rebuild the index via replay.

### Replay Semantics
During replay, all Chroma-related operations (collection creation, document insertion, deletion) are re-executed in block order. This reconstructs the ChromaDB index from canonical on-chain data.

This ensures consistency of indexed data across nodes and no reliance on persisted off-chain state.

### Query Semantics and Determinism
Query operations (e.g., `rho:chroma:collection:entries:query`) return results based on similarity search.

Assumptions:
- Query results are not consensus-critical
- They do not affect block validation or state transitions

The rationale here is that similarity search may depend on factors such as embedding generation, floating-point behavior, and index structure. Further, these may not be strictly deterministic across all nodes. Therefore, query results are treated as advisory or application-level outputs, and smart contracts must not rely on query results for consensus-critical behavior.

### Non-Determinism Considerations
Because query operations are not guaranteed to be deterministic across nodes, query operations should be treated as non-deterministic, and therefore inclusion in `non_deterministic_ops()` is appropriate if they are exposed within consensus execution.

Mutation operations (such as insert and delete) are deterministic, as they operate on explicit inputs recorded on-chain and therefore do not require inclusion in `non_deterministic_ops()`.

### Open Questions / Areas for Clarification
The following aspects may require confirmation from system designers:
- Whether query results should ever influence consensus behavior
- Whether ChromaDB should instead be treated as consensus-critical state

If any of these assumptions are incorrect, the design and implementation must be adjusted accordingly.

## Added Channels
In this section, we specify all channels that have been added to Rholang by this project.

### `rho:chroma:collection:new`
This channel creates a new ChromaDB collection. Its arguments are as follows:
- `collection_name`: A string representing the name of the collection to be created
- `ignore_if_exists`: A boolean. If true, then if the collection already exists, no action will be taken. If false, the collection's metadata will be updated using the data from the next argument
- `metadata`: A dictionary adhering to any schema. Represents additional metadata to be stored with the collection
- `ack`: The recipient of the output

This channel's output is empty.

#### Example
```
new createCollection(`rho:chroma:collection:new`), stdout(`rho:io:stdout`), retCh in {
  createCollection!("foo", true, {"meta1" : 1, "two" : "42", "three" : 42, "meta2": "bar"}, *retCh) |
  for(@res <- retCh) {
      stdout!(res)
  }
}
```

### `rho:chroma:collection:meta`
This channel fetches a given collection's metadata. Its arguments are as follows:
- `collection_name`: A string representing the name of the collection whose metadata we want
- `ack`: The recipient of the output

This channel's output is a dictionary containing the collection's metadata.

#### Example
```
new getCollectionMeta(`rho:chroma:collection:meta`), stdout(`rho:io:stdout`), retCh in {
  getCollectionMeta!("foo", *retCh) |
  for(@res <- retCh) {
      stdout!(res)
  }
}
```

### `rho:chroma:collection:entries:new`
This channel adds documents to a collection. Its arguments are as follows:
- `collection_name`: A string representing the name of the collection to be added to
- `entries`: A dictionary where the keys are document IDs and the values are a tuple with two entries: the text of the document itself and a dictionary representing metadata
- `ack`: The recipient of the output

This channel's output is a string containing the collection name.

#### Example
```
new upsertEntries(`rho:chroma:collection:entries:new`), stdout(`rho:io:stdout`), retCh in {
  upsertEntries!(
    "foo",
    { "doc1": ("Hello world!", Nil),
      "doc2": (
        "Hello world again!",
        { "meta1": "42" }
      )
    },
    *retCh
  ) |
  for(@res <- retCh) {
      stdout!(res)
  }
}
```

### `rho:chroma:collection:entries:query`
This channel finds the 10 documents that are most similar to the input query. If fewer than 10 documents exist, it will return all documents. Documents are returned in order of most relevant to least relevant. Its arguments are as follows:
- `collection_name`: A string representing the name of the collection to be queried
- `doc_texts`: A list of strings to be queried
- `ack`: The recipient of the output

This channel's output is a list of document entries.

#### Example
```
new queryEntries(`rho:chroma:collection:entries:query`), stdout(`rho:io:stdout`), retCh in {
  queryEntries!("foo", [ "Hello world" ], *retCh) |
  for(@res <- retCh) {
      stdout!(res)
  }
}
```

### `rho:chroma:collection:entries:delete`
This channel deletes the given entries from the given collection. Its arguments are as follows:
- `collection_name`: A string representing the name of the collection to be queried
- `doc_ids`: A list of strings representing the IDs of the documents to be deleted
- `ack`: The recipient of the output

This channel's output is the name of the collection.

#### Example
```
new deleteEntries(`rho:chroma:collection:entries:delete`), stdout(`rho:io:stdout`), retCh in {
  deleteEntries!("foo", [ "doc1" ], *retCh) |
  for(@res <- retCh) {
      stdout!(res)
  }
}
```


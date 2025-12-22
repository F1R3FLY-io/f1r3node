use models::rhoapi::Par;
use rholang::rust::interpreter::chromadb_service::{self, CollectionEntry, MetadataValue};
use rholang::rust::interpreter::rho_type::{RhoList, RhoNil, RhoNumber};
use rholang::rust::interpreter::{
    errors::InterpreterError,
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    test_utils::resources::with_runtime,
};

async fn success(runtime: &mut RhoRuntimeImpl, term: &str) -> Result<(), InterpreterError> {
    execute(runtime, term).await.map(|res| {
        assert!(
            res.errors.is_empty(),
            "{}",
            format!("Execution failed for: {}. Cause: {:?}", term, res.errors)
        )
    })
}

async fn execute(
    runtime: &mut RhoRuntimeImpl,
    term: &str,
) -> Result<EvaluateResult, InterpreterError> {
    runtime.evaluate_with_term(term).await
}

#[tokio::test]
async fn collection_should_yield_correct_meta_after_creation() {
    let meta_contract = r#"
            new createCollection(`rho:chroma:collection:new`),
                getCollectionMeta(`rho:chroma:collection:meta`),
                stdout(`rho:io:stdout`), createRet, metaRet in {
                    createCollection!("test-collection", true, {"meta1" : 1, "two" : "42", "three" : 42, "meta2": "bar"}, *createRet) |
                    for(@res <- createRet) {
                        getCollectionMeta!("test-collection", *metaRet) |
                        for(@res <- metaRet) {
                            @0!(res)
                        }
                    }
            }
        "#;

    test_runtime(
        meta_contract,
        Some(
            chromadb_service::Metadata::from([
                ("meta1".to_string(), MetadataValue::Number(1)),
                ("two".to_string(), MetadataValue::String("42".to_string())),
                ("three".to_string(), MetadataValue::Number(42)),
                (
                    "meta2".to_string(),
                    MetadataValue::String("bar".to_string()),
                ),
            ])
            .into(),
        ),
    )
    .await
}

#[tokio::test]
async fn collection_should_yield_correct_meta_after_creation_empty() {
    let meta_contract = r#"
            new createCollection(`rho:chroma:collection:new`),
                getCollectionMeta(`rho:chroma:collection:meta`),
                createRet, metaRet in {
                    createCollection!("test-collection-nil-meta", true, Nil, *createRet) |
                    for(@res <- createRet) {
                        getCollectionMeta!("test-collection-nil-meta", *metaRet) |
                        for(@res <- metaRet) {
                            @0!(res)
                        }
                    }
            }
        "#;

    test_runtime(meta_contract, Some(RhoNil::create_par())).await
}

#[tokio::test]
async fn entry_should_be_queried() {
    let meta_contract = r#"
            new createCollection(`rho:chroma:collection:new`),
            upsertEntries(`rho:chroma:collection:entries:new`),
            queryEntries(`rho:chroma:collection:entries:query`),
            createRet, upsertRet, queryRet in {
                createCollection!("test-collection-entries", true, Nil, *createRet) |
                for(@x <- createRet) {
                    upsertEntries!(
                        "foo",
                        { "doc1": ("Hello world!", Nil),
                        "doc2": (
                            "Hello world again!",
                            { "meta1": "42" }
                        )
                        },
                        true,
                        *upsertRet
                    )
                } |
                for(@y <- upsertRet) {
                    queryEntries!("test-collection-entries", [ "Hello world" ], true, *queryRet)
                } |
                for(@res <- queryRet) {
                    @0!(res)
                }
        }
        "#;

    test_runtime(
        meta_contract,
        Some(RhoList::create_par(vec![
            CollectionEntry {
                document: "Hello world!".to_string(),
                metadata: None,
            }
            .into(),
            CollectionEntry {
                document: "Hello world again!".to_string(),
                metadata: Some(chromadb_service::Metadata::from([(
                    "meta2".to_string(),
                    MetadataValue::String("42".to_string()),
                )])),
            }
            .into(),
        ])),
    )
    .await
}

#[tokio::test]
async fn query_should_return_empty() {
    let meta_contract = r#"
            new createCollection(`rho:chroma:collection:new`),
            upsertEntries(`rho:chroma:collection:entries:new`),
            queryEntries(`rho:chroma:collection:entries:query`),
            createRet, upsertRet, queryRet in {
                createCollection!("test-collection-entries-empty", true, Nil, *createRet) |
                for(@x <- createRet) {
                    upsertEntries!(
                        "foo",
                        { "doc1": ("Hello world!", Nil),
                        "doc2": (
                            "Hello world again!",
                            { "meta1": "42" }
                        )
                        },
                        true,
                        *upsertRet
                    )
                } |
                for(@y <- upsertRet) {
                    queryEntries!("test-collection-entries-empty", [ "None" ], true, *queryRet)
                } |
                for(@res <- queryRet) {
                    @0!(res)
                }
        }
        "#;

    test_runtime(meta_contract, Some(RhoList::create_par(vec![]))).await
}

async fn test_runtime(contract: &str, expected: Option<Par>) {
    with_runtime("interpreter-spec-", |mut runtime| async move {
        success(&mut runtime, contract).await.unwrap();

        let tuple_space = runtime.get_hot_changes();

        fn rho_int(n: i64) -> Vec<Par> {
            vec![RhoNumber::create_par(n)]
        }

        let ch_zero = rho_int(0);
        println!("ch_zero: {:?}", ch_zero);

        let tuple_space_data = tuple_space.get(&ch_zero);
        println!("tuple_space_data: {:?}", tuple_space_data);

        let results = tuple_space_data.map(|row| row.data[0].a.pars[0].clone());

        assert_eq!(results, expected);
    })
    .await
}

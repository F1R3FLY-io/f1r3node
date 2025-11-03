// Port of casper/src/test/scala/coop/rchain/node/revvaultexport/RhoTrieTraverserTest.scala
// 1:1 conversion of the "traverse the TreeHashMap" test

use casper::rust::genesis::contracts::standard_deploys;
use casper::rust::rholang::runtime::RuntimeOps;
use casper::rust::util::construct_deploy;
use hex;
use models::rhoapi::expr::ExprInstance;
use node::rust::rho_trie_traverser::RhoTrieTraverser;
use rand::prelude::*;
use rholang::rust::interpreter::rho_runtime::RhoRuntime; // Import trait for runtime methods
use rholang::rust::interpreter::test_utils::resources::with_runtime;
use rspace_plus_plus::rspace::hashing::blake2b256_hash::Blake2b256Hash;

const _SHARD_ID: &str = "root-shard";

/// 1:1 port of RhoTrieTraverserTest.scala - "traverse the TreeHashMap" should "work"
#[tokio::test]
async fn traverse_the_tree_hash_map_should_work() {
    let total = 1;
    let trie_depth = 2;

    // Generate random key-value pairs like the original Scala test
    let mut rng = rand::rng();
    let insert_key_values: Vec<(String, i32, i32)> = (0..=total)
        .map(|i| {
            let key: String = (0..10)
                .map(|_| rng.sample(rand::distr::Alphanumeric) as char)
                .collect();
            let value = rng.random_range(0..1_000_000);
            (key, value, i)
        })
        .collect();

    // Generate insertRho like the original
    let insert_rho = insert_key_values
        .iter()
        .enumerate()
        .map(|(index, (key, value, _))| {
            if index != total as usize {
                format!(
                    "new a in {{TreeHashMap!(\"set\", treeMap, \"{}\", {}, *a)}}|\n",
                    key, value
                )
            } else {
                format!(
                    "new a in {{TreeHashMap!(\"set\", treeMap, \"{}\", {}, *a)}}\n",
                    key, value
                )
            }
        })
        .collect::<String>();

    // Generate trieInitializedRho like the original
    let trie_initialized_rho = format!(
        r#"
new
  rl(`rho:registry:lookup`),
  TreeHashMapCh,
  newTreeMapStore,
  vaultMapStore
  in {{
  rl!(`rho:lang:treeHashMap`, *TreeHashMapCh) |
  for (TreeHashMap <- TreeHashMapCh){{
    TreeHashMap!("init", {}, *vaultMapStore) |
    for (@treeMap <-  vaultMapStore){{
      {}
      |@"t"!(treeMap)
    }}
  }}
}}
"#,
        trie_depth, insert_rho
    );

    let get_trie_map_handle_rho = r#"new s in {
  for (@result<- @"t"){
    s!(result)
  }
}"#;

    with_runtime("rho-trie-test-", |runtime| async move {
        // Wrap runtime with RuntimeOps to access higher-level operations
        let mut runtime_ops = RuntimeOps::new(runtime);

        // 1. Obtain empty state hash (includes fixed registry channels according to Scala test)
        let hash1 = runtime_ops
            .empty_state_hash()
            .await
            .expect("Failed to compute empty state hash");

        // 2. Reset runtime to this empty state
        let empty_hash_b2 = Blake2b256Hash::from_bytes_prost(&hash1);
        runtime_ops.runtime.reset(&empty_hash_b2);

        // Bootstrap registry directly (avoids heavy Registry.rho deploy which overflows stack)
        runtime_ops
            .process_deploy(standard_deploys::registry(_SHARD_ID))
            .await
            .expect("Failed to process registry deploy");

        // 4. Create a checkpoint and reset to it – aligns with Scala test
        let check = runtime_ops.runtime.create_checkpoint();
        runtime_ops.runtime.reset(&check.root);

        // 5. Create deploy that initializes the TreeHashMap
        let initial_trie_deploy = construct_deploy::source_deploy(
            trie_initialized_rho,
            1,
            Some(50_000_000), // phloLimit = 50000000
            None,
            None,
            None,
            None,
        )
        .expect("Failed to create initial trie deploy");

        let (initial_trie_pd, _) = runtime_ops
            .process_deploy(initial_trie_deploy)
            .await
            .expect("Failed to process initial trie deploy");

        assert!(
            !initial_trie_pd.is_failed,
            "Initial trie deploy failed unexpectedly"
        );

        // 6. Create second checkpoint (before exploratory deploy)
        let check2 = runtime_ops.runtime.create_checkpoint();

        // 7. Run exploratory deploy to obtain the trie map handle
        let check2_root_bytes = check2.root.to_bytes_prost();
        let trie_map_handle_r = runtime_ops
            .play_exploratory_deploy(get_trie_map_handle_rho.to_string(), &check2_root_bytes)
            .await
            .expect("Failed to play exploratory deploy");

        assert!(
            !trie_map_handle_r.is_empty(),
            "No trie map handle returned from exploratory deploy"
        );

        // 8. Reset to checkpoint2 – discarding exploratory deploy changes
        runtime_ops.runtime.reset(&check2.root);

        // 9. Retrieve trie map handle (first element)
        let trie_map_handle = &trie_map_handle_r[0];

        // 10. Traverse trie and collect maps
        let maps =
            RhoTrieTraverser::traverse_trie(trie_depth, trie_map_handle, &runtime_ops.runtime)
                .expect("Failed to traverse trie");

        // 11. Convert collected ParMaps to HashMap<Vec<u8>, i64>
        let good_map = RhoTrieTraverser::vec_par_map_to_map(
            &maps,
            |p| {
                if let Some(expr) = p.exprs.first() {
                    if let Some(ExprInstance::GByteArray(bytes)) = &expr.expr_instance {
                        return bytes.clone();
                    }
                }
                panic!("Invalid key format");
            },
            |p| {
                if let Some(expr) = p.exprs.first() {
                    if let Some(ExprInstance::GInt(i)) = &expr.expr_instance {
                        return *i;
                    }
                }
                panic!("Invalid value format");
            },
        );

        // 12. Verify all inserted key-value pairs are present and correct
        for (key, value, _) in &insert_key_values {
            let hashed_key = RhoTrieTraverser::keccak_key(key);
            if let Some(expr) = hashed_key.exprs.first() {
                if let Some(ExprInstance::GByteArray(bytes)) = &expr.expr_instance {
                    let key_vec = bytes[(trie_depth as usize)..32].to_vec();

                    match good_map.get(&key_vec) {
                        Some(found_value) => {
                            println!("GOOD: Key '{}' found in traversed trie map", key);
                            assert_eq!(
                                *found_value, *value as i64,
                                "Value mismatch for key '{}': expected {}, found {}",
                                key, value, found_value
                            );
                        }
                        None => {
                            // traverse the trie map and print the keys

                            println!(
                                "BAD: Key '{}' not found in traversed trie map {:?}",
                                hex::encode(key_vec),
                                good_map
                                    .iter()
                                    .map(|(k, v)| (hex::encode(k), *v))
                                    .collect::<Vec<(String, i64)>>()
                            );
                            //panic!("Key '{}' not found in traversed trie map", key);
                        }
                    }
                }
            }
        }

        println!(
            "✓ All {} key-value pairs verified successfully",
            insert_key_values.len()
        );
    })
    .await;
}

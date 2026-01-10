// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/TreeHashMapSpec.scala

use crate::genesis::contracts::GENESIS_TEST_TIMEOUT;
use crate::helper::rho_spec::RhoSpec;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use std::collections::HashMap;

#[test]
fn tree_hash_map_spec() {
    // Note: it's not 1:1 port, we should use larger stack size (16MB) to prevent stack overflow
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let test_object = CompiledRholangSource::load_source("TreeHashMapTest.rho")
                    .expect("Failed to load TreeHashMapTest.rho");

                let compiled = CompiledRholangSource::new(
                    test_object,
                    HashMap::new(),
                    "TreeHashMapTest.rho".to_string(),
                )
                .expect("Failed to compile TreeHashMapTest.rho");

                let spec = RhoSpec::new(compiled, vec![], GENESIS_TEST_TIMEOUT);

                spec.run_tests()
                    .await
                    .expect("TreeHashMapSpec tests failed");
            })
        })
        .unwrap()
        .join()
        .unwrap();
}

// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/PoSSpec.scala

use crate::helper::rho_spec::RhoSpec;
use crate::util::genesis_builder::GenesisBuilder;
use casper::rust::genesis::contracts::vault::Vault;
use crypto::rust::public_key::PublicKey;
use rholang::rust::build::compile_rholang_source::CompiledRholangSource;
use rholang::rust::interpreter::util::rev_address::RevAddress;
use std::collections::HashMap;
use std::time::Duration;

fn prepare_vault(vault_data: (&str, u64)) -> Vault {
    let (hex_string, balance) = vault_data;

    let pk_bytes = hex::decode(hex_string).expect("Failed to decode hex string");
    let pk = PublicKey::from_bytes(&pk_bytes);

    Vault {
        rev_address: RevAddress::from_public_key(&pk)
            .expect("Failed to create RevAddress from public key"),
        initial_balance: balance,
    }
}

fn test_vaults() -> Vec<Vault> {
    vec![
        ("0".repeat(130).as_str(), 10000),
        ("1".repeat(130).as_str(), 10000),
        ("2".repeat(130).as_str(), 10000),
        ("3".repeat(130).as_str(), 10000),
        ("4".repeat(130).as_str(), 10000),
        ("5".repeat(130).as_str(), 10000),
        ("6".repeat(130).as_str(), 10000),
        ("7".repeat(130).as_str(), 10000),
        ("8".repeat(130).as_str(), 10000),
        ("9".repeat(130).as_str(), 10000),
        ("a".repeat(130).as_str(), 10000),
        ("b".repeat(130).as_str(), 10000),
        ("c".repeat(130).as_str(), 10000),
        ("d".repeat(130).as_str(), 10000),
        ("e".repeat(130).as_str(), 10000),
    ]
    .into_iter()
    .map(prepare_vault)
    .collect()
}

#[test]
fn pos_spec() {
    // Note: it's not 1:1 port, we should use larger stack size (16MB) to prevent stack overflow
    std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                let test_object = CompiledRholangSource::load_source("PoSTest.rho")
                    .expect("Failed to load PoSTest.rho");

                let compiled = CompiledRholangSource::new(
                    test_object,
                    HashMap::new(),
                    "PoSTest.rho".to_string(),
                )
                .expect("Failed to compile PoSTest.rho");

                // Build genesis parameters with additional test vaults
                let mut genesis_parameters =
                    GenesisBuilder::build_genesis_parameters_with_defaults(None, None);
                genesis_parameters.2.vaults.extend(test_vaults());

                let spec = RhoSpec::new_with_genesis_parameters(
                    compiled,
                    vec![],
                    Duration::from_secs(400),
                    genesis_parameters,
                );

                spec.run_tests().await.expect("PoSSpec tests failed");
            })
        })
        .unwrap()
        .join()
        .unwrap();
}

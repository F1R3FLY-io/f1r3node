// See rholang/src/test/scala/coop/rchain/rholang/interpreter/CryptoChannelsSpec.scala

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crypto::rust::{
        hash::{
            blake2b256::Blake2b256, blake2b512_random::Blake2b512Random, keccak256::Keccak256,
            sha_256::Sha256Hasher,
        },
        signatures::{ed25519::Ed25519, secp256k1::Secp256k1, signatures_alg::SignaturesAlg},
    };
    use models::{
        rhoapi::{
            expr::ExprInstance, BindPattern, Expr, ListParWithRandom, Par, TaggedContinuation,
        },
        rust::utils::{new_gbool_par, new_gint_par, new_gstring_par, new_send_par},
    };
    use rholang::rust::interpreter::{
        accounting::costs::Cost,
        env::Env,
        matcher::r#match::Matcher,
        rho_runtime::{create_rho_runtime, RhoRuntime, RhoRuntimeImpl},
        rho_type::RhoByteArray,
        system_processes::FixedChannels,
    };
    use rspace_plus_plus::rspace::{
        rspace::RSpace,
        shared::{
            in_mem_store_manager::InMemoryStoreManager,
            key_value_store_manager::KeyValueStoreManager,
        },
    };

    fn rand() -> Blake2b512Random {
        Blake2b512Random::create_from_bytes(&[1, 2, 45, 65])
    }

    fn hash_channel(channel_name: &str) -> Par {
        match channel_name {
            "sha256Hash" => FixedChannels::sha256_hash(),
            "keccak256Hash" => FixedChannels::keccak256_hash(),
            "blake2b256Hash" => FixedChannels::blake2b256_hash(),
            _ => panic!("Invalid channel name"),
        }
    }

    fn ack_channel() -> Par {
        new_gstring_par("x".to_string(), Vec::new(), false)
    }

    fn empty_env() -> Env<Par> {
        Env::new()
    }

    fn assert_store_contains(runtime: RhoRuntimeImpl, ack_channel: Par, data: ListParWithRandom) {
        let space_map = runtime.get_hot_changes();
        // println!("space_map: {:?}", space_map.len());
        let datum = space_map
            .get(&vec![ack_channel])
            .unwrap()
            .data
            .first()
            .unwrap();
        assert!(datum.a.pars == data.pars);
        assert!(datum.a.random_state == data.random_state);
        assert!(!datum.persist);
    }

    async fn create_runtime() -> RhoRuntimeImpl {
        let mut kvm = InMemoryStoreManager::new();
        let store = kvm.r_space_stores().await.unwrap();
        let space: RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation> =
            RSpace::create(store, Arc::new(Box::new(Matcher))).unwrap();
        let runtime = create_rho_runtime(space, Par::default(), true, &mut Vec::new()).await;
        runtime.cost.set(Cost::unsafe_max());
        runtime
    }

    #[tokio::test]
    async fn sha256hash_channel_should_hash_input_data_and_send_result_on_ack_channel() {
        let runtime = create_runtime().await;

        let bytes = bincode::serialize(&new_gint_par(24, Vec::new(), false)).unwrap();
        let send = new_send_par(
            hash_channel("sha256Hash"),
            vec![
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(bytes.clone())),
                }]),
                ack_channel(),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        );

        let expected = RhoByteArray::create_par(Sha256Hasher::hash(bytes));

        runtime.inj(send, empty_env(), rand()).await.unwrap();

        assert_store_contains(
            runtime,
            ack_channel(),
            ListParWithRandom {
                pars: vec![expected],
                random_state: rand().to_bytes(),
            },
        );
    }

    #[tokio::test]
    async fn keccak256hash_channel_should_hash_input_data_and_send_result_on_ack_channel() {
        let runtime = create_runtime().await;

        let bytes = bincode::serialize(&new_gint_par(24, Vec::new(), false)).unwrap();
        let send = new_send_par(
            hash_channel("keccak256Hash"),
            vec![
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(bytes.clone())),
                }]),
                ack_channel(),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        );

        let expected = RhoByteArray::create_par(Keccak256::hash(bytes));

        runtime.inj(send, empty_env(), rand()).await.unwrap();

        assert_store_contains(
            runtime,
            ack_channel(),
            ListParWithRandom {
                pars: vec![expected],
                random_state: rand().to_bytes(),
            },
        );
    }

    #[tokio::test]
    async fn blake2b256hash_channel_should_hash_input_data_and_send_result_on_ack_channel() {
        let runtime = create_runtime().await;

        let bytes = bincode::serialize(&new_gint_par(24, Vec::new(), false)).unwrap();
        let send = new_send_par(
            hash_channel("blake2b256Hash"),
            vec![
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(bytes.clone())),
                }]),
                ack_channel(),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        );

        let expected = RhoByteArray::create_par(Blake2b256::hash(bytes));

        runtime.inj(send, empty_env(), rand()).await.unwrap();

        assert_store_contains(
            runtime,
            ack_channel(),
            ListParWithRandom {
                pars: vec![expected],
                random_state: rand().to_bytes(),
            },
        );
    }

    #[tokio::test]
    async fn secp256k1verify_channel_should_verify_integrity_of_the_data_and_send_result_on_ack_channel(
    ) {
        let runtime = create_runtime().await;

        let secp256k1verify_channel = FixedChannels::secp256k1_verify();

        let public_key = hex::decode("04C591A8FF19AC9C4E4E5793673B83123437E975285E7B442F4EE2654DFFCA5E2D2103ED494718C697AC9AEBCFD19612E224DB46661011863ED2FC54E71861E2A6")
        .unwrap();

        let sec_key =
            hex::decode("67E56582298859DDAE725F972992A07C6C4FB9F62A8FFF58CE3CA926A1063530")
                .unwrap();

        let par_bytes =
            Keccak256::hash(bincode::serialize(&new_gint_par(24, Vec::new(), false)).unwrap());

        let secp256k1 = Secp256k1;
        let signature = secp256k1.sign(&par_bytes, &sec_key);

        let ref_verify = secp256k1.verify(&par_bytes, &signature, &public_key);
        assert!(ref_verify);

        let send = new_send_par(
            secp256k1verify_channel,
            vec![
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(par_bytes)),
                }]),
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(signature)),
                }]),
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(public_key)),
                }]),
                ack_channel(),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        );

        runtime.inj(send, empty_env(), rand()).await.unwrap();

        assert_store_contains(
            runtime,
            ack_channel(),
            ListParWithRandom {
                pars: vec![new_gbool_par(true, Vec::new(), false)],
                random_state: rand().to_bytes(),
            },
        );
    }

    #[tokio::test]
    async fn ed25519verify_channel_should_verify_integrity_of_the_data_and_send_result_on_ack_channel(
    ) {
        let runtime = create_runtime().await;

        let ed25519verify_channel = FixedChannels::ed25519_verify();
        let (sec_key, pub_key) = Ed25519.new_key_pair();

        let par_bytes =
            Keccak256::hash(bincode::serialize(&new_gint_par(24, Vec::new(), false)).unwrap());

        let ed25519 = Ed25519;
        let signature = ed25519.sign(&par_bytes, &sec_key.bytes);

        let ref_verify = ed25519.verify(&par_bytes, &signature, &pub_key.bytes);
        assert!(ref_verify);

        let send = new_send_par(
            ed25519verify_channel,
            vec![
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(par_bytes)),
                }]),
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(signature)),
                }]),
                Par::default().with_exprs(vec![Expr {
                    expr_instance: Some(ExprInstance::GByteArray(pub_key.bytes.as_ref().to_vec())),
                }]),
                ack_channel(),
            ],
            false,
            Vec::new(),
            false,
            Vec::new(),
            false,
        );

        runtime.inj(send, empty_env(), rand()).await.unwrap();

        assert_store_contains(
            runtime,
            ack_channel(),
            ListParWithRandom {
                pars: vec![new_gbool_par(true, Vec::new(), false)],
                random_state: rand().to_bytes(),
            },
        );
    }
}

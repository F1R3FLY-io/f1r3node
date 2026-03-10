// use std::fs::remove_dir_all;

// use models::rhoapi::g_unforgeable::UnfInstance::GPrivateBody;
// use models::rhoapi::{
//     BindPattern, GPrivate, GUnforgeable, ListParWithRandom, Par,
// TaggedContinuation, };
// use rholang::rust::interpreter::matcher::r#match::Matcher;
// use rspace_plus_plus::rspace::rspace::{RSpace, RSpaceInstances};
// use rspace_plus_plus::rspace::shared::key_value_store_manager::KeyValueStoreManager;
// use rspace_plus_plus::rspace::shared::lmdb_dir_store_manager::GB;
// use rspace_plus_plus::rspace::shared::rspace_store_manager::mk_rspace_store_manager;

// #[tokio::test]
// async fn history_repository_should_process_insert_one_datum() {
//     let mut rspace = create_rspace().await;
//     let continuation = TaggedContinuation::default();
//     let channels = vec![
//         Par::default().with_unforgeables(vec![GUnforgeable {
//             unf_instance: Some(GPrivateBody(GPrivate { id: vec![1] })),
//         }]),
//         Par::default().with_unforgeables(vec![GUnforgeable {
//             unf_instance: Some(GPrivateBody(GPrivate { id: vec![3] })),
//         }]),
//         Par::default().with_unforgeables(vec![GUnforgeable {
//             unf_instance: Some(GPrivateBody(GPrivate { id: vec![5] })),
//         }]),
//         Par::default().with_unforgeables(vec![GUnforgeable {
//             unf_instance: Some(GPrivateBody(GPrivate { id: vec![7] })),
//         }]),
//     ];

//     let patterns = vec![
//         BindPattern::default(),
//         BindPattern::default(),
//         BindPattern::default(),
//         BindPattern::default(),
//     ];

//     // let install_opt = rspace
//     //     .install(channels.clone(), patterns.clone(), continuation.clone())
//     //     .await;
//     // assert!(install_opt.is_none());

//     let produce = rspace.produce(channels[0].clone(),
// ListParWithRandom::default(), false);     assert!(produce.is_none());

//     let install_opt2 = rspace.install(channels, patterns, continuation);
//     assert!(install_opt2.is_none());

//     teardown();
// }

// async fn create_rspace() -> RSpace<Par, BindPattern, ListParWithRandom,
// TaggedContinuation, Matcher> {
//     let mut kvm = mk_rspace_store_manager("./install_test/".into(), 1 * GB);
//     let store = kvm.r_space_stores().await.unwrap();

//     RSpaceInstances::create(store, Matcher).unwrap()
// }

// fn teardown() -> () {
//     remove_dir_all("./install_test/").unwrap();
// }

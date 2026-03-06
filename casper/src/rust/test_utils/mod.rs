// Test utilities module for sharing test helpers with other workspace members
// Test utilities have been moved from casper/tests/ to casper/src/rust/test_utils/
// and all imports have been fixed for library crate context.
//
// To use these utilities in other workspace members:
// 1. Add `casper = { path = "../casper", features = ["test-utils"] }` to your Cargo.toml
// 2. Import: `use casper::rust::test_utils::helper::test_node::TestNode;`

#[cfg(feature = "test-utils")]
pub mod helper;

#[cfg(feature = "test-utils")]
pub mod util;


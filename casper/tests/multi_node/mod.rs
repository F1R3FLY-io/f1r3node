// Multi-node tests reproducing integration-suite failure modes at unit level.
//
// These tests use `TestNode::create_network` to simulate the multi-validator
// topology of the integration test suite. They run deterministically and
// without docker rebuild cycles, so we can exercise concurrency-sensitive
// merge behavior at the unit-test layer.

mod bridge_contract_concurrent_merge;

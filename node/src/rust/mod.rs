// Node module - Main blockchain node implementation
// Empty module - ready for future implementation

pub mod api;
pub mod configuration;
pub mod diagnostics;
pub mod effects;
pub mod encode;
pub mod instances;
pub mod node_environment;
pub mod repl;
pub mod rho_trie_traverser;
pub mod runtime;
pub mod state;
pub mod web;

// Re-export for convenience
pub use encode::JsonEncoder;

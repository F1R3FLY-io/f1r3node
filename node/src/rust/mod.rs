// Node module - Main blockchain node implementation
// Empty module - ready for future implementation

pub mod api;
pub mod configuration;
pub mod effects;
pub mod encode;
pub mod repl;
pub mod rho_trie_traverser;
pub mod web;

// Re-export for convenience
pub use encode::JsonEncoder;

// RGB integration module for F1r3fly
// This module contains RGB-specific data structures and processing logic

// OLD FILES (to be removed)
pub mod types;
pub mod processor;

// NEW CLEAN FILES 
pub mod rgb_types;
pub mod rgb20_processor;
pub mod f1r3fly_rgb_bridge;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rgb20_tests;

// Export old interfaces for backward compatibility
pub use types::*;
pub use processor::*;

// Export clean interfaces (avoiding conflicts)
pub use rgb_types::{Rgb20IssuanceRequest, Rgb20IssuanceResponse, RgbError};
pub use rgb20_processor::Rgb20Processor;
pub use f1r3fly_rgb_bridge::F1r3flyRgbBridge;

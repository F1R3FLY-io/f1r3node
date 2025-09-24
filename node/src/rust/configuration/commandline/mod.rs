//! Command-line argument parsing module
//!
//! This module provides command-line argument parsing functionality using the `clap` crate,
//! replacing the Scala Scallop library.

pub mod config_mapper;
pub mod converters;
pub mod options;

pub use config_mapper::ConfigMapper;
pub use converters::*;
pub use options::Options;

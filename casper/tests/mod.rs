use std::sync::Once;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod add_block;
mod api;
mod batch1;
mod batch2;
mod blocks;
mod engine;
mod genesis;
mod helper;
mod merging;
mod sync;
mod util;

static INIT: Once = Once::new();

pub fn init_logger() {
    INIT.call_once(|| {
        let filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::DEBUG.into())
            .parse("")
            .unwrap();

        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_target(false)
                    .with_current_span(false) // logs only
                    .with_span_list(false) // logs only
                    .flatten_event(true), // put event fields at top level
            )
            .try_init()
            .ok();

        // Initialize tracing subscriber with Info level to minimize logs in tests
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_test_writer() // Capture output properly in tests
            .with_target(true) // Show module targets
            .with_file(true) // Show file names
            .with_line_number(true) // Show line numbers
            .try_init()
            .ok();
    });
}

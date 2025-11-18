use std::sync::Once;

mod add_block;
mod api;
mod batch1;
mod blocks;
mod engine;
mod helper;
mod merging;
mod batch2;
mod sync;
mod util;

static INIT: Once = Once::new();

pub fn init_logger() {
    INIT.call_once(|| {
        // Initialize env_logger for the log crate
        env_logger::builder()
            .is_test(true) // ensures logs show up in test output
            .filter_level(log::LevelFilter::Info)
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

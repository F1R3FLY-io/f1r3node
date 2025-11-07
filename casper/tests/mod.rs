use std::sync::Once;

mod add_block;
mod api;
mod batch1;
mod blocks;
mod engine;
mod estimator;
mod finality;
mod helper;
mod merging;
mod batch2;
mod sync;
mod util;
mod validate;

static INIT: Once = Once::new();

pub fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder()
            .is_test(true) // ensures logs show up in test output
            .filter_level(log::LevelFilter::Info)
            .try_init()
            .unwrap();
    });
}

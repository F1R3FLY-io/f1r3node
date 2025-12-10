use std::time::Duration;

pub mod test_util;

// Tests
#[cfg(test)]
mod auth_key_spec;

// See casper/src/test/scala/coop/rchain/casper/genesis/contracts/package.scala
pub const GENESIS_TEST_TIMEOUT: Duration = Duration::from_secs(60);

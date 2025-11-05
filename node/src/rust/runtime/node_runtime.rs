// See node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala

use casper::rust::errors::CasperError;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// Type aliases for repeatable async operations
pub type CasperLoop =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync>;
pub type EngineInit =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), CasperError>> + Send>> + Send + Sync>;

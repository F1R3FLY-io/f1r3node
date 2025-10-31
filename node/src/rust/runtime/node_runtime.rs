// See node/src/main/scala/coop/rchain/node/runtime/NodeRuntime.scala

use casper::rust::errors::CasperError;

pub type CasperLoop = Result<(), CasperError>;
pub type EngineInit = Result<(), CasperError>;

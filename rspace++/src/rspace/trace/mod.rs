use event::Event;

pub mod event;

// See rspace/src/main/scala/coop/rchain/rspace/trace/package.scala
pub type Log = Vec<Event>;

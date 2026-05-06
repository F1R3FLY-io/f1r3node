// See casper/src/main/scala/coop/rchain/casper/rholang/types/EvalCollector.scala

use std::collections::HashMap;

use models::{rhoapi::Par, rust::casper::protocol::casper_message::Event};
use rspace_plus_plus::rspace::merger::merging_logic::MergeType;

pub struct EvalCollector {
    pub event_log: Vec<Event>,
    pub mergeable_channels: HashMap<Par, MergeType>,
}

impl EvalCollector {
    pub fn new() -> Self {
        Self {
            event_log: Vec::new(),
            mergeable_channels: HashMap::new(),
        }
    }

    pub fn add_event_log(&mut self, event_log: Vec<Event>) {
        self.event_log.extend(event_log);
    }

    pub fn add_mergeable_channels(&mut self, mergeable_channels: HashMap<Par, MergeType>) {
        self.mergeable_channels.extend(mergeable_channels);
    }

    pub fn add(&mut self, event_log: Vec<Event>, mergeable_channels: HashMap<Par, MergeType>) {
        self.event_log.extend(event_log);
        self.mergeable_channels.extend(mergeable_channels);
    }
}

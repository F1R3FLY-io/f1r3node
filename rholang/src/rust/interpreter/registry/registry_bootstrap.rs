// See rholang/src/main/scala/coop/rchain/rholang/interpreter/registry/RegistryBootstrap.scala

use crate::rust::interpreter::system_processes::FixedChannels;
use models::{
    rhoapi::{New, Par, Receive, ReceiveBind, Send},
    rust::utils::{new_boundvar_par, new_freevar_par},
};
use std::collections::BTreeMap;

pub fn ast() -> Par {
    Par::default().with_news(vec![
        bootstrap(FixedChannels::reg_lookup()),
        bootstrap(FixedChannels::reg_insert_random()),
        bootstrap(FixedChannels::reg_insert_signed()),
    ])
}

/**
 * This is used to get a one-time hold of write-only-bundled fixed
 * channel, e.g `FixedChannels.REG_LOOKUP`, from within Rholang code.
 * It can be used to produce a contract, e.g. the registry lookup
 * contract, on a write-only-bundled fixed channel.
 */
fn bootstrap(channel: Par) -> New {
    New {
        bind_count: 1,
        p: Some(Par {
            sends: Vec::new(),
            // for (x <- channel) { x!(channel) }
            receives: vec![Receive {
                // for (x <- channel)
                binds: vec![ReceiveBind {
                    patterns: vec![new_freevar_par(0, Vec::new())],
                    source: Some(channel.clone()),
                    remainder: None,
                    free_count: 1,
                    pattern_modifiers: vec![],
                }],
                // x!(channel)
                body: Some(Par::default().with_sends(vec![Send {
                    chan: Some(new_boundvar_par(0, Vec::new(), false)),
                    data: vec![channel],
                    persistent: false,
                    locally_free: Vec::new(),
                    connective_used: false,
                    hyperparams: vec![], // No hyperparams for bootstrap sends
                }])),
                persistent: false,
                peek: false,
                bind_count: 1,
                locally_free: Vec::new(),
                connective_used: false,
            }],
            news: Vec::new(),
            exprs: Vec::new(),
            matches: Vec::new(),
            unforgeables: Vec::new(),
            bundles: Vec::new(),
            connectives: Vec::new(),
            locally_free: Vec::new(),
            connective_used: false,
            use_blocks: Vec::new(), // Reifying RSpaces
        }),
        uri: Vec::new(),
        injections: BTreeMap::default(),
        locally_free: Vec::new(),
        space_types: Vec::new(),
    }
}

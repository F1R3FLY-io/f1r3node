// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/StoragePrinter.scala

use models::rhoapi::tagged_continuation::TaggedCont;
use models::rhoapi::{
    BindPattern, ListParWithRandom, Par, Receive, ReceiveBind, Send, TaggedContinuation,
};
use models::rust::rholang::implicits::concatenate_pars;
use rspace_plus_plus::rspace::internal::{Datum, Row, WaitingContinuation};
use std::collections::BTreeSet;

use crate::rust::interpreter::pretty_printer::PrettyPrinter;
use crate::rust::interpreter::rho_runtime::{RhoRuntime, RhoRuntimeImpl};

pub fn pretty_print(runtime: &RhoRuntimeImpl) -> String {
    let mapped = runtime.get_hot_changes();

    let pars: Vec<Par> = mapped
        .iter()
        .map(|(channels, row)| match row {
            Row { data, wks } if data.is_empty() && wks.is_empty() => Par::default(),
            Row { data, wks } if !data.is_empty() && wks.is_empty() => to_sends(data, channels),
            Row { data, wks } if data.is_empty() && !wks.is_empty() => to_receives(wks, channels),
            Row { data, wks } => {
                let sends = to_sends(data, channels);
                let receives = to_receives(wks, channels);
                concatenate_pars(sends, receives)
            }
        })
        .collect();

    if pars.is_empty() {
        "The space is empty. Note that top level terms that are not sends or receives are discarded.".to_string()
    } else {
        let combined_par = pars
            .into_iter()
            .fold(Par::default(), |acc, par| concatenate_pars(acc, par));

        let mut pretty_printer = PrettyPrinter::new();
        pretty_printer.build_string_from_message(&combined_par)
    }
}

pub fn pretty_print_unmatched_sends(runtime: &RhoRuntimeImpl) -> String {
    let mapped = runtime.get_hot_changes();

    let pars: Vec<Par> = mapped
        .iter()
        .filter_map(|(channels, row)| {
            if !row.data.is_empty() {
                Some(to_sends(&row.data, channels))
            } else {
                None
            }
        })
        .collect();

    if pars.is_empty() {
        "The space is empty. Note that top level terms that are not sends or receives are discarded.".to_string()
    } else {
        let combined_par = pars
            .into_iter()
            .fold(Par::default(), |acc, par| concatenate_pars(acc, par));

        let mut pretty_printer = PrettyPrinter::new();
        pretty_printer.build_string_from_message(&combined_par)
    }
}

fn to_sends(data: &Vec<Datum<ListParWithRandom>>, channels: &Vec<Par>) -> Par {
    let mut sends: Vec<Send> = Vec::new();

    for datum in data {
        for channel in channels {
            sends.push(Send {
                chan: Some(channel.clone()),
                data: datum.a.pars.clone(),
                persistent: datum.persist,
                locally_free: Vec::new(),
                connective_used: false,
            });
        }
    }

    sends
        .into_iter()
        .fold(Par::default(), |mut acc, send| acc.prepend_send(send))
}

fn to_receives(
    wks: &Vec<WaitingContinuation<BindPattern, TaggedContinuation>>,
    channels: &Vec<Par>,
) -> Par {
    let mut receives: Vec<Receive> = Vec::new();

    for wk in wks {
        let patterns = &wk.patterns;
        let continuation = &wk.continuation;
        let persist = wk.persist;
        let peeks: &BTreeSet<i32> = &wk.peeks;

        let mut receive_binds: Vec<ReceiveBind> = Vec::new();
        for (i, pattern) in patterns.iter().enumerate() {
            if i < channels.len() {
                let channel = &channels[i];
                receive_binds.push(ReceiveBind {
                    patterns: pattern.patterns.clone(),
                    source: Some(channel.clone()),
                    remainder: pattern.remainder.clone(),
                    free_count: pattern.free_count,
                });
            }
        }

        match &continuation.tagged_cont {
            Some(TaggedCont::ParBody(p)) => {
                let free_count_sum = patterns.iter().map(|p| p.free_count).sum();

                receives.push(Receive {
                    binds: receive_binds,
                    body: p.body.clone(),
                    persistent: persist,
                    peek: !peeks.is_empty(),
                    bind_count: free_count_sum,
                    locally_free: Vec::new(),
                    connective_used: false,
                });
            }
            _ => {
                receives.push(Receive {
                    binds: receive_binds,
                    body: Some(Par::default()),
                    persistent: persist,
                    peek: false,
                    bind_count: 0,
                    locally_free: Vec::new(),
                    connective_used: false,
                });
            }
        }
    }

    receives
        .into_iter()
        .fold(Par::default(), |mut acc, receive| {
            acc.prepend_receive(receive)
        })
}

use models::{
    rhoapi::{Bundle, Connective, Expr, Match, New, Par, Receive, Send},
    rust::utils::union,
};

use super::matcher::has_locally_free::HasLocallyFree;

pub mod address_tools;
pub mod base58;
pub mod rev_address;
pub mod sbert_embeddings;

// Helper enum. This is 'GeneratedMessage' in Scala
#[derive(Clone, Debug)]
pub enum GeneratedMessage {
    Send(Send),
    Receive(Receive),
    New(New),
    Match(Match),
    Bundle(Bundle),
    Expr(Expr),
}

// These two functions need to be under 'rholang' dir because of HasLocallyFree Trait.
// This trait should, I think, be moved to models

// See models/src/main/scala/coop/rchain/models/rholang/implicits.scala - prepend
pub fn prepend_connective(mut p: Par, c: Connective, depth: i32) -> Par {
    let mut new_connectives = vec![c.clone()];
    new_connectives.append(&mut p.connectives);

    Par {
        connectives: new_connectives,
        locally_free: c.locally_free(c.clone(), depth),
        connective_used: p.connective_used || c.clone().connective_used(c),
        ..p.clone()
    }
}

pub fn prepend_expr(mut p: Par, e: Expr, depth: i32) -> Par {
    let mut new_exprs = vec![e.clone()];
    new_exprs.append(&mut p.exprs);

    Par {
        exprs: new_exprs,
        locally_free: union(p.locally_free.clone(), e.locally_free(e.clone(), depth)),
        connective_used: p.connective_used || e.clone().connective_used(e),
        ..p.clone()
    }
}

pub fn prepend_new(mut p: Par, n: New) -> Par {
    let mut new_news = vec![n.clone()];
    new_news.append(&mut p.news);

    Par {
        news: new_news,
        locally_free: union(p.locally_free.clone(), n.clone().locally_free),
        connective_used: p.connective_used || n.clone().connective_used(n),
        ..p.clone()
    }
}

pub fn prepend_bundle(mut p: Par, b: Bundle) -> Par {
    let mut new_bundles = vec![b.clone()];
    new_bundles.append(&mut p.bundles);

    Par {
        bundles: new_bundles,
        locally_free: union(p.locally_free.clone(), b.body.unwrap().locally_free),
        ..p.clone()
    }
}

// for locally_free parameter, in case when we have (bodyResult.par.locallyFree.from(boundCount).map(x => x - boundCount))
pub(crate) fn filter_and_adjust_bitset(bitset: Vec<u8>, bound_count: usize) -> Vec<u8> {
    bitset
        .into_iter()
        .enumerate()
        .filter_map(|(i, _)| {
            if i >= bound_count {
                Some(i as u8 - bound_count as u8)
            } else {
                None
            }
        })
        .collect()
}

use expr::ExprInstance;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::par_map::ParMap;
use super::par_map_type_mapper::ParMapTypeMapper;
use super::par_set::ParSet;
use super::par_set_type_mapper::ParSetTypeMapper;
use super::rholang::implicits::vector_par;
use crate::rhoapi::g_unforgeable::UnfInstance;
use crate::rust::utils::connective::ConnectiveInstance::*;
use crate::rust::utils::expr::ExprInstance::EVarBody;
use crate::rust::utils::expr::ExprInstance::*;
use crate::rust::utils::var::VarInstance::{BoundVar, FreeVar, Wildcard};
use crate::rust::utils::var::WildcardMsg;
use crate::{create_bit_vector, rhoapi::*};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct OptionResult<A, K> {
    pub continuation: K,
    pub data: A,
}

// Adding helper functions 'with_*' to protobuf message 'Par'
impl Par {
    pub fn with_sends(&self, new_sends: Vec<Send>) -> Par {
        Par {
            sends: new_sends,
            ..self.clone()
        }
    }

    pub fn with_receives(&self, new_receives: Vec<Receive>) -> Par {
        Par {
            receives: new_receives,
            ..self.clone()
        }
    }

    pub fn with_news(&self, new_news: Vec<New>) -> Par {
        Par {
            news: new_news,
            ..self.clone()
        }
    }

    pub fn with_exprs(&self, new_exprs: Vec<Expr>) -> Par {
        Par {
            exprs: new_exprs,
            ..self.clone()
        }
    }

    pub fn with_matches(&self, new_matches: Vec<Match>) -> Par {
        Par {
            matches: new_matches,
            ..self.clone()
        }
    }

    pub fn with_bundles(&self, new_bundles: Vec<Bundle>) -> Par {
        Par {
            bundles: new_bundles,
            ..self.clone()
        }
    }

    pub fn with_unforgeables(&self, new_unforgeables: Vec<GUnforgeable>) -> Par {
        Par {
            unforgeables: new_unforgeables,
            ..self.clone()
        }
    }

    pub fn with_connectives(&self, new_connectives: Vec<Connective>) -> Par {
        Par {
            connectives: new_connectives,
            ..self.clone()
        }
    }

    pub fn with_locally_free(&self, new_locally_free: Vec<u8>) -> Par {
        Par {
            locally_free: new_locally_free,
            ..self.clone()
        }
    }

    pub fn with_connective_used(&self, new_connective_used: bool) -> Par {
        Par {
            connective_used: new_connective_used,
            ..self.clone()
        }
    }

    // See models/src/main/scala/coop/rchain/models/rholang/implicits.scala - prepend
    pub fn prepend_send(&mut self, s: Send) -> Par {
        let mut new_sends = vec![s.clone()];
        new_sends.append(&mut self.sends);

        Par {
            sends: new_sends,
            locally_free: union(self.locally_free.clone(), s.locally_free),
            connective_used: self.connective_used || s.connective_used,
            ..self.clone()
        }
    }

    pub fn prepend_receive(&mut self, r: Receive) -> Par {
        let mut new_receives = vec![r.clone()];
        new_receives.append(&mut self.receives);

        Par {
            receives: new_receives,
            locally_free: union(self.locally_free.clone(), r.locally_free),
            connective_used: self.connective_used || r.connective_used,
            ..self.clone()
        }
    }

    pub fn prepend_match(&mut self, m: Match) -> Par {
        let mut new_matches = vec![m.clone()];
        new_matches.append(&mut self.matches);

        Par {
            matches: new_matches,
            locally_free: union(self.locally_free.clone(), m.locally_free),
            connective_used: self.connective_used || m.connective_used,
            ..self.clone()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.sends.is_empty()
            && self.receives.is_empty()
            && self.news.is_empty()
            && self.matches.is_empty()
            && self.bundles.is_empty()
            && self.exprs.is_empty()
    }

    pub fn is_nil(&self) -> bool {
        self.sends.is_empty()
            && self.receives.is_empty()
            && self.news.is_empty()
            && self.matches.is_empty()
            && self.bundles.is_empty()
            && self.unforgeables.is_empty()
            && self.connectives.is_empty()
            && self.exprs.is_empty()
    }

    pub fn single_connective(&self) -> Option<Connective> {
        if self.sends.is_empty()
            && self.receives.is_empty()
            && self.news.is_empty()
            && self.exprs.is_empty()
            && self.matches.is_empty()
            && self.bundles.is_empty()
            && self.connectives.len() == 1
        {
            Some(self.connectives[0].clone())
        } else {
            None
        }
    }

    pub fn single_bundle(&self) -> Option<Bundle> {
        if self.sends.is_empty()
            && self.receives.is_empty()
            && self.news.is_empty()
            && self.exprs.is_empty()
            && self.matches.is_empty()
            && self.unforgeables.is_empty()
            && self.connectives.is_empty()
        {
            match self.bundles.as_slice() {
                [single] => Some(single.clone()),
                _ => None,
            }
        } else {
            None
        }
    }

    pub fn append(&self, other: Par) -> Par {
        Par {
            sends: [self.sends.clone(), other.sends].concat(),
            receives: [self.receives.clone(), other.receives].concat(),
            news: [self.news.clone(), other.news].concat(),
            exprs: [self.exprs.clone(), other.exprs].concat(),
            matches: [self.matches.clone(), other.matches].concat(),
            unforgeables: [self.unforgeables.clone(), other.unforgeables].concat(),
            bundles: [self.bundles.clone(), other.bundles].concat(),
            connectives: [self.connectives.clone(), other.connectives].concat(),
            locally_free: union(self.locally_free.clone(), other.locally_free),
            connective_used: self.connective_used || other.connective_used,
            use_blocks: [self.use_blocks.clone(), other.use_blocks].concat(),  // Reifying RSpaces
        }
    }
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/package.scala - FreeMap
pub type FreeMap = BTreeMap<i32, Par>;
pub fn new_free_map() -> FreeMap {
    BTreeMap::new()
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/package.scala - runFirst
// STUBBED OUT
pub fn run_first<A>() -> Option<(FreeMap, A)> {
    None
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/package.scala - attemptOpt
// NOT FULLY IMPLEMENTED
pub fn attempt_opt(operation: Option<()>) -> Option<()> {
    operation.map(|_| ())
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/storage/package.scala - toSeq
pub fn to_vec(fm: FreeMap, max: i32) -> Vec<Par> {
    (0..max)
        .map(|i| match fm.get(&i) {
            Some(par) => par.clone(),
            None => Par::default(),
        })
        .collect()
}

pub fn union(bitset1: Vec<u8>, bitset2: Vec<u8>) -> Vec<u8> {
    let max_len = bitset1.len().max(bitset2.len());
    let mut result = vec![0; max_len];

    for i in 0..max_len {
        let bit1 = if i < bitset1.len() { bitset1[i] } else { 0 };
        let bit2 = if i < bitset2.len() { bitset2[i] } else { 0 };
        result[i] = bit1 | bit2;
    }

    result
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/ParSpatialMatcherUtils.scala - noFrees[Par]
pub fn no_frees(par: Par) -> Par {
    par.with_exprs(no_frees_exprs(par.exprs.clone()))
}

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/ParSpatialMatcherUtils.scala - noFrees[Seq[Expr]]
pub fn no_frees_exprs(exprs: Vec<Expr>) -> Vec<Expr> {
    exprs
        .iter()
        .filter(|expr| match expr.expr_instance.clone() {
            Some(EVarBody(EVar { v })) => match v.unwrap().var_instance {
                Some(FreeVar(_)) => false,
                Some(Wildcard(_)) => false,
                _ => true,
            },

            _ => true,
        })
        .cloned()
        .collect()
}

// See shared/src/main/scala/coop/rchain/catscontrib/Alternative_.scala - guard
pub fn guard(condition: bool) -> Option<()> {
    if condition {
        Some(())
    } else {
        None
    }
}

// Helper functions
pub fn new_conn_and_body_par(
    _ps: Vec<Par>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_connectives(vec![Connective {
        connective_instance: Some(ConnAndBody(ConnectiveBody { ps: _ps })),
    }])
}

pub fn new_conn_or_body_par(
    _ps: Vec<Par>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_connectives(vec![Connective {
        connective_instance: Some(ConnOrBody(ConnectiveBody { ps: _ps })),
    }])
}

pub fn new_conn_not_body_par(
    _body: Par,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_connectives(vec![Connective {
        connective_instance: Some(ConnNotBody(_body)),
    }])
}

pub fn new_send(
    _chan: Par,
    _data: Vec<Par>,
    _persistent: bool,
    _locally_free: Vec<u8>,
    _connective_used: bool,
) -> Send {
    Send {
        chan: Some(_chan),
        data: _data,
        persistent: _persistent,
        locally_free: _locally_free,
        connective_used: _connective_used,
        hyperparams: vec![],  // Hyperparam support
    }
}

pub fn new_send_par(
    _chan: Par,
    _data: Vec<Par>,
    _persistent: bool,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_sends(vec![Send {
        chan: Some(_chan),
        data: _data,
        persistent: _persistent,
        locally_free: _locally_free,
        connective_used: _connective_used,
        hyperparams: vec![],  // Hyperparam support
    }])
}

pub fn new_match_par(
    _target: Par,
    _cases: Vec<MatchCase>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_matches(vec![Match {
        target: Some(_target),
        cases: _cases,
        locally_free: _locally_free,
        connective_used: _connective_used,
    }])
}

pub fn new_receive_par(
    _binds: Vec<ReceiveBind>,
    _body: Par,
    _persistent: bool,
    _peek: bool,
    _bind_count: i32,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_receives(vec![Receive {
        binds: _binds,
        body: Some(_body),
        persistent: _persistent,
        peek: _peek,
        bind_count: _bind_count,
        locally_free: _locally_free,
        connective_used: _connective_used,
    }])
}

pub fn new_new_par(
    _bind_count: i32,
    _p: Par,
    _uri: Vec<String>,
    _injections: BTreeMap<String, Par>,
    _locally_free: Vec<u8>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_news(vec![New {
        bind_count: _bind_count,
        p: Some(_p),
        uri: _uri,
        injections: _injections,
        locally_free: _locally_free,
        space_types: vec![],
    }])
}

pub fn new_eset_par(
    _ps: Vec<Par>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _remainder: Option<Var>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_eset_expr(
        _ps,
        _locally_free,
        _connective_used,
        _remainder,
    )])
}

pub fn new_eset_expr(
    _ps: Vec<Par>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _remainder: Option<Var>,
) -> Expr {
    // println!("new_eset_expr: _ps: {:?}", _ps);
    // println!("new_eset_expr: _locally_free: {:?}", _locally_free);
    // println!("new_eset_expr: _connective_used: {:?}", _connective_used);
    // println!("new_eset_expr: _remainder: {:?}", _remainder);
    Expr {
        expr_instance: Some(ESetBody(ParSetTypeMapper::par_set_to_eset(ParSet::new(
            _ps,
            _connective_used,
            _locally_free,
            _remainder,
        )))),
    }
}

pub fn new_emap_par(
    _kvs: Vec<KeyValuePair>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _remainder: Option<Var>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_emap_expr(
        _kvs,
        _locally_free,
        _connective_used,
        _remainder,
    )])
}

pub fn new_emap_expr(
    _kvs: Vec<KeyValuePair>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _remainder: Option<Var>,
) -> Expr {
    Expr {
        expr_instance: Some(EMapBody(ParMapTypeMapper::par_map_to_emap(ParMap::new(
            _kvs.into_iter()
                .filter_map(|kv| {
                    if let (Some(key), Some(value)) = (kv.key, kv.value) {
                        Some((key, value))
                    } else {
                        None
                    }
                })
                .collect(),
            _connective_used,
            _locally_free,
            _remainder,
        )))),
    }
}

pub fn new_key_value_pair(_key: Par, _value: Par) -> KeyValuePair {
    KeyValuePair {
        key: Some(_key),
        value: Some(_value),
    }
}

pub fn new_gint_par(value: i64, _locally_free_par: Vec<u8>, _connective_used_par: bool) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_gint_expr(value)])
}

pub fn new_gint_expr(value: i64) -> Expr {
    Expr {
        expr_instance: Some(GInt(value)),
    }
}

pub fn new_gbool_par(value: bool, _locally_free_par: Vec<u8>, _connective_used_par: bool) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_gbool_expr(value)])
}

pub fn new_gbool_expr(value: bool) -> Expr {
    Expr {
        expr_instance: Some(GBool(value)),
    }
}

pub fn new_gstring_par(
    value: String,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_gstring_expr(value)])
}

pub fn new_gstring_expr(value: String) -> Expr {
    Expr {
        expr_instance: Some(GString(value)),
    }
}

pub fn new_guri_par(value: String, _locally_free_par: Vec<u8>, _connective_used_par: bool) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_guri_expr(value)])
}

pub fn new_guri_expr(value: String) -> Expr {
    Expr {
        expr_instance: Some(GUri(value)),
    }
}

pub fn new_wildcard_par(_locally_free_par: Vec<u8>, _connective_used_par: bool) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![Expr {
        expr_instance: Some(EVarBody(EVar {
            v: Some(Var {
                var_instance: Some(Wildcard(WildcardMsg {})),
            }),
        })),
    }])
}

pub fn new_wildcard_expr() -> Expr {
    Expr {
        expr_instance: Some(EVarBody(EVar {
            v: Some(Var {
                var_instance: Some(Wildcard(WildcardMsg {})),
            }),
        })),
    }
}

pub fn new_wildcard_var() -> Var {
    Var {
        var_instance: Some(Wildcard(WildcardMsg {})),
    }
}

pub fn new_boundvar_par(value: i32, _locally_free_par: Vec<u8>, _connective_used_par: bool) -> Par {
    vector_par(
        create_bit_vector(&vec![value as usize]),
        _connective_used_par,
    )
    .with_exprs(vec![new_boundvar_expr(value)])
}

pub fn new_boundvar_expr(value: i32) -> Expr {
    Expr {
        expr_instance: Some(EVarBody(EVar {
            v: Some(Var {
                var_instance: Some(BoundVar(value)),
            }),
        })),
    }
}

// "connective_used" is always "true" on "freevar"
pub fn new_freevar_par(value: i32, _locally_free_par: Vec<u8>) -> Par {
    vector_par(_locally_free_par, true).with_exprs(vec![new_freevar_expr(value)])
}

pub fn new_freevar_expr(value: i32) -> Expr {
    Expr {
        expr_instance: Some(EVarBody(EVar {
            v: Some(Var {
                var_instance: Some(FreeVar(value)),
            }),
        })),
    }
}

pub fn new_freevar_var(value: i32) -> Var {
    Var {
        var_instance: Some(FreeVar(value)),
    }
}

pub fn new_elist_par(
    _ps: Vec<Par>,
    _locally_free: Vec<u8>,
    _connective_used_elist: bool,
    _remainder: Option<Var>,
    _locally_free_par: Vec<u8>,
    _connective_used_par: bool,
) -> Par {
    vector_par(_locally_free_par, _connective_used_par).with_exprs(vec![new_elist_expr(
        _ps,
        _locally_free,
        _connective_used_elist,
        _remainder,
    )])
}

pub fn new_elist_expr(
    _ps: Vec<Par>,
    _locally_free: Vec<u8>,
    _connective_used: bool,
    _remainder: Option<Var>,
) -> Expr {
    Expr {
        expr_instance: Some(EListBody(EList {
            ps: _ps,
            locally_free: _locally_free,
            connective_used: _connective_used,
            remainder: _remainder,
        })),
    }
}

pub fn new_etuple_par(_ps: Vec<Par>) -> Par {
    vector_par(Vec::new(), false).with_exprs(vec![new_etuple_expr(_ps, Vec::new(), false)])
}

pub fn new_etuple_expr(_ps: Vec<Par>, _locally_free: Vec<u8>, _connective_used: bool) -> Expr {
    Expr {
        expr_instance: Some(ETupleBody(ETuple {
            ps: _ps,
            locally_free: _locally_free,
            connective_used: _connective_used,
        })),
    }
}

pub fn new_eplus_par_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Par {
    Par::default().with_exprs(vec![Expr {
        expr_instance: Some(ExprInstance::EPlusBody(EPlus {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }])
}

pub fn new_eplus_par(lhs_value: Par, rhs_value: Par) -> Par {
    let locally_free = union(
        lhs_value.locally_free.clone(),
        rhs_value.locally_free.clone(),
    );
    let connective_used = lhs_value.connective_used || rhs_value.connective_used;

    Par::default()
        .with_exprs(vec![Expr {
            expr_instance: Some(EPlusBody(EPlus {
                p1: Some(lhs_value),
                p2: Some(rhs_value),
            })),
        }])
        .with_locally_free(locally_free)
        .with_connective_used(connective_used)
}

pub fn new_bundle_par(body: Par, write_flag: bool, read_flag: bool) -> Par {
    Par::default().with_bundles(vec![Bundle {
        body: Some(body),
        write_flag,
        read_flag,
    }])
}

pub fn new_eminus_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EMinusBody(EMinus {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_ediv_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EDivBody(EDiv {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_eplus_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EPlusBody(EPlus {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_emult_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EMultBody(EMult {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_eeq_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EEqBody(EEq {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_eneq_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(ENeqBody(ENeq {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_elt_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(ELtBody(ELt {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_elte_expr_gint(
    lhs_value: i64,
    rhs_value: i64,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(ELteBody(ELte {
            p1: Some(new_gint_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gint_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_egt_expr_gbool(
    lhs_value: bool,
    rhs_value: bool,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EGtBody(EGt {
            p1: Some(new_gbool_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gbool_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_egte_expr_gbool(
    lhs_value: bool,
    rhs_value: bool,
    locally_free_par: Vec<u8>,
    connective_used_par: bool,
) -> Expr {
    Expr {
        expr_instance: Some(EGteBody(EGte {
            p1: Some(new_gbool_par(
                lhs_value,
                locally_free_par.clone(),
                connective_used_par,
            )),
            p2: Some(new_gbool_par(
                rhs_value,
                locally_free_par,
                connective_used_par,
            )),
        })),
    }
}

pub fn new_eor_expr(lhs: Par, rhs: Par) -> Expr {
    Expr {
        expr_instance: Some(EOrBody(EOr {
            p1: Some(lhs),
            p2: Some(rhs),
        })),
    }
}

pub fn new_emethod_expr(
    method_name: String,
    target: Par,
    arguments: Vec<Par>,
    locally_free: Vec<u8>,
) -> Expr {
    Expr {
        expr_instance: Some(EMethodBody(EMethod {
            method_name,
            target: Some(target),
            arguments,
            locally_free,
            connective_used: false,
        })),
    }
}

pub fn new_par_from_par_set(
    elements: Vec<Par>,
    locally_free: Vec<u8>,
    connective_used: bool,
    remainder: Option<Var>,
) -> Par {
    let par_set = ParSet::new(elements, connective_used, locally_free, remainder);

    Par {
        exprs: vec![Expr {
            expr_instance: Some(ESetBody(ParSetTypeMapper::par_set_to_eset(par_set))),
        }],
        ..Default::default()
    }
}

pub fn new_gbytearray_par(bytes: Vec<u8>, locally_free: Vec<u8>, connective_used: bool) -> Par {
    Par {
        exprs: vec![Expr {
            expr_instance: Some(GByteArray(bytes)),
        }],
        locally_free,
        connective_used,
        ..Default::default()
    }
}

pub fn new_gsys_auth_token_par(locally_free: Vec<u8>, connective_used: bool) -> Par {
    Par {
        unforgeables: vec![GUnforgeable {
            unf_instance: Some(UnfInstance::GSysAuthTokenBody(GSysAuthToken {})),
        }],
        locally_free,
        connective_used,
        ..Default::default()
    }
}

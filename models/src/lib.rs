pub mod rust;

pub mod casper {
    include!(concat!(env!("OUT_DIR"), "/casper.rs"));
    pub mod v1 {
        tonic::include_proto!("casper.v1");
    }
}

pub mod rhoapi {
    include!(concat!(env!("OUT_DIR"), "/rhoapi.rs"));
}

pub mod rholang_scala_rust_types {
    include!(concat!(env!("OUT_DIR"), "/rholang_scala_rust_types.rs"));
}

pub mod rspace_plus_plus_types {
    include!(concat!(env!("OUT_DIR"), "/rspace_plus_plus_types.rs"));
}

pub mod servicemodelapi {
    include!(concat!(env!("OUT_DIR"), "/servicemodelapi.rs"));
}

pub mod routing {
    include!(concat!(env!("OUT_DIR"), "/routing.rs"));
}

pub use prost::bytes::Bytes as ByteString;

use shared::rust::BitSet;

pub fn create_bit_vector(indices: &[usize]) -> BitSet {
    let max_index = *indices.iter().max().unwrap_or(&0);
    let mut bit_vector = vec![0; max_index + 1];
    for &index in indices {
        bit_vector[index] = 1;
    }
    // println!("\nbitvector: {:?}", bit_vector);
    bit_vector
}

use crate::connective::ConnectiveInstance;
use crate::expr::ExprInstance;
use crate::rhoapi::*;
use crate::servicemodelapi::ServiceError;
use crate::var::VarInstance;
use g_unforgeable::UnfInstance;
use tagged_continuation::TaggedCont;
use var::WildcardMsg;

use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};

// See models/src/main/scala/coop/rchain/models/AlwaysEqual.scala

impl PartialEq for Par {
    fn eq(&self, other: &Self) -> bool {
        self.sends == other.sends
            && self.receives == other.receives
            && self.news == other.news
            && self.exprs == other.exprs
            && self.matches == other.matches
            && self.unforgeables == other.unforgeables
            && self.bundles == other.bundles
            && self.connectives == other.connectives
            && self.connective_used == other.connective_used
    }
}

impl Hash for Par {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sends.hash(state);
        self.receives.hash(state);
        self.news.hash(state);
        self.exprs.hash(state);
        self.matches.hash(state);
        self.unforgeables.hash(state);
        self.bundles.hash(state);
        self.connectives.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for TaggedContinuation {
    fn eq(&self, other: &Self) -> bool {
        self.tagged_cont == other.tagged_cont
    }
}

impl Hash for TaggedContinuation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.tagged_cont.hash(state);
    }
}

impl PartialEq for TaggedCont {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TaggedCont::ParBody(par_body), TaggedCont::ParBody(other_par_body)) => {
                par_body == other_par_body
            }

            (
                TaggedCont::ScalaBodyRef(scala_body_ref),
                TaggedCont::ScalaBodyRef(other_scala_body_ref),
            ) => scala_body_ref == other_scala_body_ref,

            _ => false,
        }
    }
}

impl Hash for TaggedCont {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            TaggedCont::ParBody(par_body) => par_body.hash(state),
            TaggedCont::ScalaBodyRef(scala_body_ref) => scala_body_ref.hash(state),
        }
    }
}

impl PartialEq for ParWithRandom {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body && self.random_state == other.random_state
    }
}

impl Hash for ParWithRandom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.body.hash(state);
        self.random_state.hash(state);
    }
}

impl PartialEq for PCost {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Hash for PCost {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cost.hash(state);
    }
}

impl PartialEq for ListParWithRandom {
    fn eq(&self, other: &Self) -> bool {
        self.pars == other.pars && self.random_state == other.random_state
    }
}

impl Hash for ListParWithRandom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pars.hash(state);
        self.random_state.hash(state);
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.var_instance == other.var_instance
    }
}

impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.var_instance.hash(state);
    }
}

impl PartialEq for VarInstance {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (VarInstance::BoundVar(a), VarInstance::BoundVar(b)) => a == b,
            (VarInstance::FreeVar(a), VarInstance::FreeVar(b)) => a == b,
            (VarInstance::Wildcard(a), VarInstance::Wildcard(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for VarInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            VarInstance::BoundVar(a) => a.hash(state),
            VarInstance::FreeVar(a) => a.hash(state),
            VarInstance::Wildcard(a) => a.hash(state),
        }
    }
}

impl PartialEq for WildcardMsg {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Hash for WildcardMsg {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // No need to hash anything
    }
}

impl PartialEq for Bundle {
    fn eq(&self, other: &Self) -> bool {
        self.body == other.body
            && self.write_flag == other.write_flag
            && self.read_flag == other.read_flag
    }
}

impl Hash for Bundle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.body.hash(state);
        self.write_flag.hash(state);
        self.read_flag.hash(state);
    }
}

impl PartialEq for Send {
    fn eq(&self, other: &Self) -> bool {
        self.chan == other.chan
            && self.data == other.data
            && self.persistent == other.persistent
            && self.connective_used == other.connective_used
    }
}

impl Hash for Send {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.chan.hash(state);
        self.data.hash(state);
        self.persistent.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for ReceiveBind {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns
            && self.source == other.source
            && self.remainder == other.remainder
            && self.free_count == other.free_count
    }
}

impl Hash for ReceiveBind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.patterns.hash(state);
        self.source.hash(state);
        self.remainder.hash(state);
        self.free_count.hash(state);
    }
}

impl PartialEq for BindPattern {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns
            && self.remainder == other.remainder
            && self.free_count == other.free_count
    }
}

impl Hash for BindPattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.patterns.hash(state);
        self.remainder.hash(state);
        self.free_count.hash(state);
    }
}

impl PartialEq for ListBindPatterns {
    fn eq(&self, other: &Self) -> bool {
        self.patterns == other.patterns
    }
}

impl Hash for ListBindPatterns {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.patterns.hash(state);
    }
}

impl PartialEq for Receive {
    fn eq(&self, other: &Self) -> bool {
        self.binds == other.binds
            && self.body == other.body
            && self.persistent == other.persistent
            && self.peek == other.peek
            && self.bind_count == other.bind_count
            && self.connective_used == other.connective_used
    }
}

impl Hash for Receive {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.binds.hash(state);
        self.body.hash(state);
        self.persistent.hash(state);
        self.peek.hash(state);
        self.bind_count.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for New {
    fn eq(&self, other: &Self) -> bool {
        self.bind_count == other.bind_count
            && self.p == other.p
            && self.uri == other.uri
            && self.injections == other.injections
    }
}

impl Hash for New {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bind_count.hash(state);
        self.p.hash(state);
        self.uri.hash(state);
        self.injections.hash(state);
    }
}

impl PartialEq for MatchCase {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern
            && self.source == other.source
            && self.free_count == other.free_count
    }
}

impl Hash for MatchCase {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pattern.hash(state);
        self.source.hash(state);
        self.free_count.hash(state);
    }
}

impl PartialEq for Match {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target
            && self.cases == other.cases
            && self.connective_used == other.connective_used
    }
}

impl Hash for Match {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.target.hash(state);
        self.cases.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        self.expr_instance == other.expr_instance
    }
}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expr_instance.hash(state);
    }
}

impl PartialEq for expr::ExprInstance {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ExprInstance::GBool(a), ExprInstance::GBool(b)) => a == b,
            (ExprInstance::GInt(a), ExprInstance::GInt(b)) => a == b,
            (ExprInstance::GString(a), ExprInstance::GString(b)) => a == b,
            (ExprInstance::GUri(a), ExprInstance::GUri(b)) => a == b,
            (ExprInstance::GByteArray(a), ExprInstance::GByteArray(b)) => a == b,
            (ExprInstance::ENotBody(a), ExprInstance::ENotBody(b)) => a == b,
            (ExprInstance::ENegBody(a), ExprInstance::ENegBody(b)) => a == b,
            (ExprInstance::EMultBody(a), ExprInstance::EMultBody(b)) => a == b,
            (ExprInstance::EDivBody(a), ExprInstance::EDivBody(b)) => a == b,
            (ExprInstance::EPlusBody(a), ExprInstance::EPlusBody(b)) => a == b,
            (ExprInstance::EMinusBody(a), ExprInstance::EMinusBody(b)) => a == b,
            (ExprInstance::ELtBody(a), ExprInstance::ELtBody(b)) => a == b,
            (ExprInstance::ELteBody(a), ExprInstance::ELteBody(b)) => a == b,
            (ExprInstance::EGtBody(a), ExprInstance::EGtBody(b)) => a == b,
            (ExprInstance::EGteBody(a), ExprInstance::EGteBody(b)) => a == b,
            (ExprInstance::EEqBody(a), ExprInstance::EEqBody(b)) => a == b,
            (ExprInstance::ENeqBody(a), ExprInstance::ENeqBody(b)) => a == b,
            (ExprInstance::EAndBody(a), ExprInstance::EAndBody(b)) => a == b,
            (ExprInstance::EOrBody(a), ExprInstance::EOrBody(b)) => a == b,
            (ExprInstance::EVarBody(a), ExprInstance::EVarBody(b)) => a == b,
            (ExprInstance::EListBody(a), ExprInstance::EListBody(b)) => a == b,
            (ExprInstance::ETupleBody(a), ExprInstance::ETupleBody(b)) => a == b,
            (ExprInstance::ESetBody(a), ExprInstance::ESetBody(b)) => a == b,
            (ExprInstance::EMapBody(a), ExprInstance::EMapBody(b)) => a == b,
            (ExprInstance::EMethodBody(a), ExprInstance::EMethodBody(b)) => a == b,
            (ExprInstance::EMatchesBody(a), ExprInstance::EMatchesBody(b)) => a == b,
            (ExprInstance::EPercentPercentBody(a), ExprInstance::EPercentPercentBody(b)) => a == b,
            (ExprInstance::EPlusPlusBody(a), ExprInstance::EPlusPlusBody(b)) => a == b,
            (ExprInstance::EMinusMinusBody(a), ExprInstance::EMinusMinusBody(b)) => a == b,
            (ExprInstance::EModBody(a), ExprInstance::EModBody(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for expr::ExprInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ExprInstance::GBool(a) => a.hash(state),
            ExprInstance::GInt(a) => a.hash(state),
            ExprInstance::GString(a) => a.hash(state),
            ExprInstance::GUri(a) => a.hash(state),
            ExprInstance::GByteArray(a) => a.hash(state),
            ExprInstance::ENotBody(a) => a.hash(state),
            ExprInstance::ENegBody(a) => a.hash(state),
            ExprInstance::EMultBody(a) => a.hash(state),
            ExprInstance::EDivBody(a) => a.hash(state),
            ExprInstance::EPlusBody(a) => a.hash(state),
            ExprInstance::EMinusBody(a) => a.hash(state),
            ExprInstance::ELtBody(a) => a.hash(state),
            ExprInstance::ELteBody(a) => a.hash(state),
            ExprInstance::EGtBody(a) => a.hash(state),
            ExprInstance::EGteBody(a) => a.hash(state),
            ExprInstance::EEqBody(a) => a.hash(state),
            ExprInstance::ENeqBody(a) => a.hash(state),
            ExprInstance::EAndBody(a) => a.hash(state),
            ExprInstance::EOrBody(a) => a.hash(state),
            ExprInstance::EVarBody(a) => a.hash(state),
            ExprInstance::EListBody(a) => a.hash(state),
            ExprInstance::ETupleBody(a) => a.hash(state),
            ExprInstance::ESetBody(a) => a.hash(state),
            ExprInstance::EMapBody(a) => a.hash(state),
            ExprInstance::EMethodBody(a) => a.hash(state),
            ExprInstance::EMatchesBody(a) => a.hash(state),
            ExprInstance::EPercentPercentBody(a) => a.hash(state),
            ExprInstance::EPlusPlusBody(a) => a.hash(state),
            ExprInstance::EMinusMinusBody(a) => a.hash(state),
            ExprInstance::EModBody(a) => a.hash(state),
        }
    }
}

impl PartialEq for EList {
    fn eq(&self, other: &Self) -> bool {
        self.ps == other.ps
            && self.connective_used == other.connective_used
            && self.remainder == other.remainder
    }
}

impl Hash for EList {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ps.hash(state);
        self.connective_used.hash(state);
        self.remainder.hash(state);
    }
}

impl PartialEq for ETuple {
    fn eq(&self, other: &Self) -> bool {
        self.ps == other.ps && self.connective_used == other.connective_used
    }
}

impl Hash for ETuple {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ps.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for ESet {
    fn eq(&self, other: &Self) -> bool {
        self.ps == other.ps
            && self.connective_used == other.connective_used
            && self.remainder == other.remainder
    }
}

impl Hash for ESet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ps.hash(state);
        self.connective_used.hash(state);
        self.remainder.hash(state);
    }
}

impl PartialEq for EMap {
    fn eq(&self, other: &Self) -> bool {
        self.kvs == other.kvs
            && self.connective_used == other.connective_used
            && self.remainder == other.remainder
    }
}

impl Hash for EMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kvs.hash(state);
        self.connective_used.hash(state);
        self.remainder.hash(state);
    }
}

impl PartialEq for EMethod {
    fn eq(&self, other: &Self) -> bool {
        self.method_name == other.method_name
            && self.target == other.target
            && self.arguments == other.arguments
            && self.connective_used == other.connective_used
    }
}

impl Hash for EMethod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.method_name.hash(state);
        self.target.hash(state);
        self.arguments.hash(state);
        self.connective_used.hash(state);
    }
}

impl PartialEq for KeyValuePair {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.value == other.value
    }
}

impl Hash for KeyValuePair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key.hash(state);
        self.value.hash(state);
    }
}

impl PartialEq for EVar {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Hash for EVar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.v.hash(state);
    }
}

impl PartialEq for ENot {
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
    }
}

impl Hash for ENot {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p.hash(state);
    }
}

impl PartialEq for ENeg {
    fn eq(&self, other: &Self) -> bool {
        self.p == other.p
    }
}

impl Hash for ENeg {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p.hash(state);
    }
}

impl PartialEq for EMult {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EMult {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EDiv {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EDiv {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EMod {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EMod {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EPlus {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EPlus {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EMinus {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EMinus {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for ELt {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for ELt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for ELte {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for ELte {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EGt {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EGt {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EGte {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EGte {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EEq {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EEq {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for ENeq {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for ENeq {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EAnd {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EAnd {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EOr {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EOr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EMatches {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.pattern == other.pattern
    }
}

impl Hash for EMatches {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.target.hash(state);
        self.pattern.hash(state);
    }
}

impl PartialEq for EPercentPercent {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EPercentPercent {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EPlusPlus {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EPlusPlus {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for EMinusMinus {
    fn eq(&self, other: &Self) -> bool {
        self.p1 == other.p1 && self.p2 == other.p2
    }
}

impl Hash for EMinusMinus {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.p1.hash(state);
        self.p2.hash(state);
    }
}

impl PartialEq for Connective {
    fn eq(&self, other: &Self) -> bool {
        self.connective_instance == other.connective_instance
    }
}

impl Hash for Connective {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.connective_instance.hash(state);
    }
}

impl PartialEq for ConnectiveInstance {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ConnectiveInstance::ConnAndBody(a), ConnectiveInstance::ConnAndBody(b)) => a == b,
            (ConnectiveInstance::ConnOrBody(a), ConnectiveInstance::ConnOrBody(b)) => a == b,
            (ConnectiveInstance::ConnNotBody(a), ConnectiveInstance::ConnNotBody(b)) => a == b,
            (ConnectiveInstance::VarRefBody(a), ConnectiveInstance::VarRefBody(b)) => a == b,
            (ConnectiveInstance::ConnBool(a), ConnectiveInstance::ConnBool(b)) => a == b,
            (ConnectiveInstance::ConnInt(a), ConnectiveInstance::ConnInt(b)) => a == b,
            (ConnectiveInstance::ConnString(a), ConnectiveInstance::ConnString(b)) => a == b,
            (ConnectiveInstance::ConnUri(a), ConnectiveInstance::ConnUri(b)) => a == b,
            (ConnectiveInstance::ConnByteArray(a), ConnectiveInstance::ConnByteArray(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for ConnectiveInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ConnectiveInstance::ConnAndBody(a) => a.hash(state),
            ConnectiveInstance::ConnOrBody(a) => a.hash(state),
            ConnectiveInstance::ConnNotBody(a) => a.hash(state),
            ConnectiveInstance::VarRefBody(a) => a.hash(state),
            ConnectiveInstance::ConnBool(a) => a.hash(state),
            ConnectiveInstance::ConnInt(a) => a.hash(state),
            ConnectiveInstance::ConnString(a) => a.hash(state),
            ConnectiveInstance::ConnUri(a) => a.hash(state),
            ConnectiveInstance::ConnByteArray(a) => a.hash(state),
        }
    }
}

impl PartialEq for VarRef {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.depth == other.depth
    }
}

impl Hash for VarRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.depth.hash(state);
    }
}

impl PartialEq for ConnectiveBody {
    fn eq(&self, other: &Self) -> bool {
        self.ps == other.ps
    }
}

impl Hash for ConnectiveBody {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.ps.hash(state);
    }
}

impl PartialEq for DeployId {
    fn eq(&self, other: &Self) -> bool {
        self.sig == other.sig
    }
}

impl Hash for DeployId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
    }
}

impl PartialEq for DeployerId {
    fn eq(&self, other: &Self) -> bool {
        self.public_key == other.public_key
    }
}

impl Hash for DeployerId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.public_key.hash(state);
    }
}

impl PartialEq for GUnforgeable {
    fn eq(&self, other: &Self) -> bool {
        self.unf_instance == other.unf_instance
    }
}

impl Hash for GUnforgeable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.unf_instance.hash(state);
    }
}

impl PartialEq for UnfInstance {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (UnfInstance::GPrivateBody(a), UnfInstance::GPrivateBody(b)) => a == b,
            (UnfInstance::GDeployIdBody(a), UnfInstance::GDeployIdBody(b)) => a == b,
            (UnfInstance::GDeployerIdBody(a), UnfInstance::GDeployerIdBody(b)) => a == b,
            (UnfInstance::GSysAuthTokenBody(a), UnfInstance::GSysAuthTokenBody(b)) => a == b,
            _ => false,
        }
    }
}

impl Hash for UnfInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            UnfInstance::GPrivateBody(a) => a.hash(state),
            UnfInstance::GDeployIdBody(a) => a.hash(state),
            UnfInstance::GDeployerIdBody(a) => a.hash(state),
            UnfInstance::GSysAuthTokenBody(a) => a.hash(state),
        }
    }
}

impl PartialEq for GPrivate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for GPrivate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for GDeployId {
    fn eq(&self, other: &Self) -> bool {
        self.sig == other.sig
    }
}

impl Hash for GDeployId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.sig.hash(state);
    }
}

impl PartialEq for GDeployerId {
    fn eq(&self, other: &Self) -> bool {
        self.public_key == other.public_key
    }
}

impl Hash for GDeployerId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.public_key.hash(state);
    }
}

impl PartialEq for GSysAuthToken {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Hash for GSysAuthToken {
    fn hash<H: Hasher>(&self, _state: &mut H) {
        // No fields to hash
    }
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ServiceError: {:#?}", self.messages)
    }
}

// Implement the Error trait
impl Error for ServiceError {}

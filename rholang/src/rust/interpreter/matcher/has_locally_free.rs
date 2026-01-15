use crate::rust::interpreter::matcher::spatial_matcher::SpatialMatcherContext;

use models::{create_bit_vector, rust::utils::union};

use super::exports::*;

// See models/src/main/scala/coop/rchain/models/HasLocallyFree.scala
pub trait HasLocallyFree<T> {
    /** Return true if a connective (including free variables and wildcards) is
     *  used anywhere in {@code source}.
     *  @param source the object in question
     *  Specifically looks for constructions that make a pattern non-concrete.
     *  A non-concrete pattern cannot be viewed as if it were a term.
     */
    fn connective_used(&self, source: T) -> bool;

    /** Returns a bitset representing which variables are locally free if the term
     *  is located at depth {@code depth}
     *  @param source the object in question
     *  @param depth pattern nesting depth
     *  This relies on cached values based on the actual depth of a term and will
     *  only return the correct answer if asked about the actual depth of a term.
     *  The reason we pass depth is that building the caches calls this API and for
     *  the few instances where we don't rely on the cache, we need to know the
     *  depth.
     *
     *  Depth is related to pattern nesting. A top level term is depth 0. A pattern
     *  in a top-level term is depth 1. A pattern in a depth 1 term is depth 2,
     *  etc.
     */
    fn locally_free(&self, source: T, depth: i32) -> Vec<u8>;
}

// forTuple
impl HasLocallyFree<(Par, Par)> for SpatialMatcherContext {
    fn connective_used(&self, source: (Par, Par)) -> bool {
        self.connective_used(source.0) || self.connective_used(source.1)
    }

    fn locally_free(&self, source: (Par, Par), depth: i32) -> Vec<u8> {
        union(
            self.locally_free(source.0, depth),
            self.locally_free(source.1, depth),
        )
    }
}

// See models/src/main/scala/coop/rchain/models/rholang/implicits.scala - line 357 and beyond
impl HasLocallyFree<Par> for SpatialMatcherContext {
    fn connective_used(&self, p: Par) -> bool {
        p.connective_used
    }

    fn locally_free(&self, p: Par, _depth: i32) -> Vec<u8> {
        p.locally_free
    }
}

impl HasLocallyFree<Bundle> for SpatialMatcherContext {
    fn connective_used(&self, _source: Bundle) -> bool {
        false
    }

    fn locally_free(&self, source: Bundle, _depth: i32) -> Vec<u8> {
        source.body.unwrap().locally_free
    }
}

impl HasLocallyFree<Send> for SpatialMatcherContext {
    fn connective_used(&self, s: Send) -> bool {
        s.connective_used
    }

    fn locally_free(&self, s: Send, _depth: i32) -> Vec<u8> {
        s.locally_free
    }
}

impl HasLocallyFree<GUnforgeable> for SpatialMatcherContext {
    fn connective_used(&self, _unf: GUnforgeable) -> bool {
        false
    }

    fn locally_free(&self, _s: GUnforgeable, _depth: i32) -> Vec<u8> {
        Default::default()
    }
}

impl HasLocallyFree<Expr> for SpatialMatcherContext {
    fn connective_used(&self, e: Expr) -> bool {
        match e.expr_instance {
            Some(GBool(_)) => false,
            Some(GInt(_)) => false,
            Some(GString(_)) => false,
            Some(GUri(_)) => false,
            Some(GByteArray(_)) => false,

            Some(EListBody(e)) => e.connective_used,
            Some(ETupleBody(e)) => e.connective_used,
            Some(ESetBody(e)) => e.connective_used,
            Some(EMapBody(e)) => e.connective_used,
            Some(EPathmapBody(e)) => e.connective_used,
            Some(EZipperBody(e)) => e.connective_used,

            Some(EVarBody(EVar { v })) => self.connective_used(v.unwrap()),
            Some(ENotBody(ENot { p })) => p.unwrap().connective_used,
            Some(ENegBody(ENeg { p })) => p.unwrap().connective_used,

            Some(EMultBody(EMult { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EDivBody(EDiv { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EModBody(EMod { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EPlusBody(EPlus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EMinusBody(EMinus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ELtBody(ELt { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ELteBody(ELte { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EGtBody(EGt { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EGteBody(EGte { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EEqBody(EEq { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ENeqBody(ENeq { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EAndBody(EAnd { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EOrBody(EOr { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }

            Some(EMethodBody(e)) => e.connective_used,
            Some(EFunctionBody(e)) => e.connective_used,
            Some(EMatchesBody(EMatches { target, .. })) => target.unwrap().connective_used,

            Some(EPercentPercentBody(EPercentPercent { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EPlusPlusBody(EPlusPlus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EMinusMinusBody(EMinusMinus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }

            // EFree: theory specification marker - propagate from body
            Some(EFreeBody(e)) => e.body.map_or(false, |b| b.connective_used),

            None => false,
        }
    }

    fn locally_free(&self, e: Expr, depth: i32) -> Vec<u8> {
        match e.expr_instance {
            Some(GBool(_)) => Default::default(),
            Some(GInt(_)) => Default::default(),
            Some(GString(_)) => Default::default(),
            Some(GUri(_)) => Default::default(),
            Some(GByteArray(_)) => Default::default(),

            Some(EListBody(e)) => e.locally_free,
            Some(ETupleBody(e)) => e.locally_free,
            Some(ESetBody(e)) => e.locally_free,
            Some(EMapBody(e)) => e.locally_free,
            Some(EPathmapBody(e)) => e.locally_free,
            Some(EZipperBody(e)) => e.locally_free,

            Some(EVarBody(EVar { v })) => self.locally_free(v.unwrap(), depth),
            Some(ENotBody(ENot { p })) => p.unwrap().locally_free,
            Some(ENegBody(ENeg { p })) => p.unwrap().locally_free,

            Some(EMultBody(EMult { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EDivBody(EDiv { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EModBody(EMod { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EPlusBody(EPlus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EMinusBody(EMinus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ELtBody(ELt { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ELteBody(ELte { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EGtBody(EGt { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EGteBody(EGte { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EEqBody(EEq { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ENeqBody(ENeq { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EAndBody(EAnd { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EOrBody(EOr { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }

            Some(EMethodBody(e)) => e.locally_free,
            Some(EFunctionBody(e)) => e.locally_free,
            Some(EMatchesBody(EMatches { target, .. })) => target.unwrap().locally_free,

            Some(EPercentPercentBody(EPercentPercent { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EPlusPlusBody(EPlusPlus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EMinusMinusBody(EMinusMinus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }

            // EFree: theory specification marker - propagate from body
            Some(EFreeBody(e)) => e.body.map_or(Default::default(), |b| b.locally_free),

            None => Default::default(),
        }
    }
}

impl HasLocallyFree<New> for SpatialMatcherContext {
    fn connective_used(&self, n: New) -> bool {
        n.p.unwrap().connective_used
    }

    fn locally_free(&self, n: New, _depth: i32) -> Vec<u8> {
        n.locally_free
    }
}

impl HasLocallyFree<VarInstance> for SpatialMatcherContext {
    fn connective_used(&self, v: VarInstance) -> bool {
        match v {
            BoundVar(_) => false,
            FreeVar(_) => true,
            Wildcard(_) => true,
        }
    }

    fn locally_free(&self, v: VarInstance, depth: i32) -> Vec<u8> {
        match v {
            BoundVar(index) => {
                if depth == 0 {
                    create_bit_vector(&vec![index as usize])
                } else {
                    Default::default()
                }
            }
            FreeVar(_) => Default::default(),
            Wildcard(_) => Default::default(),
        }
    }
}

impl HasLocallyFree<Var> for SpatialMatcherContext {
    fn connective_used(&self, v: Var) -> bool {
        self.connective_used(v.var_instance.unwrap())
    }

    fn locally_free(&self, v: Var, depth: i32) -> Vec<u8> {
        self.locally_free(v.var_instance.unwrap(), depth)
    }
}

impl HasLocallyFree<Receive> for SpatialMatcherContext {
    fn connective_used(&self, r: Receive) -> bool {
        r.connective_used
    }

    fn locally_free(&self, r: Receive, _depth: i32) -> Vec<u8> {
        r.locally_free
    }
}

impl HasLocallyFree<ReceiveBind> for SpatialMatcherContext {
    fn connective_used(&self, rb: ReceiveBind) -> bool {
        self.connective_used(rb.source.unwrap())
    }

    fn locally_free(&self, rb: ReceiveBind, depth: i32) -> Vec<u8> {
        union(
            self.locally_free(rb.source.unwrap(), depth),
            rb.patterns.iter().fold(Default::default(), |acc, pat| {
                union(acc, self.locally_free(pat.to_owned(), depth + 1))
            }),
        )
    }
}

impl HasLocallyFree<Match> for SpatialMatcherContext {
    fn connective_used(&self, m: Match) -> bool {
        m.connective_used
    }

    fn locally_free(&self, m: Match, _depth: i32) -> Vec<u8> {
        m.locally_free
    }
}

impl HasLocallyFree<MatchCase> for SpatialMatcherContext {
    fn connective_used(&self, mc: MatchCase) -> bool {
        mc.source.unwrap().connective_used
    }

    fn locally_free(&self, mc: MatchCase, depth: i32) -> Vec<u8> {
        union(
            mc.source.unwrap().locally_free,
            self.locally_free(mc.pattern.unwrap(), depth + 1),
        )
    }
}

impl HasLocallyFree<Connective> for SpatialMatcherContext {
    fn connective_used(&self, conn: Connective) -> bool {
        match conn.connective_instance {
            Some(ConnAndBody(_)) => true,
            Some(ConnOrBody(_)) => true,
            Some(ConnNotBody(_)) => true,
            Some(VarRefBody(_)) => false,
            Some(ConnBool(_)) => true,
            Some(ConnInt(_)) => true,
            Some(ConnString(_)) => true,
            Some(ConnUri(_)) => true,
            Some(ConnByteArray(_)) => true,
            None => false,
        }
    }

    fn locally_free(&self, conn: Connective, depth: i32) -> Vec<u8> {
        match conn.connective_instance {
            Some(VarRefBody(VarRef {
                index: idx,
                depth: var_depth,
            })) => {
                if depth == var_depth {
                    create_bit_vector(&vec![idx as usize])
                } else {
                    Default::default()
                }
            }
            _ => Default::default(),
        }
    }
}

impl HasLocallyFree<VarInstance> for VarInstance {
    fn connective_used(&self, v: VarInstance) -> bool {
        match v {
            BoundVar(_) => false,
            FreeVar(_) => true,
            Wildcard(_) => true,
        }
    }

    fn locally_free(&self, v: VarInstance, depth: i32) -> Vec<u8> {
        match v {
            BoundVar(index) => {
                if depth == 0 {
                    create_bit_vector(&vec![index as usize])
                } else {
                    Default::default()
                }
            }
            FreeVar(_) => Default::default(),
            Wildcard(_) => Default::default(),
        }
    }
}

impl HasLocallyFree<Var> for Var {
    fn connective_used(&self, v: Var) -> bool {
        v.clone()
            .var_instance
            .unwrap()
            .connective_used(v.var_instance.unwrap())
    }

    fn locally_free(&self, v: Var, depth: i32) -> Vec<u8> {
        v.clone()
            .var_instance
            .unwrap()
            .locally_free(v.var_instance.unwrap(), depth)
    }
}

impl HasLocallyFree<Expr> for Expr {
    fn connective_used(&self, e: Expr) -> bool {
        match e.expr_instance {
            Some(GBool(_)) => false,
            Some(GInt(_)) => false,
            Some(GString(_)) => false,
            Some(GUri(_)) => false,
            Some(GByteArray(_)) => false,

            Some(EListBody(e)) => e.connective_used,
            Some(ETupleBody(e)) => e.connective_used,
            Some(ESetBody(e)) => e.connective_used,
            Some(EMapBody(e)) => e.connective_used,
            Some(EPathmapBody(e)) => e.connective_used,
            Some(EZipperBody(e)) => e.connective_used,

            Some(EVarBody(EVar { v })) => v.clone().unwrap().connective_used(v.unwrap()),
            Some(ENotBody(ENot { p })) => p.unwrap().connective_used,
            Some(ENegBody(ENeg { p })) => p.unwrap().connective_used,

            Some(EMultBody(EMult { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EDivBody(EDiv { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EModBody(EMod { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EPlusBody(EPlus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EMinusBody(EMinus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ELtBody(ELt { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ELteBody(ELte { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EGtBody(EGt { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EGteBody(EGte { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EEqBody(EEq { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(ENeqBody(ENeq { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EAndBody(EAnd { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EOrBody(EOr { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }

            Some(EMethodBody(e)) => e.connective_used,
            Some(EFunctionBody(e)) => e.connective_used,
            Some(EMatchesBody(EMatches { target, .. })) => target.unwrap().connective_used,

            Some(EPercentPercentBody(EPercentPercent { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EPlusPlusBody(EPlusPlus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }
            Some(EMinusMinusBody(EMinusMinus { p1, p2 })) => {
                p1.unwrap().connective_used | p2.unwrap().connective_used
            }

            // EFree: theory specification marker - propagate from body
            Some(EFreeBody(e)) => e.body.map_or(false, |b| b.connective_used),

            None => false,
        }
    }

    fn locally_free(&self, e: Expr, depth: i32) -> Vec<u8> {
        match e.expr_instance {
            Some(GBool(_)) => Default::default(),
            Some(GInt(_)) => Default::default(),
            Some(GString(_)) => Default::default(),
            Some(GUri(_)) => Default::default(),
            Some(GByteArray(_)) => Default::default(),

            Some(EListBody(e)) => e.locally_free,
            Some(ETupleBody(e)) => e.locally_free,
            Some(ESetBody(e)) => e.locally_free,
            Some(EMapBody(e)) => e.locally_free,
            Some(EPathmapBody(e)) => e.locally_free,
            Some(EZipperBody(e)) => e.locally_free,

            Some(EVarBody(EVar { v })) => v.clone().unwrap().locally_free(v.unwrap(), depth),
            Some(ENotBody(ENot { p })) => p.unwrap().locally_free,
            Some(ENegBody(ENeg { p })) => p.unwrap().locally_free,

            Some(EMultBody(EMult { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EDivBody(EDiv { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EModBody(EMod { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EPlusBody(EPlus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EMinusBody(EMinus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ELtBody(ELt { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ELteBody(ELte { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EGtBody(EGt { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EGteBody(EGte { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EEqBody(EEq { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(ENeqBody(ENeq { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EAndBody(EAnd { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EOrBody(EOr { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }

            Some(EMethodBody(e)) => e.locally_free,
            Some(EFunctionBody(e)) => e.locally_free,
            Some(EMatchesBody(EMatches { target, .. })) => target.unwrap().locally_free,

            Some(EPercentPercentBody(EPercentPercent { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EPlusPlusBody(EPlusPlus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }
            Some(EMinusMinusBody(EMinusMinus { p1, p2 })) => {
                union(p1.unwrap().locally_free, p2.unwrap().locally_free)
            }

            // EFree: theory specification marker - propagate from body
            Some(EFreeBody(e)) => e.body.map_or(Default::default(), |b| b.locally_free),

            None => Default::default(),
        }
    }
}

impl HasLocallyFree<Connective> for Connective {
    fn connective_used(&self, conn: Connective) -> bool {
        match conn.connective_instance {
            Some(ConnAndBody(_)) => true,
            Some(ConnOrBody(_)) => true,
            Some(ConnNotBody(_)) => true,
            Some(VarRefBody(_)) => false,
            Some(ConnBool(_)) => true,
            Some(ConnInt(_)) => true,
            Some(ConnString(_)) => true,
            Some(ConnUri(_)) => true,
            Some(ConnByteArray(_)) => true,
            None => false,
        }
    }

    fn locally_free(&self, conn: Connective, depth: i32) -> Vec<u8> {
        match conn.connective_instance {
            Some(VarRefBody(VarRef {
                index: idx,
                depth: var_depth,
            })) => {
                if depth == var_depth {
                    create_bit_vector(&vec![idx as usize])
                } else {
                    Default::default()
                }
            }
            _ => Default::default(),
        }
    }
}

impl HasLocallyFree<Par> for Par {
    fn connective_used(&self, p: Par) -> bool {
        p.connective_used
    }

    fn locally_free(&self, p: Par, _depth: i32) -> Vec<u8> {
        p.locally_free
    }
}

impl HasLocallyFree<New> for New {
    fn connective_used(&self, n: New) -> bool {
        n.p.unwrap().connective_used
    }

    fn locally_free(&self, n: New, _depth: i32) -> Vec<u8> {
        n.locally_free
    }
}

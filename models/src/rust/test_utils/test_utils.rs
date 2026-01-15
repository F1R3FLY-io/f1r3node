use crate::rhoapi::{
    connective, expr, var, Bundle, Connective, ENot, Match, MatchCase, New, Par, Receive,
    ReceiveBind, Send,
};
use crate::rhoapi::{Expr, Var};
use crate::rust::rholang::sorter::par_sort_matcher::ParSortMatcher;
use crate::rust::rholang::sorter::sortable::Sortable;
use proptest::prelude::*;

// models/src/test/scala/coop/rchain/models/testUtils/TestUtils.scala
pub fn sort(par: &Par) -> Par {
    ParSortMatcher::sort_match(par).term
}

pub fn for_all_similar_a<A: Clone + std::fmt::Debug>(
    generator: BoxedStrategy<A>,
    block: impl Fn(&A, &A),
) {
    proptest!(|(values in prop::collection::vec(generator, 5))| {
        for x in &values {
            for y in &values {
                block(x, y);
            }
        }
    });
}

/*
In this code, we chose a manual approach to generate recursive structures for testing instead of relying on the automatic Arbitrary trait provided by proptest. Here's why:

1. Recursive Nature of Structures:
   Many Rholang data structures (e.g., `Par`, `Bundle`, `Match`, etc.) have recursive fields, allowing them to contain other instances of similar structures. This makes automatic generation challenging and prone to issues like infinite recursion or stack overflow.

2. Control Over Depth:
   By using a manual generator with a `depth` parameter, we can explicitly limit the depth of recursion during generation. This ensures that:
   - Tests remain performant.
   - We avoid excessively nested structures that may cause panics or unmanageable complexity.

3. Handling Optional Fields:
   Some fields, such as `Option<Par>` in `Bundle`, require non-`None` values for meaningful tests. The automatic approach doesn't guarantee that these fields will be populated correctly, whereas manual generation allows us to enforce such constraints.

4. Custom Combinations:
   Recursive structures often involve multiple variants and edge cases. Manual generators let us explicitly define how these combinations are generated, providing better test coverage and control compared to the default Arbitrary implementation.

5. Avoiding Arbitrary's Limitations:
   While proptest's Arbitrary trait simplifies test data generation for flat or non-recursive types, it struggles with deeply recursive or highly interdependent types. Manual generation avoids the inherent limitations of the trait and ensures reliable and predictable behavior.
*/

pub fn generate_par(depth: usize) -> BoxedStrategy<Par> {
    if depth == 0 {
        return Just(Par {
            sends: vec![],
            receives: vec![],
            news: vec![],
            exprs: vec![],
            matches: vec![],
            bundles: vec![],
            connectives: vec![],
            locally_free: vec![],
            connective_used: false,
            unforgeables: vec![],
            use_blocks: vec![],  // Reifying RSpaces
        })
        .boxed();
    }

    (
        proptest::collection::vec(generate_send(depth - 1), 0..1),
        proptest::collection::vec(generate_receive(depth - 1), 0..1),
        proptest::collection::vec(generate_new(depth - 1), 0..1),
        proptest::collection::vec(generate_expr(depth - 1), 0..1),
        proptest::collection::vec(generate_match(depth - 1), 0..1),
        proptest::collection::vec(generate_bundle(depth - 1), 0..1),
        proptest::collection::vec(generate_connective(depth - 1), 0..1),
        proptest::collection::vec(any::<u8>(), 0..1),
        any::<bool>(),
    )
        .prop_map(
            |(
                sends,
                receives,
                news,
                exprs,
                matches,
                bundles,
                connectives,
                locally_free,
                connective_used,
            )| Par {
                sends,
                receives,
                news,
                exprs,
                matches,
                bundles,
                connectives,
                locally_free,
                connective_used,
                unforgeables: vec![],
                use_blocks: vec![],  // Reifying RSpaces
            },
        )
        .boxed()
}

pub fn generate_send(depth: usize) -> BoxedStrategy<Send> {
    if depth == 0 {
        return Just(Send {
            chan: Some(
                generate_par(0)
                    .boxed()
                    .new_tree(&mut Default::default())
                    .unwrap()
                    .current(),
            ),
            data: vec![],
            persistent: false,
            locally_free: vec![],
            connective_used: false,
            hyperparams: vec![],  // Hyperparam support
        })
        .boxed();
    }

    (
        generate_par(depth - 1),
        proptest::collection::vec(generate_par(depth - 1), 0..1),
        any::<bool>(),
        proptest::collection::vec(any::<u8>(), 0..1),
        any::<bool>(),
    )
        .prop_map(
            |(chan, data, persistent, locally_free, connective_used)| Send {
                chan: Some(chan),
                data,
                persistent,
                locally_free,
                connective_used,
                hyperparams: vec![],  // Hyperparam support
            },
        )
        .boxed()
}

pub fn generate_receive(depth: usize) -> BoxedStrategy<Receive> {
    if depth == 0 {
        return Just(Receive {
            binds: vec![],
            body: Some(
                generate_par(0)
                    .boxed()
                    .new_tree(&mut Default::default())
                    .unwrap()
                    .current(),
            ),
            persistent: false,
            peek: false,
            bind_count: 0,
            locally_free: vec![],
            connective_used: false,
        })
        .boxed();
    }

    (
        proptest::collection::vec(
            (
                proptest::collection::vec(generate_par(depth - 1), 0..1),
                generate_par(depth - 1),
                generate_option_var(depth),
                any::<i32>(),
            )
                .prop_map(|(patterns, source, remainder, free_count)| ReceiveBind {
                    patterns,
                    source: Some(source),
                    remainder,
                    free_count,
                    pattern_modifiers: vec![],
                }),
            0..1,
        ),
        generate_par(depth - 1),
        any::<bool>(),
        any::<bool>(),
        any::<i32>(),
        proptest::collection::vec(any::<u8>(), 0..1),
        any::<bool>(),
    )
        .prop_map(
            |(binds, body, persistent, peek, bind_count, locally_free, connective_used)| Receive {
                binds,
                body: Some(body),
                persistent,
                peek,
                bind_count,
                locally_free,
                connective_used,
            },
        )
        .boxed()
}

pub fn generate_new(depth: usize) -> BoxedStrategy<New> {
    if depth == 0 {
        return Just(New {
            bind_count: 0,
            p: Some(
                generate_par(0)
                    .boxed()
                    .new_tree(&mut Default::default())
                    .unwrap()
                    .current(),
            ),
            uri: vec![],
            injections: Default::default(),
            locally_free: vec![],
            space_types: vec![],
        })
        .boxed();
    }

    (
        any::<i32>(),
        generate_par(depth - 1),
        proptest::collection::vec(any::<String>(), 0..1),
        proptest::collection::btree_map(any::<String>(), generate_par(depth - 1), 0..1),
        proptest::collection::vec(any::<u8>(), 0..1),
    )
        .prop_map(|(bind_count, p, uri, injections, locally_free)| New {
            bind_count,
            p: Some(p),
            uri,
            injections,
            locally_free,
            space_types: vec![],
        })
        .boxed()
}

pub fn generate_expr(depth: usize) -> BoxedStrategy<Expr> {
    if depth == 0 {
        return Just(Expr {
            expr_instance: None,
        })
        .boxed();
    }

    proptest::strategy::Union::new(vec![
        any::<bool>()
            .prop_map(|v| Expr {
                expr_instance: Some(expr::ExprInstance::GBool(v)),
            })
            .boxed(),
        any::<i64>()
            .prop_map(|v| Expr {
                expr_instance: Some(expr::ExprInstance::GInt(v)),
            })
            .boxed(),
        any::<String>()
            .prop_map(|v| Expr {
                expr_instance: Some(expr::ExprInstance::GString(v)),
            })
            .boxed(),
        generate_par(depth - 1)
            .prop_map(|p| Expr {
                expr_instance: Some(expr::ExprInstance::ENotBody(ENot { p: Some(p) })),
            })
            .boxed(),
    ])
    .boxed()
}

pub fn generate_match(depth: usize) -> BoxedStrategy<Match> {
    if depth == 0 {
        return Just(Match {
            target: Some(
                generate_par(0)
                    .boxed()
                    .new_tree(&mut Default::default())
                    .unwrap()
                    .current(),
            ),
            cases: vec![],
            locally_free: vec![],
            connective_used: false,
        })
        .boxed();
    }

    (
        generate_par(depth - 1),
        proptest::collection::vec(
            (
                generate_par(depth - 1),
                generate_par(depth - 1),
                any::<i32>(),
            )
                .prop_map(|(pattern, source, free_count)| MatchCase {
                    pattern: Some(pattern),
                    source: Some(source),
                    free_count,
                }),
            0..1,
        ),
        proptest::collection::vec(any::<u8>(), 0..1),
        any::<bool>(),
    )
        .prop_map(|(target, cases, locally_free, connective_used)| Match {
            target: Some(target),
            cases,
            locally_free,
            connective_used,
        })
        .boxed()
}

pub fn generate_bundle(depth: usize) -> BoxedStrategy<Bundle> {
    if depth == 0 {
        return Just(Bundle {
            body: Some(
                generate_par(0)
                    .boxed()
                    .new_tree(&mut Default::default())
                    .unwrap()
                    .current(),
            ),
            write_flag: false,
            read_flag: false,
        })
        .boxed();
    }

    (generate_par(depth - 1), any::<bool>(), any::<bool>())
        .prop_map(|(body, write_flag, read_flag)| Bundle {
            body: Some(body),
            write_flag,
            read_flag,
        })
        .boxed()
}

pub fn generate_connective(depth: usize) -> BoxedStrategy<Connective> {
    if depth == 0 {
        return Just(Connective {
            connective_instance: None,
        })
        .boxed();
    }

    proptest::strategy::Union::new(vec![
        generate_par(depth - 1)
            .prop_map(|p| Connective {
                connective_instance: Some(connective::ConnectiveInstance::ConnNotBody(p)),
            })
            .boxed(),
        any::<bool>()
            .prop_map(|v| Connective {
                connective_instance: Some(connective::ConnectiveInstance::ConnBool(v)),
            })
            .boxed(),
    ])
    .boxed()
}

pub fn generate_var(depth: usize) -> BoxedStrategy<Var> {
    if depth == 0 {
        return Just(Var { var_instance: None }).boxed();
    }

    proptest::strategy::Union::new(vec![
        any::<i32>()
            .prop_map(|v| Var {
                var_instance: Some(var::VarInstance::BoundVar(v)),
            })
            .boxed(),
        any::<i32>()
            .prop_map(|v| Var {
                var_instance: Some(var::VarInstance::FreeVar(v)),
            })
            .boxed(),
        Just(Var {
            var_instance: Some(var::VarInstance::Wildcard(var::WildcardMsg {})),
        })
        .boxed(),
    ])
    .boxed()
}

pub fn generate_option_var(depth: usize) -> BoxedStrategy<Option<Var>> {
    if depth == 0 {
        return Just(None).boxed();
    }

    proptest::option::of(generate_var(depth)).boxed()
}

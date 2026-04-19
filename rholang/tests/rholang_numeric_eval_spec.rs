use models::rhoapi::{expr::ExprInstance, Expr, Par};
use rholang::rust::interpreter::{
    errors::InterpreterError,
    interpreter::EvaluateResult,
    rho_runtime::{RhoRuntime, RhoRuntimeImpl},
    test_utils::resources::with_runtime,
};
use std::collections::HashSet;

async fn execute(
    runtime: &mut RhoRuntimeImpl,
    term: &str,
) -> Result<EvaluateResult, InterpreterError> {
    runtime.evaluate_with_term(term).await
}

async fn eval_ok(runtime: &mut RhoRuntimeImpl, term: &str) {
    let res = execute(runtime, term).await.unwrap();
    assert!(
        res.errors.is_empty(),
        "Expected success for: {}\nErrors: {:?}",
        term,
        res.errors
    );
}

async fn eval_err(runtime: &mut RhoRuntimeImpl, term: &str) {
    let res = execute(runtime, term).await.unwrap();
    assert!(
        !res.errors.is_empty(),
        "Expected error for: {}\nGot success",
        term
    );
}

async fn channel_data(runtime: &RhoRuntimeImpl, channel_expr: ExprInstance) -> HashSet<Par> {
    let ch = vec![Par {
        exprs: vec![Expr {
            expr_instance: Some(channel_expr),
        }],
        ..Default::default()
    }];
    runtime
        .get_hot_changes()
        .await
        .get(&ch)
        .map(|row| row.data.iter().flat_map(|d| d.a.pars.clone()).collect())
        .unwrap_or_default()
}

fn int_channel(n: i64) -> ExprInstance {
    ExprInstance::GInt(n)
}

fn has_par_with_bool(data: &HashSet<Par>, expected: bool) -> bool {
    data.iter().any(|p| {
        p.exprs
            .iter()
            .any(|e| e.expr_instance == Some(ExprInstance::GBool(expected)))
    })
}

fn has_par_with_double(data: &HashSet<Par>, expected: f64) -> bool {
    let bits = expected.to_bits();
    data.iter().any(|p| {
        p.exprs
            .iter()
            .any(|e| e.expr_instance == Some(ExprInstance::GDouble(bits)))
    })
}

fn has_par_with_string(data: &HashSet<Par>, expected: &str) -> bool {
    data.iter().any(|p| {
        p.exprs
            .iter()
            .any(|e| e.expr_instance == Some(ExprInstance::GString(expected.to_string())))
    })
}

// --- Example file tests ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn numeric_types_example_evaluates_without_errors() {
    with_runtime("numeric-types-example-", |mut runtime| async move {
        let source = include_str!("../examples/numeric-types.rho");
        eval_ok(&mut runtime, source).await;
    })
    .await
}

// --- Float end-to-end ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_arithmetic_produces_correct_values() {
    with_runtime("float-arith-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1.5f64 + 2.25f64) | @1!(3.0f64 * 4.0f64) | @2!(10.0f64 / 4.0f64) | @3!(10.0f64 - 3.5f64)"#,
        )
        .await;

        let ch0 = channel_data(&runtime, int_channel(0)).await;
        let ch1 = channel_data(&runtime, int_channel(1)).await;
        let ch2 = channel_data(&runtime, int_channel(2)).await;
        let ch3 = channel_data(&runtime, int_channel(3)).await;
        assert!(has_par_with_double(&ch0, 3.75));
        assert!(has_par_with_double(&ch1, 12.0));
        assert!(has_par_with_double(&ch2, 2.5));
        assert!(has_par_with_double(&ch3, 6.5));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_comparisons_produce_correct_booleans() {
    with_runtime("float-cmp-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1.0f64 < 2.0f64) | @1!(2.0f64 < 1.0f64) | @2!(1.0f64 <= 1.0f64) | @3!(2.0f64 > 1.0f64)"#,
        )
        .await;

        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(0)).await, true));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(1)).await, false));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(2)).await, true));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(3)).await, true));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_division_by_zero_produces_ieee754_values() {
    with_runtime("float-divz-", |mut runtime| async move {
        eval_ok(&mut runtime, r#"@0!(1.0f64 / 0.0f64)"#).await;
        assert!(has_par_with_double(
            &channel_data(&runtime, int_channel(0)).await,
            f64::INFINITY
        ));

        eval_ok(&mut runtime, r#"@1!(-1.0f64 / 0.0f64)"#).await;
        assert!(has_par_with_double(
            &channel_data(&runtime, int_channel(1)).await,
            f64::NEG_INFINITY
        ));

        eval_ok(&mut runtime, r#"@2!(0.0f64 / 0.0f64)"#).await;
        let ch2 = channel_data(&runtime, int_channel(2)).await;
        let has_nan = ch2.iter().any(|p| {
            p.exprs.iter().any(|e| {
                matches!(&e.expr_instance, Some(ExprInstance::GDouble(bits)) if f64::from_bits(*bits).is_nan())
            })
        });
        assert!(has_nan, "0.0/0.0 should produce NaN");
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_nan_equality_follows_ieee754() {
    with_runtime("float-nan-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"
            new x in {
                x!(0.0f64 / 0.0f64) |
                for (@nan <- x) {
                    @0!(nan == nan) |
                    @1!(nan != nan)
                }
            }
            "#,
        )
        .await;

        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(0)).await, false));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(1)).await, true));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_nan_nested_in_list_follows_ieee754() {
    with_runtime("float-nan-nested-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"
            new x in {
                x!(0.0f64 / 0.0f64) |
                for (@nan <- x) {
                    @0!([nan] == [nan]) |
                    @1!([nan] != [nan])
                }
            }
            "#,
        )
        .await;

        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(0)).await, false));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(1)).await, true));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_nan_comparisons_return_false() {
    with_runtime("float-nan-cmp-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"
            new x in {
                x!(0.0f64 / 0.0f64) |
                for (@nan <- x) {
                    @0!(nan < 1.0f64) |
                    @1!(nan > 1.0f64) |
                    @2!(nan <= 1.0f64) |
                    @3!(nan >= 1.0f64)
                }
            }
            "#,
        )
        .await;

        for i in 0..4 {
            assert!(
                has_par_with_bool(&channel_data(&runtime, int_channel(i)).await, false),
                "NaN comparison on channel {} should be false",
                i
            );
        }
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn float_modulo_by_zero_is_error() {
    with_runtime("float-modz-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1.0f64 % 0.0f64)"#).await;
    })
    .await
}

// --- BigInt end-to-end ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigint_arithmetic_produces_correct_values() {
    with_runtime("bigint-arith-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(100n + 200n) | @1!(7n * 13n) | @2!(100n / 3n) | @3!(100n % 3n) | @4!(50n - 30n)"#,
        )
        .await;

        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("300n"));
        assert!(storage.contains("91n"));
        assert!(storage.contains("33n"));
        assert!(storage.contains("1n"));
        assert!(storage.contains("20n"));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigint_division_by_zero_is_error() {
    with_runtime("bigint-divz-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1n / 0n)"#).await;
        eval_err(&mut runtime, r#"@0!(1n % 0n)"#).await;
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigint_comparisons_produce_correct_booleans() {
    with_runtime("bigint-cmp-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(5n < 10n) | @1!(10n < 5n) | @2!(5n <= 5n) | @3!(10n > 5n) | @4!(5n >= 5n) | @5!(5n == 5n) | @6!(5n != 10n)"#,
        )
        .await;

        for i in [0, 2, 3, 4, 5, 6] {
            assert!(
                has_par_with_bool(&channel_data(&runtime, int_channel(i)).await, true),
                "channel {} should be true",
                i
            );
        }
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(1)).await, false));
    })
    .await
}

// --- BigRat end-to-end ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigrat_arithmetic_produces_correct_values() {
    with_runtime("bigrat-arith-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1r / 3r + 1r / 6r) | @1!(2r * 3r) | @2!(10r - 4r)"#,
        )
        .await;

        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("1/2r"));
        assert!(storage.contains("6/1r"));
        assert!(storage.contains("6/1r"));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigrat_division_by_zero_is_error() {
    with_runtime("bigrat-divz-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1r / 0r)"#).await;
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bigrat_modulo_returns_zero() {
    with_runtime("bigrat-mod-", |mut runtime| async move {
        eval_ok(&mut runtime, r#"@0!(7r % 3r)"#).await;
        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("0/1r"));
    })
    .await
}

// --- FixedPoint end-to-end ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fixedpoint_arithmetic_produces_correct_values() {
    with_runtime("fp-arith-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1.50p2 + 2.25p2) | @1!(1.5p1 * 2.0p1) | @2!(10.00p2 - 3.25p2)"#,
        )
        .await;

        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("3.75p2"));
        assert!(storage.contains("3.0p1"));
        assert!(storage.contains("6.75p2"));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fixedpoint_modulo_regression() {
    with_runtime("fp-mod-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1.50p2 % 1.00p2) | @1!(10.0p1 % 3.0p1)"#,
        )
        .await;

        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("0.50p2"));
        assert!(storage.contains("1.0p1"));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fixedpoint_scale_mismatch_is_error() {
    with_runtime("fp-scale-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1.5p1 + 2.50p2)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 - 2.50p2)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 * 2.50p2)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 / 2.50p2)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 % 2.50p2)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 < 2.50p2)"#).await;
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fixedpoint_division_by_zero_is_error() {
    with_runtime("fp-divz-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1.5p1 / 0.0p1)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 % 0.0p1)"#).await;
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fixedpoint_comparisons_produce_correct_booleans() {
    with_runtime("fp-cmp-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!(1.5p1 < 2.0p1) | @1!(2.0p1 < 1.5p1) | @2!(1.5p1 == 1.5p1) | @3!(1.5p1 != 2.0p1)"#,
        )
        .await;

        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(0)).await, true));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(1)).await, false));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(2)).await, true));
        assert!(has_par_with_bool(&channel_data(&runtime, int_channel(3)).await, true));
    })
    .await
}

// --- Cross-type errors ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_type_operations_are_errors() {
    with_runtime("cross-type-", |mut runtime| async move {
        eval_err(&mut runtime, r#"@0!(1n + 1r)"#).await;
        eval_err(&mut runtime, r#"@0!(1.0f64 + 1n)"#).await;
        eval_err(&mut runtime, r#"@0!(1.5p1 + 1n)"#).await;
        eval_err(&mut runtime, r#"@0!(1.0f64 < 1n)"#).await;
    })
    .await
}

// --- Channel-based numeric data flow ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn numeric_values_survive_channel_round_trip() {
    with_runtime("channel-rt-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"
            new ch in {
                ch!(1.5f64) |
                for (@v <- ch) { @0!(v + 2.5f64) }
            }
            "#,
        )
        .await;

        assert!(has_par_with_double(
            &channel_data(&runtime, int_channel(0)).await,
            4.0
        ));
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn numeric_values_in_lists_and_tuples() {
    with_runtime("list-tuple-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"@0!([1n, 2n, 3n]) | @1!((1.5f64, "label"))"#,
        )
        .await;

        let storage = rholang::rust::interpreter::storage::storage_printer::pretty_print(&runtime).await;
        assert!(storage.contains("1n"));
        assert!(storage.contains("2n"));
        assert!(storage.contains("3n"));
        assert!(storage.contains("1.5f64"));
    })
    .await
}

// --- Pattern matching with numeric types ---

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pattern_match_on_numeric_values() {
    with_runtime("pattern-match-", |mut runtime| async move {
        eval_ok(
            &mut runtime,
            r#"
            new ch in {
                ch!(42) |
                for (@42 <- ch) { @0!("matched") }
            }
            "#,
        )
        .await;

        assert!(has_par_with_string(
            &channel_data(&runtime, int_channel(0)).await,
            "matched"
        ));
    })
    .await
}

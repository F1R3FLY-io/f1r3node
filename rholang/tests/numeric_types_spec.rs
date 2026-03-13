use models::{
    rhoapi::{
        expr::ExprInstance, EDiv, EEq, EGt, EGte, ELt, ELte, EMinus, EMod, EMult, ENeg, ENeq,
        EPlus, Expr,
    },
    rhoapi::{BindPattern, ListParWithRandom, Par, TaggedContinuation},
    rust::utils::{
        new_gbigint_expr, new_gbigrat_expr, new_gbool_expr, new_gdouble_expr,
        new_gfixedpoint_expr, new_gint_expr,
    },
};
use rholang::rust::interpreter::{
    env::Env, errors::InterpreterError, test_utils::persistent_store_tester::create_test_space,
};
use rspace_plus_plus::rspace::rspace::RSpace;

fn gdouble_par(value: f64) -> Par {
    Par::default().with_exprs(vec![new_gdouble_expr(value)])
}

fn bigint_par(bytes: Vec<u8>) -> Par {
    Par::default().with_exprs(vec![new_gbigint_expr(bytes)])
}

fn fixedpoint_par(unscaled: Vec<u8>, scale: u32) -> Par {
    Par::default().with_exprs(vec![new_gfixedpoint_expr(unscaled, scale)])
}

fn gint_par(value: i64) -> Par {
    Par::default().with_exprs(vec![new_gint_expr(value)])
}

fn i64_to_tc(v: i64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let is_neg = v < 0;
    let trim = if is_neg { 0xFF } else { 0x00 };
    let mut start = 0;
    while start < bytes.len() - 1 {
        if bytes[start] != trim {
            break;
        }
        if (bytes[start + 1] & 0x80 != 0) != is_neg {
            break;
        }
        start += 1;
    }
    bytes[start..].to_vec()
}

fn bigint_from_i64(v: i64) -> Vec<u8> {
    if v == 0 {
        vec![0]
    } else {
        i64_to_tc(v)
    }
}

fn rat(num: i64, den: i64) -> Par {
    Par::default().with_exprs(vec![new_gbigrat_expr(
        bigint_from_i64(num),
        bigint_from_i64(den),
    )])
}

fn rat_expr(num: i64, den: i64) -> Expr {
    new_gbigrat_expr(bigint_from_i64(num), bigint_from_i64(den))
}

fn fixed(unscaled: i64, scale: u32) -> Par {
    fixedpoint_par(bigint_from_i64(unscaled), scale)
}

fn fixed_expr(unscaled: i64, scale: u32) -> Expr {
    new_gfixedpoint_expr(bigint_from_i64(unscaled), scale)
}

macro_rules! binop_expr {
    (plus, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EPlusBody(EPlus {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (minus, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMinusBody(EMinus {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (mult, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EMultBody(EMult {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (div, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EDivBody(EDiv {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (modulo, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EModBody(EMod {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (lt, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ELtBody(ELt {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (lte, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ELteBody(ELte {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (gt, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EGtBody(EGt {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (gte, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EGteBody(EGte {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (eq, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::EEqBody(EEq {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
    (neq, $p1:expr, $p2:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ENeqBody(ENeq {
                p1: Some($p1),
                p2: Some($p2),
            })),
        }])
    };
}

macro_rules! neg_expr {
    ($p:expr) => {
        Par::default().with_exprs(vec![Expr {
            expr_instance: Some(ExprInstance::ENegBody(ENeg { p: Some($p) })),
        }])
    };
}

macro_rules! setup {
    () => {{
        let (_, reducer) = create_test_space::<
            RSpace<Par, BindPattern, ListParWithRandom, TaggedContinuation>,
        >()
        .await;
        let env: Env<Par> = Env::new();
        (reducer, env)
    }};
}

macro_rules! assert_ok_expr {
    ($reducer:expr, $env:expr, $input:expr, $expected:expr) => {{
        let result = $reducer.eval_expr(&$input, &$env).unwrap();
        assert_eq!(result.exprs, vec![$expected]);
    }};
}

macro_rules! assert_err {
    ($reducer:expr, $env:expr, $input:expr, $msg:expr) => {{
        let result = $reducer.eval_expr(&$input, &$env);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            InterpreterError::ReduceError($msg.to_string())
        );
    }};
}

macro_rules! assert_err_contains {
    ($reducer:expr, $env:expr, $input:expr, $substr:expr) => {{
        let result = $reducer.eval_expr(&$input, &$env);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains($substr), "Expected '{}' in: {}", $substr, msg);
    }};
}

// ============================================================================
// GDouble (Float)
// ============================================================================

#[tokio::test]
async fn float_arithmetic() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(plus, gdouble_par(1.5), gdouble_par(2.25)), new_gdouble_expr(3.75));
    assert_ok_expr!(r, e, binop_expr!(minus, gdouble_par(5.0), gdouble_par(3.25)), new_gdouble_expr(1.75));
    assert_ok_expr!(r, e, binop_expr!(mult, gdouble_par(3.0), gdouble_par(4.5)), new_gdouble_expr(13.5));
    assert_ok_expr!(r, e, binop_expr!(div, gdouble_par(10.0), gdouble_par(4.0)), new_gdouble_expr(2.5));
    assert_ok_expr!(r, e, neg_expr!(gdouble_par(3.14)), new_gdouble_expr(-3.14));
    assert_ok_expr!(r, e, neg_expr!(gdouble_par(-2.5)), new_gdouble_expr(2.5));
}

#[tokio::test]
async fn float_division_by_zero_produces_ieee754() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(div, gdouble_par(1.0), gdouble_par(0.0)), new_gdouble_expr(f64::INFINITY));
    assert_ok_expr!(r, e, binop_expr!(div, gdouble_par(-1.0), gdouble_par(0.0)), new_gdouble_expr(f64::NEG_INFINITY));
    let nan_result = r.eval_expr(&binop_expr!(div, gdouble_par(0.0), gdouble_par(0.0)), &e).unwrap();
    match &nan_result.exprs[0].expr_instance {
        Some(ExprInstance::GDouble(bits)) => assert!(f64::from_bits(*bits).is_nan()),
        other => panic!("Expected GDouble(NaN), got {:?}", other),
    }
}

#[tokio::test]
async fn float_modulo_rejected() {
    let (r, e) = setup!();
    assert_err!(r, e, binop_expr!(modulo, gdouble_par(5.0), gdouble_par(2.0)), "Modulus not defined on floating point");
}

#[tokio::test]
async fn float_nan_equality() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(eq, gdouble_par(f64::NAN), gdouble_par(f64::NAN)), new_gbool_expr(false));
    assert_ok_expr!(r, e, binop_expr!(neq, gdouble_par(f64::NAN), gdouble_par(f64::NAN)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(eq, gdouble_par(f64::NAN), gdouble_par(42.0)), new_gbool_expr(false));
    assert_ok_expr!(r, e, binop_expr!(neq, gdouble_par(42.0), gdouble_par(f64::NAN)), new_gbool_expr(true));
}

#[tokio::test]
async fn float_nan_comparisons() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(lt, gdouble_par(f64::NAN), gdouble_par(1.0)), new_gbool_expr(false));
    assert_ok_expr!(r, e, binop_expr!(gt, gdouble_par(1.0), gdouble_par(f64::NAN)), new_gbool_expr(false));
    assert_ok_expr!(r, e, binop_expr!(lte, gdouble_par(f64::NAN), gdouble_par(f64::NAN)), new_gbool_expr(false));
    assert_ok_expr!(r, e, binop_expr!(gte, gdouble_par(f64::NAN), gdouble_par(f64::NAN)), new_gbool_expr(false));
}

#[tokio::test]
async fn float_ieee754_special_values() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(plus, gdouble_par(f64::INFINITY), gdouble_par(1.0)), new_gdouble_expr(f64::INFINITY));
    assert_ok_expr!(r, e, binop_expr!(lt, gdouble_par(f64::NEG_INFINITY), gdouble_par(f64::MAX)), new_gbool_expr(true));

    // inf - inf = NaN
    let result = r.eval_expr(&binop_expr!(minus, gdouble_par(f64::INFINITY), gdouble_par(f64::INFINITY)), &e).unwrap();
    match &result.exprs[0].expr_instance {
        Some(ExprInstance::GDouble(bits)) => assert!(f64::from_bits(*bits).is_nan()),
        other => panic!("Expected GDouble(NaN), got {:?}", other),
    }

    // -0.0 and 0.0 have different bit patterns in proto
    assert_ok_expr!(r, e, binop_expr!(eq, gdouble_par(-0.0), gdouble_par(0.0)), new_gbool_expr(false));
}

#[tokio::test]
async fn float_comparison() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(lt, gdouble_par(1.0), gdouble_par(2.0)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(gte, gdouble_par(2.0), gdouble_par(2.0)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(eq, gdouble_par(3.14), gdouble_par(3.14)), new_gbool_expr(true));
}

// ============================================================================
// BigInt
// ============================================================================

#[tokio::test]
async fn bigint_arithmetic() {
    let (r, e) = setup!();
    let bi = |v: i64| bigint_par(bigint_from_i64(v));
    let bx = |v: i64| new_gbigint_expr(bigint_from_i64(v));

    assert_ok_expr!(r, e, binop_expr!(plus, bi(100), bi(200)), bx(300));
    assert_ok_expr!(r, e, binop_expr!(minus, bi(500), bi(200)), bx(300));
    assert_ok_expr!(r, e, binop_expr!(minus, bi(10), bi(20)), bx(-10));
    assert_ok_expr!(r, e, binop_expr!(mult, bi(7), bi(13)), bx(91));
    assert_ok_expr!(r, e, binop_expr!(div, bi(100), bi(7)), bx(14));
    assert_ok_expr!(r, e, binop_expr!(modulo, bi(100), bi(7)), bx(2));
    assert_ok_expr!(r, e, neg_expr!(bi(42)), bx(-42));
    assert_ok_expr!(r, e, neg_expr!(bi(-99)), bx(99));
    assert_ok_expr!(r, e, binop_expr!(plus, bi(0), bi(5)), bx(5));
    assert_ok_expr!(r, e, binop_expr!(mult, bi(0), bi(5)), bx(0));
}

#[tokio::test]
async fn bigint_errors() {
    let (r, e) = setup!();
    let bi = |v: i64| bigint_par(bigint_from_i64(v));

    assert_err!(r, e, binop_expr!(div, bi(10), bi(0)), "Division by zero");
    assert_err!(r, e, binop_expr!(modulo, bi(10), bi(0)), "Modulo by zero");
}

#[tokio::test]
async fn bigint_comparison() {
    let (r, e) = setup!();
    let bi = |v: i64| bigint_par(bigint_from_i64(v));

    assert_ok_expr!(r, e, binop_expr!(lt, bi(5), bi(10)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(lt, bi(-100), bi(1)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(eq, bi(42), bi(42)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(neq, bi(1), bi(2)), new_gbool_expr(true));
}

// ============================================================================
// BigRat
// ============================================================================

#[tokio::test]
async fn bigrat_arithmetic() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(plus, rat(1, 3), rat(1, 6)), rat_expr(1, 2));
    assert_ok_expr!(r, e, binop_expr!(minus, rat(3, 4), rat(1, 4)), rat_expr(1, 2));
    assert_ok_expr!(r, e, binop_expr!(mult, rat(2, 3), rat(3, 4)), rat_expr(1, 2));
    assert_ok_expr!(r, e, binop_expr!(div, rat(1, 2), rat(1, 4)), rat_expr(2, 1));
    assert_ok_expr!(r, e, neg_expr!(rat(3, 4)), rat_expr(-3, 4));
    assert_ok_expr!(r, e, binop_expr!(modulo, rat(7, 3), rat(2, 5)), rat_expr(0, 1));
}

#[tokio::test]
async fn bigrat_errors() {
    let (r, e) = setup!();
    assert_err!(r, e, binop_expr!(div, rat(1, 2), rat(0, 1)), "Division by zero");
    assert_err!(r, e, binop_expr!(modulo, rat(1, 2), rat(0, 1)), "Modulo by zero");
}

#[tokio::test]
async fn bigrat_comparison() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(lt, rat(1, 3), rat(1, 2)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(gte, rat(3, 4), rat(3, 4)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(lt, rat(-1, 2), rat(1, 2)), new_gbool_expr(true));
}

// ============================================================================
// FixedPoint
// ============================================================================

#[tokio::test]
async fn fixedpoint_arithmetic() {
    let (r, e) = setup!();
    // 1.50 + 2.25 = 3.75 (scale 2)
    assert_ok_expr!(r, e, binop_expr!(plus, fixed(150, 2), fixed(225, 2)), fixed_expr(375, 2));
    // 5.00 - 3.25 = 1.75 (scale 2)
    assert_ok_expr!(r, e, binop_expr!(minus, fixed(500, 2), fixed(325, 2)), fixed_expr(175, 2));
    // 1.5 * 2.0 = 3.0 (scale-preserving: (15*20)/10 = 30)
    assert_ok_expr!(r, e, binop_expr!(mult, fixed(15, 1), fixed(20, 1)), fixed_expr(30, 1));
    // Precision loss: 0.1 * 0.1 = 0.0 (unscaled: (1*1)/10 = 0, floor)
    assert_ok_expr!(r, e, binop_expr!(mult, fixed(1, 1), fixed(1, 1)), fixed_expr(0, 1));
    // 10.0 / 3.0 = 3.3 (shifted integer division, scale preserved)
    assert_ok_expr!(r, e, binop_expr!(div, fixed(100, 1), fixed(30, 1)), fixed_expr(33, 1));
    // C99 identity: 10.0 % 3.0 = 0.1 (quotient=33, remainder=100-(33*30)/10=1)
    assert_ok_expr!(r, e, binop_expr!(modulo, fixed(100, 1), fixed(30, 1)), fixed_expr(1, 1));
    // Exact division: 6.0 % 3.0 = 0.0
    assert_ok_expr!(r, e, binop_expr!(modulo, fixed(60, 1), fixed(30, 1)), fixed_expr(0, 1));
    assert_ok_expr!(r, e, neg_expr!(fixed(150, 2)), fixed_expr(-150, 2));
}

#[tokio::test]
async fn fixedpoint_scale_mismatch_errors() {
    let (r, e) = setup!();
    assert_err_contains!(r, e, binop_expr!(plus, fixed(10, 1), fixed(10, 2)), "is not defined on FixedPoint(p2)");
    assert_err_contains!(r, e, binop_expr!(minus, fixed(10, 1), fixed(10, 2)), "is not defined on FixedPoint(p2)");
    assert_err_contains!(r, e, binop_expr!(mult, fixed(10, 1), fixed(10, 2)), "is not defined on FixedPoint(p2)");
    assert_err_contains!(r, e, binop_expr!(lt, fixed(10, 1), fixed(10, 2)), "is not defined on FixedPoint(p2)");
    assert_err_contains!(r, e, binop_expr!(modulo, fixed(10, 1), fixed(10, 2)), "is not defined on FixedPoint(p2)");
}

#[tokio::test]
async fn fixedpoint_division_errors() {
    let (r, e) = setup!();
    assert_err!(r, e, binop_expr!(div, fixed(100, 1), fixed(0, 1)), "Division by zero");
    assert_err!(r, e, binop_expr!(modulo, fixed(100, 1), fixed(0, 1)), "Modulo by zero");
}

#[tokio::test]
async fn fixedpoint_comparison() {
    let (r, e) = setup!();
    assert_ok_expr!(r, e, binop_expr!(lt, fixed(15, 1), fixed(20, 1)), new_gbool_expr(true));
    assert_ok_expr!(r, e, binop_expr!(eq, fixed(150, 2), fixed(150, 2)), new_gbool_expr(true));
}

// ============================================================================
// Cross-Type Errors
// ============================================================================

#[tokio::test]
async fn cross_type_errors() {
    let (r, e) = setup!();
    let bi = |v: i64| bigint_par(bigint_from_i64(v));

    assert_err_contains!(r, e, binop_expr!(plus, gint_par(1), gdouble_par(1.0)), "is not defined on");
    assert_err_contains!(r, e, binop_expr!(mult, bi(2), rat(1, 2)), "is not defined on");
    assert_err_contains!(r, e, binop_expr!(minus, gdouble_par(1.0), bi(1)), "is not defined on");
    assert!(r.eval_expr(&binop_expr!(lt, gint_par(1), bi(2)), &e).is_err());
    assert!(r.eval_expr(&binop_expr!(plus, fixed(10, 1), rat(1, 2)), &e).is_err());
}

// ============================================================================
// Ground Passthrough
// ============================================================================

#[tokio::test]
async fn ground_passthrough() {
    let (r, e) = setup!();
    let bi = |v: i64| bigint_par(bigint_from_i64(v));

    assert_ok_expr!(r, e, gdouble_par(3.14), new_gdouble_expr(3.14));
    assert_ok_expr!(r, e, bi(42), new_gbigint_expr(bigint_from_i64(42)));
    assert_ok_expr!(r, e, rat(3, 4), rat_expr(3, 4));
    assert_ok_expr!(r, e, fixed(150, 2), fixed_expr(150, 2));
}

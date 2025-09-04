use crate::rust::interpreter::compiler::rholang_ast::{Proc, UriLiteral};
use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Expr;
use models::rust::utils::{new_gbool_expr, new_gint_expr, new_gstring_expr, new_guri_expr};

// New AST imports for parallel functions
use rholang_parser::ast::Proc as NewProc;

/*
 This normalizer works with various types of "ground" (primitive) values, such as Bool, Int, String, and Uri.
*/
pub fn normalize_ground(proc: &Proc) -> Result<Expr, InterpreterError> {
    match proc.clone() {
        Proc::BoolLiteral { value, .. } => Ok(new_gbool_expr(value)),

        Proc::LongLiteral { value, .. } => Ok(new_gint_expr(value)),

        // The 'value' here is already stripped. This happens in custom parser.
        Proc::StringLiteral { value, .. } => Ok(new_gstring_expr(value)),

        // The 'value' here is already stripped. This happens in custom parser.
        Proc::UriLiteral(UriLiteral { value, .. }) => Ok(new_guri_expr(value)),

        _ => Err(InterpreterError::BugFoundError(format!(
            "Expected a ground type, found: {:?}",
            proc
        ))),
    }
}

/// Parallel normalizer for new AST ground types from rholang-rs parser
/// This preserves the exact same logic as normalize_ground but works directly with new AST
pub fn normalize_ground_new_ast<'ast>(proc: &NewProc<'ast>) -> Result<Expr, InterpreterError> {
    match proc {
        NewProc::BoolLiteral(value) => Ok(new_gbool_expr(*value)),

        NewProc::LongLiteral(value) => Ok(new_gint_expr(*value)),

        // The value from rholang-rs parser includes quotes, so we need to handle stripping
        NewProc::StringLiteral(value) => {
            // Convert &str to String, the string is already properly parsed by rholang-rs
            Ok(new_gstring_expr(value.to_string()))
        }

        // The value from rholang-rs parser includes backticks, handle stripping like the original
        NewProc::UriLiteral(uri) => {
            // Convert &str to String, strip backticks if present (similar to original logic)
            let uri_value = uri.to_string();
            let stripped_value = if uri_value.starts_with('`') && uri_value.ends_with('`') {
                uri_value[1..uri_value.len()-1].to_string()
            } else {
                uri_value
            };
            Ok(new_guri_expr(stripped_value))
        }

        _ => Err(InterpreterError::BugFoundError(format!(
            "Expected a ground type in new AST, found unsupported variant"
        ))),
    }
}

/*
 In the new engine, we don't have a separate BoolMatcher normalizer for BoolLiteral,
 which is why the tests with BoolMatcherSpec as well as with GroundMatcherSpec will be described below.
*/
#[cfg(test)]
mod tests {
    use super::*;
    use models::rhoapi::expr::ExprInstance;

    //rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/BoolMatcherSpec.scala
    #[test]
    fn bool_true_should_compile_as_gbool_true() {
        let proc = Proc::BoolLiteral {
            value: true,
            line_num: 1,
            col_num: 1,
        };
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(true)));
    }

    //rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/BoolMatcherSpec.scala
    #[test]
    fn bool_false_should_compile_as_gbool_false() {
        let proc = Proc::BoolLiteral {
            value: false,
            line_num: 1,
            col_num: 1,
        };
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(false)));
    }

    //rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundMatcherSpec.scala
    #[test]
    fn ground_int_should_compile_as_gint() {
        let proc = Proc::LongLiteral {
            value: 7,
            line_num: 1,
            col_num: 1,
        };
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GInt(7)));
    }

    //rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundMatcherSpec.scala
    #[test]
    fn ground_string_should_compile_as_gstring() {
        let proc = Proc::StringLiteral {
            value: "String".to_string(),
            line_num: 1,
            col_num: 1,
        };
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(
            expr.expr_instance,
            Some(ExprInstance::GString("String".to_string()))
        );
    }

    //rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundMatcherSpec.scala
    #[test]
    fn ground_uri_should_compile_as_guri() {
        let proc = Proc::UriLiteral(UriLiteral {
            value: "rho:uri".to_string(),
            line_num: 1,
            col_num: 1,
        });
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(
            expr.expr_instance,
            Some(ExprInstance::GUri("rho:uri".to_string()))
        );
    }

    #[test]
    fn unsupported_proc_should_return_bug_found_error() {
        let proc = Proc::Nil {
            line_num: 1,
            col_num: 1,
        };
        let result = normalize_ground(&proc);
        assert!(matches!(result, Err(InterpreterError::BugFoundError(_))));
    }

    // Tests for new AST normalizer - parallel to the original tests above
    #[test]
    fn new_ast_bool_true_should_compile_as_gbool_true() {
        let proc = NewProc::BoolLiteral(true);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(true)));
    }

    #[test]
    fn new_ast_bool_false_should_compile_as_gbool_false() {
        let proc = NewProc::BoolLiteral(false);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(false)));
    }

    #[test]
    fn new_ast_long_should_compile_as_gint() {
        let proc = NewProc::LongLiteral(42);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GInt(42)));
    }

    #[test]
    fn new_ast_string_should_compile_as_gstring() {
        let proc = NewProc::StringLiteral("hello");
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GString("hello".to_string())));
    }

    // TODO: URI tests omitted because Uri struct has private fields and can't be constructed in tests
    // The URI normalization logic is tested through integration tests with actual parsing

    #[test]
    fn new_ast_unsupported_type_should_return_error() {
        let proc = NewProc::Nil;
        let result = normalize_ground_new_ast(&proc);
        assert!(matches!(result, Err(InterpreterError::BugFoundError(_))));
    }
}

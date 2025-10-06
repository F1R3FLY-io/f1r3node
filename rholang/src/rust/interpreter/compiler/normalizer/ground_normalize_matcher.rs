use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Expr;
use models::rust::utils::{new_gbool_expr, new_gint_expr, new_gstring_expr, new_guri_expr};

use rholang_parser::ast::Proc as NewProc;

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
                uri_value[1..uri_value.len() - 1].to_string()
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

 rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/BoolMatcherSpec.scala
 rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/GroundMatcherSpec.scala
*/
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::{
        compiler::normalizer::ground_normalize_matcher::normalize_ground_new_ast,
        errors::InterpreterError,
    };
    use models::rhoapi::expr::ExprInstance;
    use rholang_parser::ast::Proc;

    #[test]
    fn bool_true_should_compile_as_gbool_true() {
        let proc = Proc::BoolLiteral(true);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(true)));
    }

    #[test]
    fn bool_false_should_compile_as_gbool_false() {
        let proc = Proc::BoolLiteral(false);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(false)));
    }

    #[test]
    fn long_should_compile_as_gint() {
        let proc = Proc::LongLiteral(42);
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GInt(42)));
    }

    #[test]
    fn string_should_compile_as_gstring() {
        let proc = Proc::StringLiteral("hello");
        let result = normalize_ground_new_ast(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(
            expr.expr_instance,
            Some(ExprInstance::GString("hello".to_string()))
        );
    }

    // TODO: URI tests omitted because Uri struct has private fields and can't be constructed in tests
    // The URI normalization logic is tested through integration tests with actual parsing

    #[test]
    fn unsupported_type_should_return_error() {
        let proc = Proc::Nil;
        let result = normalize_ground_new_ast(&proc);
        assert!(matches!(result, Err(InterpreterError::BugFoundError(_))));
    }
}

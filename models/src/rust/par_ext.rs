use crate::rhoapi::{expr::ExprInstance, ETuple, Par};

/// Helper functions to extract values from Par expressions
/// Equivalent to Scala's getGString, getGInt, getGBool, getETupleBody methods
pub trait ParExt {
    /// Extract a string value from a Par expression
    fn get_g_string(&self) -> Option<String>;

    /// Extract an integer value from a Par expression
    fn get_g_int(&self) -> Option<i64>;

    /// Extract a boolean value from a Par expression
    fn get_g_bool(&self) -> Option<bool>;

    /// Extract an ETuple body from a Par expression
    fn get_e_tuple_body(&self) -> Option<&ETuple>;
}

impl ParExt for Par {
    /// Extract a string value from a Par expression
    fn get_g_string(&self) -> Option<String> {
        if let Some(expr) = self.exprs.first() {
            if let Some(ExprInstance::GString(s)) = &expr.expr_instance {
                return Some(s.clone());
            }
        }
        None
    }

    /// Extract an integer value from a Par expression
    fn get_g_int(&self) -> Option<i64> {
        if let Some(expr) = self.exprs.first() {
            if let Some(ExprInstance::GInt(i)) = &expr.expr_instance {
                return Some(*i);
            }
        }
        None
    }

    /// Extract a boolean value from a Par expression
    fn get_g_bool(&self) -> Option<bool> {
        if let Some(expr) = self.exprs.first() {
            if let Some(ExprInstance::GBool(b)) = &expr.expr_instance {
                return Some(*b);
            }
        }
        None
    }

    /// Extract an ETuple body from a Par expression
    fn get_e_tuple_body(&self) -> Option<&ETuple> {
        if let Some(expr) = self.exprs.first() {
            if let Some(ExprInstance::ETupleBody(tuple)) = &expr.expr_instance {
                return Some(tuple);
            }
        }
        None
    }
}

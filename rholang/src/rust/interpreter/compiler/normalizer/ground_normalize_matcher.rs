use crate::rust::interpreter::errors::InterpreterError;
use models::rhoapi::Expr;
use models::rust::utils::{
    new_gbigint_expr, new_gbigrat_expr, new_gbool_expr, new_gdouble_expr, new_gfixedpoint_expr,
    new_gint_expr, new_gstring_expr, new_guri_expr,
};

use rholang_parser::ast::Proc as NewProc;

pub fn normalize_ground<'ast>(proc: &NewProc<'ast>) -> Result<Expr, InterpreterError> {
    match proc {
        NewProc::BoolLiteral(value) => Ok(new_gbool_expr(*value)),

        NewProc::LongLiteral(value) => Ok(new_gint_expr(*value)),

        NewProc::SignedIntLiteral { value, bits } => {
            let parsed: i64 = value.parse().map_err(|_| {
                InterpreterError::NormalizerError(format!(
                    "Invalid signed integer literal: {}",
                    value
                ))
            })?;
            if *bits <= 64 {
                Ok(new_gint_expr(parsed))
            } else {
                Ok(new_gbigint_expr(i64_to_twos_complement(parsed)))
            }
        }

        NewProc::UnsignedIntLiteral { value, bits } => {
            let parsed: u64 = value.parse().map_err(|_| {
                InterpreterError::NormalizerError(format!(
                    "Invalid unsigned integer literal: {}",
                    value
                ))
            })?;
            if *bits <= 64 && parsed <= i64::MAX as u64 {
                Ok(new_gint_expr(parsed as i64))
            } else {
                Ok(new_gbigint_expr(u64_to_twos_complement(parsed)))
            }
        }

        NewProc::BigIntLiteral(value) => {
            let s = value.trim_end_matches('n');
            let bytes = decimal_str_to_twos_complement(s)?;
            Ok(new_gbigint_expr(bytes))
        }

        NewProc::BigRatLiteral(value) => {
            let num_bytes = decimal_str_to_twos_complement(value)?;
            let den_bytes = vec![1];
            Ok(new_gbigrat_expr(num_bytes, den_bytes))
        }

        NewProc::FloatLiteral { value, .. } => {
            let f: f64 = value.parse().map_err(|_| {
                InterpreterError::NormalizerError(format!("Invalid float literal: {}", value))
            })?;
            Ok(new_gdouble_expr(f))
        }

        NewProc::FixedPointLiteral { value, scale } => {
            let unscaled_bytes = decimal_str_to_unscaled(value, *scale)?;
            Ok(new_gfixedpoint_expr(unscaled_bytes, *scale))
        }

        NewProc::StringLiteral(value) => {
            Ok(new_gstring_expr(value.to_string()))
        }

        NewProc::UriLiteral(uri) => {
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

fn i64_to_twos_complement(v: i64) -> Vec<u8> {
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

fn u64_to_twos_complement(v: u64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let mut start = 0;
    while start < bytes.len() - 1 && bytes[start] == 0 {
        if bytes[start + 1] & 0x80 != 0 {
            break;
        }
        start += 1;
    }
    bytes[start..].to_vec()
}

fn decimal_str_to_twos_complement(s: &str) -> Result<Vec<u8>, InterpreterError> {
    let s = s.trim();
    if s.is_empty() || s == "0" {
        return Ok(vec![0]);
    }
    let (negative, digits) = if s.starts_with('-') {
        (true, &s[1..])
    } else if s.starts_with('+') {
        (false, &s[1..])
    } else {
        (false, s)
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return Err(InterpreterError::NormalizerError(format!(
            "Invalid decimal number: {}",
            s
        )));
    }
    let mag_bytes = decimal_to_magnitude(digits);
    let mut result = mag_bytes;
    if result[0] & 0x80 != 0 {
        result.insert(0, 0x00);
    }
    if negative {
        result = negate_tc_bytes(&result);
    }
    Ok(result)
}

fn decimal_to_magnitude(s: &str) -> Vec<u8> {
    let mut result = vec![0u8];
    for ch in s.chars() {
        let digit = ch as u8 - b'0';
        let mut carry = digit as u16;
        for byte in result.iter_mut().rev() {
            let val = (*byte as u16) * 10 + carry;
            *byte = (val & 0xFF) as u8;
            carry = val >> 8;
        }
        while carry > 0 {
            result.insert(0, (carry & 0xFF) as u8);
            carry >>= 8;
        }
    }
    if result.is_empty() {
        vec![0]
    } else {
        result
    }
}

fn negate_tc_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut result = bytes.to_vec();
    for byte in result.iter_mut() {
        *byte = !*byte;
    }
    let mut carry = true;
    for byte in result.iter_mut().rev() {
        if carry {
            let (val, overflow) = byte.overflowing_add(1);
            *byte = val;
            carry = overflow;
        }
    }
    let is_neg = result[0] & 0x80 != 0;
    let trim = if is_neg { 0xFF } else { 0x00 };
    let mut start = 0;
    while start < result.len() - 1 {
        if result[start] != trim {
            break;
        }
        if (result[start + 1] & 0x80 != 0) != is_neg {
            break;
        }
        start += 1;
    }
    result[start..].to_vec()
}

fn decimal_str_to_unscaled(s: &str, scale: u32) -> Result<Vec<u8>, InterpreterError> {
    let s = s.trim();
    let (negative, digits) = if s.starts_with('-') {
        (true, &s[1..])
    } else {
        (false, s)
    };
    let (integer_part, frac_part) = if let Some(dot_pos) = digits.find('.') {
        (&digits[..dot_pos], &digits[dot_pos + 1..])
    } else {
        (digits, "")
    };
    let scale_usize = scale as usize;
    let padded_frac = if frac_part.len() < scale_usize {
        format!("{:0<width$}", frac_part, width = scale_usize)
    } else {
        frac_part[..scale_usize].to_string()
    };
    let unscaled_str = format!("{}{}", integer_part, padded_frac);
    let full_str = if negative {
        format!("-{}", unscaled_str)
    } else {
        unscaled_str
    };
    decimal_str_to_twos_complement(&full_str)
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
        compiler::normalizer::ground_normalize_matcher::normalize_ground, errors::InterpreterError,
    };
    use models::rhoapi::expr::ExprInstance;
    use rholang_parser::ast::Proc;

    #[test]
    fn bool_true_should_compile_as_gbool_true() {
        let proc = Proc::BoolLiteral(true);
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(true)));
    }

    #[test]
    fn bool_false_should_compile_as_gbool_false() {
        let proc = Proc::BoolLiteral(false);
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GBool(false)));
    }

    #[test]
    fn long_should_compile_as_gint() {
        let proc = Proc::LongLiteral(42);
        let result = normalize_ground(&proc);
        assert!(result.is_ok());
        let expr = result.unwrap();
        assert_eq!(expr.expr_instance, Some(ExprInstance::GInt(42)));
    }

    #[test]
    fn string_should_compile_as_gstring() {
        let proc = Proc::StringLiteral("hello");
        let result = normalize_ground(&proc);
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
        let result = normalize_ground(&proc);
        assert!(matches!(result, Err(InterpreterError::BugFoundError(_))));
    }
}

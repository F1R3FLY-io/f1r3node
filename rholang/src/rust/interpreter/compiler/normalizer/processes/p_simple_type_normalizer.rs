use crate::rust::interpreter::compiler::exports::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::Connective;

use rholang_parser::ast::SimpleType;

pub fn normalize_simple_type<'ast>(
    simple_type: &SimpleType,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let connective_instance = match simple_type {
        SimpleType::Bool => ConnectiveInstance::ConnBool(true),
        SimpleType::Int => ConnectiveInstance::ConnInt(true),
        SimpleType::String => ConnectiveInstance::ConnString(true),
        SimpleType::Uri => ConnectiveInstance::ConnUri(true),
        SimpleType::ByteArray => ConnectiveInstance::ConnByteArray(true),
    };

    let connective = Connective {
        connective_instance: Some(connective_instance),
    };

    Ok(ProcVisitOutputs {
        par: {
            let mut updated_par = prepend_connective(
                input.par.clone(),
                connective,
                input.bound_map_chain.depth() as i32,
            );
            updated_par.connective_used = true;
            updated_par
        },
        free_map: input.free_map,
    })
}

//rholang/src/test/scala/coop/rchain/rholang/interpreter/compiler/normalizer/ProcMatcherSpec.scala
#[cfg(test)]
mod tests {
    use crate::rust::interpreter::compiler::exports::ProcVisitInputs;
    use crate::rust::interpreter::compiler::normalizer::processes::p_simple_type_normalizer::normalize_simple_type;
    use models::rhoapi::connective::ConnectiveInstance::{
        ConnBool, ConnByteArray, ConnInt, ConnString, ConnUri,
    };
    use pretty_assertions::assert_eq;
    use rholang_parser::ast::SimpleType;

    #[test]
    fn simple_type_should_result_in_correct_connectives() {
        let input = ProcVisitInputs::new();

        // Test all SimpleType variants
        let result_bool = normalize_simple_type(&SimpleType::Bool, input.clone());
        let result_int = normalize_simple_type(&SimpleType::Int, input.clone());
        let result_string = normalize_simple_type(&SimpleType::String, input.clone());
        let result_uri = normalize_simple_type(&SimpleType::Uri, input.clone());
        let result_byte_array = normalize_simple_type(&SimpleType::ByteArray, input.clone());

        // Verify Bool
        assert!(result_bool.is_ok());
        let bool_par = result_bool.unwrap().par;
        assert!(bool_par.connective_used);
        assert!(!bool_par.connectives.is_empty());
        if let Some(conn) = bool_par.connectives.first() {
            assert_eq!(conn.connective_instance, Some(ConnBool(true)));
        }

        // Verify Int
        assert!(result_int.is_ok());
        let int_par = result_int.unwrap().par;
        assert!(int_par.connective_used);
        assert!(!int_par.connectives.is_empty());
        if let Some(conn) = int_par.connectives.first() {
            assert_eq!(conn.connective_instance, Some(ConnInt(true)));
        }

        // Verify String
        assert!(result_string.is_ok());
        let string_par = result_string.unwrap().par;
        assert!(string_par.connective_used);
        assert!(!string_par.connectives.is_empty());
        if let Some(conn) = string_par.connectives.first() {
            assert_eq!(conn.connective_instance, Some(ConnString(true)));
        }

        // Verify Uri
        assert!(result_uri.is_ok());
        let uri_par = result_uri.unwrap().par;
        assert!(uri_par.connective_used);
        assert!(!uri_par.connectives.is_empty());
        if let Some(conn) = uri_par.connectives.first() {
            assert_eq!(conn.connective_instance, Some(ConnUri(true)));
        }

        // Verify ByteArray
        assert!(result_byte_array.is_ok());
        let byte_array_par = result_byte_array.unwrap().par;
        assert!(byte_array_par.connective_used);
        assert!(!byte_array_par.connectives.is_empty());
        if let Some(conn) = byte_array_par.connectives.first() {
            assert_eq!(conn.connective_instance, Some(ConnByteArray(true)));
        }
    }
}

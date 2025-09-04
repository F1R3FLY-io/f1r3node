use crate::rust::interpreter::compiler::normalize::{ProcVisitInputs, ProcVisitOutputs};
use crate::rust::interpreter::compiler::rholang_ast::SimpleType;
use crate::rust::interpreter::errors::InterpreterError;
use crate::rust::interpreter::util::prepend_connective;
use models::rhoapi::connective::ConnectiveInstance;
use models::rhoapi::Connective;

// New AST imports for parallel functions
use rholang_parser::ast::SimpleType as NewSimpleType;

pub fn normalize_simple_type(
    simple_type: &SimpleType,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let connective_instance = match simple_type {
        SimpleType::Bool { .. } => ConnectiveInstance::ConnBool(true),
        SimpleType::Int { .. } => ConnectiveInstance::ConnInt(true),
        SimpleType::String { .. } => ConnectiveInstance::ConnString(true),
        SimpleType::Uri { .. } => ConnectiveInstance::ConnUri(true),
        SimpleType::ByteArray { .. } => ConnectiveInstance::ConnByteArray(true),
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

/// Parallel normalizer for new AST SimpleType from rholang-rs parser
/// This preserves the exact same logic as normalize_simple_type but works directly with new AST
pub fn normalize_simple_type_new_ast(
    simple_type: &NewSimpleType,
    input: ProcVisitInputs,
) -> Result<ProcVisitOutputs, InterpreterError> {
    let connective_instance = match simple_type {
        NewSimpleType::Bool => ConnectiveInstance::ConnBool(true),
        NewSimpleType::Int => ConnectiveInstance::ConnInt(true),
        NewSimpleType::String => ConnectiveInstance::ConnString(true),
        NewSimpleType::Uri => ConnectiveInstance::ConnUri(true),
        NewSimpleType::ByteArray => ConnectiveInstance::ConnByteArray(true),
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
    use crate::rust::interpreter::compiler::normalize::{normalize_match_proc, ProcVisitInputs};
    use crate::rust::interpreter::compiler::rholang_ast::{Proc, SimpleType};
    use crate::rust::interpreter::test_utils::utils::proc_visit_inputs_and_env;
    use models::rhoapi::connective::ConnectiveInstance::{
        ConnBool, ConnByteArray, ConnInt, ConnString, ConnUri,
    };

    use models::rhoapi::{Connective, Par};
    use pretty_assertions::assert_eq;
    
    // Imports for new AST tests
    use super::{normalize_simple_type_new_ast, NewSimpleType};

    #[test]
    fn p_simple_type_should_result_in_a_connective_of_the_correct_type() {
        let (inputs, env) = proc_visit_inputs_and_env();
        let proc_bool = Proc::SimpleType(SimpleType::new_bool());
        let proc_int = Proc::SimpleType(SimpleType::new_int());
        let proc_string = Proc::SimpleType(SimpleType::new_string());
        let proc_uri = Proc::SimpleType(SimpleType::new_uri());
        let proc_byte_array = Proc::SimpleType(SimpleType::new_bytearray());

        let result_bool = normalize_match_proc(&proc_bool, inputs.clone(), &env);
        let result_int = normalize_match_proc(&proc_int, inputs.clone(), &env);
        let result_string = normalize_match_proc(&proc_string, inputs.clone(), &env);
        let result_uri = normalize_match_proc(&proc_uri, inputs.clone(), &env);
        let result_byte_array = normalize_match_proc(&proc_byte_array, inputs.clone(), &env);

        assert_eq!(
            result_bool.unwrap().par,
            Par {
                connectives: vec![Connective {
                    connective_instance: Some(ConnBool(true))
                }],
                connective_used: true,
                ..Par::default().clone()
            }
        );

        assert_eq!(
            result_int.unwrap().par,
            Par {
                connectives: vec![Connective {
                    connective_instance: Some(ConnInt(true))
                }],
                connective_used: true,
                ..Par::default().clone()
            }
        );

        assert_eq!(
            result_string.unwrap().par,
            Par {
                connectives: vec![Connective {
                    connective_instance: Some(ConnString(true))
                }],
                connective_used: true,
                ..Par::default().clone()
            }
        );

        assert_eq!(
            result_uri.unwrap().par,
            Par {
                connectives: vec![Connective {
                    connective_instance: Some(ConnUri(true))
                }],
                connective_used: true,
                ..Par::default().clone()
            }
        );

        assert_eq!(
            result_byte_array.unwrap().par,
            Par {
                connectives: vec![Connective {
                    connective_instance: Some(ConnByteArray(true))
                }],
                connective_used: true,
                ..Par::default().clone()
            }
        );
    }

    // Tests for new AST normalizer - parallel to the original test above
    #[test]
    fn new_ast_simple_type_should_result_in_correct_connectives() {
        let input = ProcVisitInputs::new();

        // Test all SimpleType variants
        let result_bool = normalize_simple_type_new_ast(&NewSimpleType::Bool, input.clone());
        let result_int = normalize_simple_type_new_ast(&NewSimpleType::Int, input.clone());
        let result_string = normalize_simple_type_new_ast(&NewSimpleType::String, input.clone());
        let result_uri = normalize_simple_type_new_ast(&NewSimpleType::Uri, input.clone());
        let result_byte_array = normalize_simple_type_new_ast(&NewSimpleType::ByteArray, input.clone());

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

use log::debug;
use hypersonic::{CallError, fe256};
use rholang::rust::interpreter::errors::InterpreterError;

/// Custom error codes for RGB-specific failures
pub mod codes {
    pub const CONTRACT_GENERATION_ERROR: u32 = 100;
    pub const CHANNEL_MAPPING_ERROR: u32 = 101;
    pub const MEMORY_ACCESS_ERROR: u32 = 102;
    pub const STATE_SERIALIZATION_ERROR: u32 = 103;
    pub const CONTRACT_EXECUTION_ERROR: u32 = 104;
    pub const CONTRACT_VALIDATION_ERROR: u32 = 105;
    pub const DATA_EXTRACTION_ERROR: u32 = 106;
    pub const INVALID_RGB_DATA: u32 = 107;
    pub const MISSING_REQUIRED_STATE: u32 = 108;
    pub const INSUFFICIENT_BALANCE: u32 = 109;
    pub const INVALID_RECIPIENT: u32 = 110;
}

/// Create a CallError for RGB-specific contract failures
pub fn rgb_contract_error(code: u32, context: &str) -> CallError {
    debug!("🔴 RGB Contract Error [{}]: {}", code, context);
    CallError::Script(fe256::from(code))
}

/// Create a CallError for RGB data extraction failures
pub fn rgb_data_extraction_error(context: &str) -> CallError {
    debug!("🔴 RGB Data Extraction Error: {}", context);
    CallError::Script(fe256::from(codes::DATA_EXTRACTION_ERROR))
}

/// Create a CallError for invalid RGB data
pub fn rgb_invalid_data_error(context: &str) -> CallError {
    debug!("🔴 RGB Invalid Data Error: {}", context);
    CallError::Script(fe256::from(codes::INVALID_RGB_DATA))
}

/// Create a CallError for missing required RGB state
pub fn rgb_missing_state_error(context: &str) -> CallError {
    debug!("🔴 RGB Missing State Error: {}", context);
    CallError::Script(fe256::from(codes::MISSING_REQUIRED_STATE))
}

/// Create a CallError for insufficient balance
pub fn rgb_insufficient_balance_error(context: &str) -> CallError {
    debug!("🔴 RGB Insufficient Balance Error: {}", context);
    CallError::Script(fe256::from(codes::INSUFFICIENT_BALANCE))
}

/// Create a CallError for invalid recipient
pub fn rgb_invalid_recipient_error(context: &str) -> CallError {
    debug!("🔴 RGB Invalid Recipient Error: {}", context);
    CallError::Script(fe256::from(codes::INVALID_RECIPIENT))
}

/// Convert Rholang InterpreterError to RGB CallError
/// 
/// This function maps various Rholang execution errors to appropriate RGB error codes.
/// We cannot use From<> trait due to Rust orphan rules (both types are external).
pub fn interpreter_error_to_call_error(err: InterpreterError) -> CallError {
    debug!("🔴 Converting InterpreterError to CallError: {:?}", err);
    
    match err {
        // Syntax and parsing errors (1-10)
        InterpreterError::SyntaxError(_) => {
            CallError::Script(fe256::from(1u32))
        },
        InterpreterError::LexerError(_) => {
            CallError::Script(fe256::from(2u32))
        },
        InterpreterError::ParserError(_) => {
            CallError::Script(fe256::from(3u32))
        },
        InterpreterError::NormalizerError(_) => {
            CallError::Script(fe256::from(4u32))
        },
        InterpreterError::EncodeError(_) | InterpreterError::DecodeError(_) => {
            CallError::Script(fe256::from(5u32))
        },
        
        // Variable and context errors (11-20)
        InterpreterError::UnboundVariableRef { .. } => {
            CallError::Script(fe256::from(11u32))
        },
        InterpreterError::UnexpectedNameContext { .. } => {
            CallError::Script(fe256::from(12u32))
        },
        InterpreterError::UnexpectedReuseOfNameContextFree { .. } => {
            CallError::Script(fe256::from(13u32))
        },
        InterpreterError::UnexpectedProcContext { .. } => {
            CallError::Script(fe256::from(14u32))
        },
        InterpreterError::UnexpectedReuseOfProcContextFree { .. } => {
            CallError::Script(fe256::from(15u32))
        },
        
        // Runtime execution errors (21-30)
        InterpreterError::ReduceError(_) => {
            CallError::Script(fe256::from(21u32))
        },
        InterpreterError::PatternReceiveError(_) => {
            CallError::Script(fe256::from(22u32))
        },
        InterpreterError::SortMatchError(_) => {
            CallError::Script(fe256::from(23u32))
        },
        InterpreterError::MethodNotDefined { .. } => {
            CallError::Script(fe256::from(24u32))
        },
        InterpreterError::MethodArgumentNumberMismatch { .. } => {
            CallError::Script(fe256::from(25u32))
        },
        InterpreterError::OperatorNotDefined { .. } => {
            CallError::Script(fe256::from(26u32))
        },
        InterpreterError::OperatorExpectedError { .. } => {
            CallError::Script(fe256::from(27u32))
        },
        
        // Resource and system errors (31-40) 
        InterpreterError::OutOfPhlogistonsError => {
            CallError::Lock(Some(fe256::from(31u32)))
        },
        InterpreterError::RSpaceError(_) => {
            CallError::Lock(Some(fe256::from(32u32)))
        },
        InterpreterError::SetupError(_) => {
            CallError::Lock(Some(fe256::from(33u32)))
        },
        InterpreterError::IoError(_) => {
            CallError::Lock(Some(fe256::from(34u32)))
        },
        
        // Contract validation errors (41-50)
        InterpreterError::TopLevelWildcardsNotAllowedError(_) => {
            CallError::Script(fe256::from(41u32))
        },
        InterpreterError::TopLevelFreeVariablesNotAllowedError(_) => {
            CallError::Script(fe256::from(42u32))
        },
        InterpreterError::TopLevelLogicalConnectivesNotAllowedError(_) => {
            CallError::Script(fe256::from(43u32))
        },
        InterpreterError::ReceiveOnSameChannelsError { .. } => {
            CallError::Script(fe256::from(44u32))
        },
        InterpreterError::IllegalArgumentError(_) => {
            CallError::Script(fe256::from(45u32))
        },
        
        // Aggregate and complex errors (51-60)
        InterpreterError::AggregateError { .. } => {
            CallError::Script(fe256::from(51u32))
        },
        InterpreterError::SubstituteError(_) => {
            CallError::Script(fe256::from(52u32))
        },
        
        // Fallback errors (61-70)
        InterpreterError::BugFoundError(_) => {
            CallError::Script(fe256::from(61u32))
        },
        InterpreterError::UndefinedRequiredProtobufFieldError(_) => {
            CallError::Script(fe256::from(62u32))
        },
        InterpreterError::UnexpectedBundleContent(_) => {
            CallError::Script(fe256::from(63u32))
        },
        InterpreterError::UnrecognizedNormalizerError(_) => {
            CallError::Script(fe256::from(64u32))
        },
        InterpreterError::UnrecognizedInterpreterError(_) => {
            CallError::Script(fe256::from(65u32))
        },
        InterpreterError::OpenAIError(_) => {
            CallError::Script(fe256::from(66u32))
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rgb_contract_error() {
        let error = rgb_contract_error(codes::CONTRACT_GENERATION_ERROR, "test context");
        match error {
            CallError::Script(code) => {
                assert_eq!(code, fe256::from(100u32));
            },
            _ => panic!("Expected CallError::Script"),
        }
    }
    
    #[test]
    fn test_interpreter_error_mapping() {
        let syntax_error = InterpreterError::SyntaxError("test syntax error".to_string());
        let result = interpreter_error_to_call_error(syntax_error);
        
        match result {
            CallError::Script(code) => {
                assert_eq!(code, fe256::from(1u32));
            },
            _ => panic!("Expected CallError::Script for syntax error"),
        }
    }
    
    #[test]  
    fn test_resource_error_mapping() {
        let phlogiston_error = InterpreterError::OutOfPhlogistonsError;
        let result = interpreter_error_to_call_error(phlogiston_error);
        
        match result {
            CallError::Lock(Some(code)) => {
                assert_eq!(code, fe256::from(31u32));
            },
            _ => panic!("Expected CallError::Lock(Some(_)) for phlogiston error"),
        }
    }
}

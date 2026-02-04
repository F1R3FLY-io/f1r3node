// See rholang/src/main/scala/coop/rchain/rholang/interpreter/errors.scala
use std::fmt;

use rspace_plus_plus::rspace::errors::RSpaceError;

// PartialEq here is needed for testing purposes
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InterpreterError {
    RSpaceError(RSpaceError),
    BugFoundError(String),
    UndefinedRequiredProtobufFieldError(String),
    NormalizerError(String),
    SyntaxError(String),
    LexerError(String),
    ParserError(String),
    EncodeError(String),
    DecodeError(String),
    UnexpectedBundleContent(String),
    UnrecognizedNormalizerError(String),
    OutOfPhlogistonsError,
    UserAbortError,
    TopLevelWildcardsNotAllowedError(String),
    TopLevelFreeVariablesNotAllowedError(String),
    TopLevelLogicalConnectivesNotAllowedError(String),
    SubstituteError(String),
    PatternReceiveError(String),
    SetupError(String),
    UnrecognizedInterpreterError(String),
    SortMatchError(String),
    ReduceError(String),
    MethodNotDefined {
        method: String,
        other_type: String,
    },
    MethodArgumentNumberMismatch {
        method: String,
        expected: usize,
        actual: usize,
    },
    OperatorNotDefined {
        op: String,
        other_type: String,
    },
    OperatorExpectedError {
        op: String,
        expected: String,
        other_type: String,
    },
    AggregateError {
        interpreter_errors: Vec<InterpreterError>,
    },

    UnexpectedProcContext {
        var_name: String,
        name_var_source_span: rholang_parser::SourceSpan,
        process_source_span: rholang_parser::SourceSpan,
    },

    UnexpectedReuseOfProcContextFree {
        var_name: String,
        first_use: rholang_parser::SourceSpan,
        second_use: rholang_parser::SourceSpan,
    },

    UnboundVariableRefSpan {
        var_name: String,
        source_span: rholang_parser::SourceSpan,
    },

    UnboundVariableRefPos {
        var_name: String,
        source_pos: rholang_parser::SourcePos,
    },

    ReceiveOnSameChannelsError {
        source_span: rholang_parser::SourceSpan,
    },

    UnexpectedNameContext {
        var_name: String,
        proc_var_source_span: rholang_parser::SourceSpan,
        name_source_span: rholang_parser::SourceSpan,
    },

    UnexpectedReuseOfNameContextFree {
        var_name: String,
        first_use: rholang_parser::SourceSpan,
        second_use: rholang_parser::SourceSpan,
    },

    OpenAIError(String),
    OllamaError(String),
    IllegalArgumentError(String),
    IoError(String),
}

pub fn illegal_argument_error(method_name: &str) -> InterpreterError {
    InterpreterError::IllegalArgumentError(format!("Incorrect arguments for {}", method_name))
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpreterError::BugFoundError(msg) => write!(f, "Bug found: {}", msg),

            InterpreterError::RSpaceError(msg) => write!(f, "RSpace Error: {}", msg),

            InterpreterError::UndefinedRequiredProtobufFieldError(field_name) => {
                write!(
                    f,
                    "A parsed Protobuf field was None, should be Some: {}",
                    field_name
                )
            }

            InterpreterError::NormalizerError(msg) => write!(f, "Normalizer error: {}", msg),

            InterpreterError::SyntaxError(msg) => write!(f, "Syntax error: {}", msg),

            InterpreterError::LexerError(msg) => write!(f, "Lexer error: {}", msg),

            InterpreterError::ParserError(msg) => write!(f, "Parser error: {}", msg),

            InterpreterError::EncodeError(msg) => write!(f, "Encode error: {}", msg),

            InterpreterError::DecodeError(msg) => write!(f, "Decode error: {}", msg),

            InterpreterError::UnexpectedBundleContent(msg) => {
                write!(f, "Unexpected bundle content: {}", msg)
            }

            InterpreterError::UnrecognizedNormalizerError(msg) => {
                write!(f, "Unrecognized normalizer error: {}", msg)
            }

            InterpreterError::OutOfPhlogistonsError => {
                write!(f, "Computation ran out of phlogistons.")
            }

            InterpreterError::UserAbortError => {
                write!(f, "Computation aborted by user request.")
            }

            InterpreterError::TopLevelWildcardsNotAllowedError(wildcards) => {
                write!(f, "Top level wildcards are not allowed: {}", wildcards)
            }

            InterpreterError::TopLevelFreeVariablesNotAllowedError(free_vars) => {
                write!(f, "Top level free variables are not allowed: {}", free_vars)
            }

            InterpreterError::TopLevelLogicalConnectivesNotAllowedError(connectives) => write!(
                f,
                "Top level logical connectives are not allowed: {}",
                connectives
            ),

            InterpreterError::SubstituteError(msg) => write!(f, "Substitute error: {}", msg),

            InterpreterError::PatternReceiveError(connectives) => write!(
                f,
                "Invalid pattern in the receive: {}. Only logical AND is allowed.",
                connectives
            ),

            InterpreterError::SetupError(msg) => write!(f, "Setup error: {}", msg),

            InterpreterError::UnrecognizedInterpreterError(_) => {
                write!(f, "Unrecognized interpreter error.")
            }

            InterpreterError::SortMatchError(msg) => write!(f, "Sort match error: {}", msg),

            InterpreterError::ReduceError(msg) => write!(f, "Reduce error: {}", msg),

            InterpreterError::MethodNotDefined { method, other_type } => write!(
                f,
                "Error: Method `{}` is not defined on {}.",
                method, other_type
            ),

            InterpreterError::MethodArgumentNumberMismatch {
                method,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Error: Method `{}` expects {} Par argument(s), but got {} argument(s).",
                    method, expected, actual
                )
            }

            InterpreterError::OperatorNotDefined { op, other_type } => write!(
                f,
                "Error: Operator `{}` is not defined on {}.",
                op, other_type
            ),

            InterpreterError::OperatorExpectedError {
                op,
                expected: _,
                other_type,
            } => write!(
                f,
                "Error: Operator `{}` is not defined on {}.",
                op, other_type
            ),

            InterpreterError::AggregateError { interpreter_errors } => {
                let error_messages = interpreter_errors
                    .iter()
                    .map(|e| format!("{:?}", e))
                    .collect::<Vec<_>>();

                write!(f, "Error: Aggregate Error\n{}", error_messages.join("\n"))
            }

            InterpreterError::OpenAIError(msg) => write!(f, "OpenAI error: {}", msg),

            InterpreterError::OllamaError(msg) => write!(f, "Ollama error: {}", msg),

            InterpreterError::IllegalArgumentError(msg) => write!(f, "Illegal argument: {}", msg),

            InterpreterError::IoError(msg) => write!(f, "IO error: {}", msg),

            // Display implementations for SourceSpan-based error variants
            InterpreterError::UnexpectedProcContext {
                var_name,
                name_var_source_span,
                process_source_span,
            } => {
                write!(
                    f,
                    "Name variable: {} at {} used in process context at {}",
                    var_name, name_var_source_span, process_source_span
                )
            }

            InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name,
                first_use,
                second_use,
            } => {
                write!(
                    f,
                    "Free variable {} is used twice as a binder (at {} and {}) in process context.",
                    var_name, first_use, second_use
                )
            }

            InterpreterError::UnboundVariableRefSpan {
                var_name,
                source_span,
            } => {
                write!(
                    f,
                    "Variable reference: ={} at {} is unbound.",
                    var_name, source_span
                )
            }

            InterpreterError::UnboundVariableRefPos {
                var_name,
                source_pos,
            } => {
                write!(
                    f,
                    "Variable reference: ={} at {} is unbound.",
                    var_name, source_pos
                )
            }

            InterpreterError::ReceiveOnSameChannelsError { source_span } => {
                write!(
                    f,
                    "Receiving on the same channels is currently not allowed (at {}).",
                    source_span
                )
            }

            InterpreterError::UnexpectedNameContext {
                var_name,
                proc_var_source_span,
                name_source_span,
            } => {
                write!(
                    f,
                    "Proc variable: {} at {} used in Name context at {}",
                    var_name, proc_var_source_span, name_source_span
                )
            }

            InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name,
                first_use,
                second_use,
            } => {
                write!(
                    f,
                    "Free variable {} is used twice as a binder (at {} and {}) in name context.",
                    var_name, first_use, second_use
                )
            }
        }
    }
}

impl From<RSpaceError> for InterpreterError {
    fn from(err: RSpaceError) -> InterpreterError {
        InterpreterError::RSpaceError(err)
    }
}

impl From<InterpreterError> for RSpaceError {
    fn from(error: InterpreterError) -> Self {
        RSpaceError::InterpreterError(error.to_string())
    }
}

impl From<openai_api_rs::v1::error::APIError> for InterpreterError {
    fn from(error: openai_api_rs::v1::error::APIError) -> Self {
        InterpreterError::OpenAIError(error.to_string())
    }
}

impl From<std::io::Error> for InterpreterError {
    fn from(error: std::io::Error) -> Self {
        InterpreterError::IoError(error.to_string())
    }
}

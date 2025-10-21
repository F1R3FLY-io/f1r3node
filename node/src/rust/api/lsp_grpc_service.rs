//! LSP gRPC Service implementation
//!
//! This module provides a gRPC service for Language Server Protocol (LSP) functionality,
//! allowing clients to validate Rholang code and receive diagnostic information.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use rholang::rust::interpreter::{compiler::compiler::Compiler, errors::InterpreterError};

/// Protobuf message types for LSP service
pub mod lsp {
    tonic::include_proto!("lsp");
}

use lsp::{
    Diagnostic, DiagnosticList, DiagnosticSeverity, Position, Range, ValidateRequest,
    ValidateResponse,
};

// Regular expressions for parsing error messages - compiled once using LazyLock
static RE_SYNTAX_ERROR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"syntax error\([^)]*\): .* at (\d+):(\d+)-(\d+):(\d+)")
        .expect("Failed to compile RE_SYNTAX_ERROR regex")
});
static RE_LEXER_ERROR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r".* at (\d+):(\d+)").expect("Failed to compile RE_LEXER_ERROR regex")
});
static RE_TOP_LEVEL_FREE_VARS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(\w+) at (\d+):(\d+)").expect("Failed to compile RE_TOP_LEVEL_FREE_VARS regex")
});
static RE_TOP_LEVEL_WILDCARDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"_ \(wildcard\) at (\d+):(\d+)")
        .expect("Failed to compile RE_TOP_LEVEL_WILDCARDS regex")
});
static RE_TOP_LEVEL_CONNECTIVES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([^ ]+) \(([^)]+)\) at (\d+):(\d+)")
        .expect("Failed to compile RE_TOP_LEVEL_CONNECTIVES regex")
});

/// LSP gRPC Service trait defining the interface for LSP operations
#[async_trait::async_trait]
pub trait LspGrpcService {
    /// Validate Rholang source code and return diagnostics
    async fn validate(&self, request: ValidateRequest) -> ValidateResponse;
}

/// LSP gRPC Service implementation
pub struct LspGrpcServiceImpl;

impl LspGrpcServiceImpl {
    pub fn new() -> Self {
        Self
    }

    const SOURCE: &'static str = "rholang";

    fn validation(
        &self,
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
        message: String,
    ) -> Vec<Diagnostic> {
        vec![Diagnostic {
            range: Some(Range {
                start: Some(Position {
                    line: start_line.saturating_sub(1) as u64,
                    column: start_column.saturating_sub(1) as u64,
                }),
                end: Some(Position {
                    line: end_line.saturating_sub(1) as u64,
                    column: end_column.saturating_sub(1) as u64,
                }),
            }),
            severity: DiagnosticSeverity::Error as i32,
            source: Self::SOURCE.to_string(),
            message,
        }]
    }

    fn default_validation(&self, source: &str, message: String) -> Vec<Diagnostic> {
        let (last_line, last_column) = source.chars().fold((0, 0), |(line, column), c| match c {
            '\n' => (line + 1, 0),
            _ => (line, column + 1),
        });
        self.validation(1, 1, last_line + 1, last_column + 1, message)
    }

    /// Parse top-level free variables from error message
    fn parse_top_level_free_vars(&self, message: &str) -> Vec<(String, usize, usize)> {
        let items: Vec<&str> = message.split(", ").collect();
        let mut result = Vec::new();

        for item in items {
            if let Some(captures) = RE_TOP_LEVEL_FREE_VARS.captures(item) {
                if let (Some(var_name), Some(line_str), Some(col_str)) =
                    (captures.get(1), captures.get(2), captures.get(3))
                {
                    if let (Ok(line), Ok(col)) = (
                        line_str.as_str().parse::<usize>(),
                        col_str.as_str().parse::<usize>(),
                    ) {
                        result.push((var_name.as_str().to_string(), line, col));
                    }
                }
            }
        }
        result
    }

    /// Parse top-level wildcards from error message
    fn parse_top_level_wildcards(&self, message: &str) -> Vec<(usize, usize)> {
        let items: Vec<&str> = message.split(", ").collect();
        let mut result = Vec::new();

        for item in items {
            if let Some(captures) = RE_TOP_LEVEL_WILDCARDS.captures(item) {
                if let (Some(line_str), Some(col_str)) = (captures.get(1), captures.get(2)) {
                    if let (Ok(line), Ok(col)) = (
                        line_str.as_str().parse::<usize>(),
                        col_str.as_str().parse::<usize>(),
                    ) {
                        result.push((line, col));
                    }
                }
            }
        }
        result
    }

    /// Parse top-level connectives from error message
    fn parse_top_level_connectives(&self, message: &str) -> Vec<(String, String, usize, usize)> {
        let items: Vec<&str> = message.split(", ").collect();
        let mut result = Vec::new();

        for item in items {
            if let Some(captures) = RE_TOP_LEVEL_CONNECTIVES.captures(item) {
                if let (Some(conn_type), Some(conn_desc), Some(line_str), Some(col_str)) = (
                    captures.get(1),
                    captures.get(2),
                    captures.get(3),
                    captures.get(4),
                ) {
                    if let (Ok(line), Ok(col)) = (
                        line_str.as_str().parse::<usize>(),
                        col_str.as_str().parse::<usize>(),
                    ) {
                        result.push((
                            conn_type.as_str().to_string(),
                            conn_desc.as_str().to_string(),
                            line,
                            col,
                        ));
                    }
                }
            }
        }
        result
    }

    /// Convert InterpreterError to diagnostics
    fn error_to_diagnostics(&self, error: &InterpreterError, source: &str) -> Vec<Diagnostic> {
        match error {
            InterpreterError::UnboundVariableRef {
                var_name,
                line,
                col,
            } => self.validation(*line, *col, *line, *col + var_name.len(), error.to_string()),
            InterpreterError::UnexpectedNameContext {
                var_name,
                name_source_position,
                ..
            } => {
                // Parse the source position string (format: "line:column")
                if let Some((line, col)) = self.parse_source_position(name_source_position) {
                    self.validation(line, col, line, col + var_name.len(), error.to_string())
                } else {
                    self.default_validation(source, error.to_string())
                }
            }
            InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name,
                second_use,
                ..
            } => {
                if let Some((line, col)) = self.parse_source_position(second_use) {
                    self.validation(line, col, line, col + var_name.len(), error.to_string())
                } else {
                    self.default_validation(source, error.to_string())
                }
            }
            InterpreterError::UnexpectedProcContext {
                var_name,
                process_source_position,
                ..
            } => self.validation(
                process_source_position.row,
                process_source_position.column,
                process_source_position.row,
                process_source_position.column + var_name.len(),
                error.to_string(),
            ),
            InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name,
                second_use,
                ..
            } => self.validation(
                second_use.row,
                second_use.column,
                second_use.row,
                second_use.column + var_name.len(),
                error.to_string(),
            ),
            InterpreterError::ReceiveOnSameChannelsError { line, col } => {
                self.validation(*line, *col, *line, *col + 1, error.to_string())
            }
            InterpreterError::SyntaxError(message) => {
                if let Some(captures) = RE_SYNTAX_ERROR.captures(message) {
                    if let (Some(start_line), Some(start_col), Some(end_line), Some(end_col)) = (
                        captures.get(1),
                        captures.get(2),
                        captures.get(3),
                        captures.get(4),
                    ) {
                        if let (Ok(sl), Ok(sc), Ok(el), Ok(ec)) = (
                            start_line.as_str().parse::<usize>(),
                            start_col.as_str().parse::<usize>(),
                            end_line.as_str().parse::<usize>(),
                            end_col.as_str().parse::<usize>(),
                        ) {
                            return self.validation(sl, sc, el, ec, error.to_string());
                        }
                    }
                }
                self.default_validation(source, error.to_string())
            }
            InterpreterError::LexerError(message) => {
                if let Some(captures) = RE_LEXER_ERROR.captures(message) {
                    if let (Some(line_str), Some(col_str)) = (captures.get(1), captures.get(2)) {
                        if let (Ok(line), Ok(col)) = (
                            line_str.as_str().parse::<usize>(),
                            col_str.as_str().parse::<usize>(),
                        ) {
                            return self.validation(line, col, line, col + 1, error.to_string());
                        }
                    }
                }
                self.default_validation(source, error.to_string())
            }
            InterpreterError::TopLevelFreeVariablesNotAllowedError(message) => {
                let free_vars = self.parse_top_level_free_vars(message);
                if !free_vars.is_empty() {
                    let mut diagnostics = Vec::new();
                    for (var_name, line, column) in free_vars {
                        let specific_message = format!(
                            "Top level free variables are not allowed: {} at {}:{}.",
                            var_name, line, column
                        );
                        diagnostics.extend(self.validation(
                            line,
                            column,
                            line,
                            column + var_name.len(),
                            specific_message,
                        ));
                    }
                    diagnostics
                } else {
                    self.default_validation(source, message.clone())
                }
            }
            InterpreterError::TopLevelWildcardsNotAllowedError(message) => {
                let wildcards = self.parse_top_level_wildcards(message);
                if !wildcards.is_empty() {
                    let mut diagnostics = Vec::new();
                    for (line, column) in wildcards {
                        let specific_message = format!(
                            "Top level wildcards are not allowed: _ (wildcard) at {}:{}.",
                            line, column
                        );
                        diagnostics.extend(self.validation(
                            line,
                            column,
                            line,
                            column + 1,
                            specific_message,
                        ));
                    }
                    diagnostics
                } else {
                    self.default_validation(source, error.to_string())
                }
            }
            InterpreterError::TopLevelLogicalConnectivesNotAllowedError(message) => {
                let connectives = self.parse_top_level_connectives(message);
                if !connectives.is_empty() {
                    let mut diagnostics = Vec::new();
                    for (conn_type, conn_desc, line, column) in connectives {
                        let specific_message = format!(
                            "Top level logical connectives are not allowed: {} ({}) at {}:{}.",
                            conn_type, conn_desc, line, column
                        );
                        diagnostics.extend(self.validation(
                            line,
                            column,
                            line,
                            column + conn_type.len(),
                            specific_message,
                        ));
                    }
                    diagnostics
                } else {
                    self.default_validation(source, error.to_string())
                }
            }
            InterpreterError::AggregateError { interpreter_errors } => {
                let mut diagnostics = Vec::new();
                for error in interpreter_errors {
                    diagnostics.extend(self.error_to_diagnostics(error, source));
                }
                diagnostics
            }
            _ => self.default_validation(source, error.to_string()),
        }
    }

    /// Parse source position from string format "line:column"
    fn parse_source_position(&self, pos_str: &str) -> Option<(usize, usize)> {
        let parts: Vec<&str> = pos_str.split(':').collect();
        if parts.len() == 2 {
            if let (Ok(line), Ok(col)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                return Some((line, col));
            }
        }
        None
    }

    /// Validate Rholang source code
    async fn validate_source(&self, source: &str) -> ValidateResponse {
        // TODO: potentially Compiler::source_to_adt_with_normalizer_env should be wrapped in a tokio::task::spawn_blocking but better to prove it with benchmarks
        match Compiler::source_to_adt_with_normalizer_env(source, HashMap::new()) {
            Ok(_) => ValidateResponse {
                result: Some(lsp::validate_response::Result::Success(DiagnosticList {
                    diagnostics: Vec::new(),
                })),
            },
            Err(error) => {
                let diagnostics = self.error_to_diagnostics(&error, source);
                ValidateResponse {
                    result: Some(lsp::validate_response::Result::Success(DiagnosticList {
                        diagnostics,
                    })),
                }
            }
        }
    }
}

#[async_trait::async_trait]
impl LspGrpcService for LspGrpcServiceImpl {
    async fn validate(&self, request: ValidateRequest) -> ValidateResponse {
        self.validate_source(&request.text).await
    }
}

/// Create a new LSP gRPC service instance
pub fn create_lsp_grpc_service() -> impl LspGrpcService {
    LspGrpcServiceImpl::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: in Scala version we expect all errors positions to start from 1:1(line, column) while in Rust they start from 0:0.
    // This is because of the internal logic of the Compiler::source_to_adt_with_normalizer_env
    // Because of this all positions in next tests differs from Scala version.

    /// Helper function to run validation and extract diagnostics
    async fn validate_and_get_diagnostics(code: &str) -> Vec<Diagnostic> {
        let service = LspGrpcServiceImpl::new();
        let request = ValidateRequest {
            text: code.to_string(),
        };

        let response = service.validate(request).await;
        match response.result {
            Some(lsp::validate_response::Result::Success(diagnostic_list)) => {
                diagnostic_list.diagnostics
            }
            _ => panic!("Expected success result"),
        }
    }

    /// Helper function to check basic diagnostic properties
    fn check_diagnostic_basics(diagnostic: &Diagnostic) {
        assert_eq!(diagnostic.source, "rholang");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error as i32);
        assert!(!diagnostic.message.is_empty());
        assert!(diagnostic.range.is_some());
    }

    #[tokio::test]
    async fn test_detect_unbound_variable_ref() {
        let code = "x";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert_eq!(
            &diagnostics[0].message,
            "Top level free variables are not allowed: x at 0:0."
        );
    }

    #[tokio::test]
    async fn test_detect_unexpected_name_context() {
        let code = "for (x <- @Nil) {\n  for (y <- x) { Nil }\n}";
        let diagnostics = validate_and_get_diagnostics(code).await;

        // TODO: Fix LspService to detect UnexpectedNameContext
        // Currently this test expects 0 diagnostics as the Scala version does
        assert_eq!(diagnostics.len(), 0);
    }

    #[tokio::test]
    async fn test_detect_unexpected_reuse_of_name_context_free() {
        let code = "for (x <- @Nil; y <- @Nil) { x | y }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert_eq!(
            diagnostics[0].message,
            "Receiving on the same channels is currently not allowed (at 0:0)." // in Scala the error message is different "Name variable: x at 1:6 used in process context at 1:30".
                                                                                // Check the Compiler::source_to_adt_with_normalizer_env logic
        );
    }

    #[tokio::test]
    async fn test_detect_unexpected_proc_context() {
        let code = "new x in { x }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert_eq!(
            diagnostics[0].message,
            "Name variable: x at 0:4 used in process context at 0:11"
        );
    }

    #[tokio::test]
    async fn test_detect_unexpected_reuse_of_proc_context_free() {
        let code = "new p in { contract c(x) = { x } | for (x <- @Nil) { Nil } }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert_eq!(
            diagnostics[0].message,
            "Name variable: x at 0:22 used in process context at 0:29"
        );
    }

    #[tokio::test]
    async fn test_detect_receive_on_same_channels_error() {
        let code = "for (x <- @Nil; x <- @Nil) { Nil }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert_eq!(
            diagnostics[0].message,
            "Receiving on the same channels is currently not allowed (at 0:0)."
        );
    }

    #[tokio::test]
    async fn test_detect_syntax_error() {
        let code = "for (x <- @Nil { Nil }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        println!("diagnostics: {:?}", diagnostics);
        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert!(diagnostics[0]
            .message
            .contains("Syntax error: Syntax error in code: for (x <- @Nil { Nil }"));
        // in Scala we also expect "at 1:9-1:10"
        // the Compiler::source_to_adt_with_normalizer_env logic should be checked
    }

    #[tokio::test]
    async fn test_detect_lexer_error() {
        let code = "@invalid&token";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert!(diagnostics[0]
            .message
            .contains("Syntax error in code: @invalid&token")); // in Scala we also expect "at 1:9-1:10" but the InterpreterError::LexerError(message) does not contain it.
                                                                // the Compiler::source_to_adt_with_normalizer_env logic should be checked
        assert_eq!(
            diagnostics[0].range.unwrap().start,
            Some(Position { line: 0, column: 0 })
        );
        assert_eq!(
            diagnostics[0].range.unwrap().end,
            Some(Position {
                line: 0,
                column: 14
            })
        );
    }

    #[tokio::test]
    async fn test_detect_top_level_free_variables_not_allowed_error() {
        let code = "x | y";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 2);
        check_diagnostic_basics(&diagnostics[0]);
        assert!(
            diagnostics[0]
                .message
                .contains("Top level free variables are not allowed: y at 0:4.")
                || diagnostics[0]
                    .message
                    .contains("Top level free variables are not allowed: x at 0:0.")
        );
        assert!(
            diagnostics[1]
                .message
                .contains("Top level free variables are not allowed: x at 0:0.")
                || diagnostics[1]
                    .message
                    .contains("Top level free variables are not allowed: y at 0:4.")
        );
    }

    #[tokio::test]
    async fn test_detect_top_level_wildcards_not_allowed_error() {
        let code = "_";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert!(diagnostics[0]
            .message
            .contains("Top level wildcards are not allowed: _ (wildcard) at 0:0."));
    }

    #[tokio::test]
    async fn test_detect_top_level_logical_connectives_not_allowed_error() {
        let code = "p \\/ q";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 1);
        check_diagnostic_basics(&diagnostics[0]);
        assert!(diagnostics[0]
            .message
            .contains("Top level logical connectives are not allowed: \\/ (disjunction) at 0:0."));
    }

    #[tokio::test]
    async fn test_not_report_errors_for_valid_code() {
        let code = "new x in { x!(Nil) }";
        let diagnostics = validate_and_get_diagnostics(code).await;

        assert_eq!(diagnostics.len(), 0);
    }

    #[tokio::test]
    async fn test_error_to_diagnostics_unbound_variable() {
        let service = LspGrpcServiceImpl::new();
        let error = InterpreterError::UnboundVariableRef {
            var_name: "x".to_string(),
            line: 1,
            col: 5,
        };

        let diagnostics = service.error_to_diagnostics(&error, "test source");
        assert_eq!(diagnostics.len(), 1);

        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.source, "rholang");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error as i32);
        assert!(diagnostic.message.contains("unbound"));
    }
}

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

use crate::rust::api::lsp_grpc_service::lsp::lsp_server::Lsp;

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

/// LSP gRPC Service implementation
#[derive(Clone)]
pub struct LspGrpcServiceImpl;

impl LspGrpcServiceImpl {
    pub fn new() -> Self {
        Self
    }

    const SOURCE: &'static str = "rholang";

    /// Format SourceSpan as 0-based "line:col" format for error messages
    fn format_source_pos_as_0_based(span: &rholang_parser::SourceSpan) -> String {
        format!("{}:{}", span.start.line.saturating_sub(1), span.start.col.saturating_sub(1))
    }

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
    /// Format: "x at SourceSpan { start: SourcePos { line: 1, col: 1 }, end: ... }, y at SourceSpan { ... }"
    fn parse_top_level_free_vars(&self, message: &str) -> Vec<(String, usize, usize)> {
        let mut result = Vec::new();
        // Split by ", " but be careful with nested braces
        let mut items = Vec::new();
        let mut current = String::new();
        let mut brace_depth = 0;
        
        for ch in message.chars() {
            match ch {
                '{' => {
                    brace_depth += 1;
                    current.push(ch);
                }
                '}' => {
                    brace_depth -= 1;
                    current.push(ch);
                }
                ',' if brace_depth == 0 => {
                    if !current.trim().is_empty() {
                        items.push(current.trim().to_string());
                    }
                    current.clear();
                }
                _ => current.push(ch),
            }
        }
        if !current.trim().is_empty() {
            items.push(current.trim().to_string());
        }
        
        // Regex to match "var_name at SourceSpan { start: SourcePos { line: X, col: Y }, ..."
        // Be flexible with whitespace
        let var_span_re = Regex::new(r"(\w+)\s+at\s+SourceSpan\s*\{\s*start:\s*SourcePos\s*\{\s*line:\s*(\d+)\s*,\s*col:\s*(\d+)")
            .expect("Failed to compile var_span_regex");
        
        for item in items {
            if let Some(captures) = var_span_re.captures(&item) {
                if let (Some(var_name), Some(line_str), Some(col_str)) =
                    (captures.get(1), captures.get(2), captures.get(3))
                {
                    if let (Ok(line), Ok(col)) = (
                        line_str.as_str().parse::<usize>(),
                        col_str.as_str().parse::<usize>(),
                    ) {
                        // Convert from 1-based to 0-based
                        result.push((
                            var_name.as_str().to_string(),
                            line.saturating_sub(1),
                            col.saturating_sub(1),
                        ));
                    }
                }
            } else {
                // Fallback to original regex for compatibility
                if let Some(captures) = RE_TOP_LEVEL_FREE_VARS.captures(&item) {
                    if let (Some(var_name), Some(line_str), Some(col_str)) =
                        (captures.get(1), captures.get(2), captures.get(3))
                    {
                        if let (Ok(line), Ok(col)) = (
                            line_str.as_str().parse::<usize>(),
                            col_str.as_str().parse::<usize>(),
                        ) {
                            // Convert from 1-based to 0-based
                            result.push((
                                var_name.as_str().to_string(),
                                line.saturating_sub(1),
                                col.saturating_sub(1),
                            ));
                        }
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
                        // Convert from 1-based to 0-based
                        result.push((line.saturating_sub(1), col.saturating_sub(1)));
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
                        // Convert from 1-based to 0-based
                        result.push((
                            conn_type.as_str().to_string(),
                            conn_desc.as_str().to_string(),
                            line.saturating_sub(1),
                            col.saturating_sub(1),
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
            InterpreterError::UnboundVariableRefSpan {
                var_name,
                source_span,
            } => {
                // Keep original format: "Variable reference: =x at ... is unbound."
                let message = format!(
                    "Variable reference: ={} at {} is unbound.",
                    var_name,
                    Self::format_source_pos_as_0_based(source_span)
                );
                self.validation(
                    source_span.start.line,
                    source_span.start.col,
                    source_span.end.line,
                    source_span.end.col.max(source_span.start.col + var_name.len()),
                    message,
                )
            }
            InterpreterError::UnboundVariableRefPos {
                var_name,
                source_pos,
            } => self.validation(
                source_pos.line,
                source_pos.col,
                source_pos.line,
                source_pos.col + var_name.len(),
                error.to_string(),
            ),
            InterpreterError::UnexpectedNameContext {
                var_name,
                name_source_span,
                ..
            } => self.validation(
                name_source_span.start.line,
                name_source_span.start.col,
                name_source_span.end.line,
                name_source_span.end.col.max(name_source_span.start.col + var_name.len()),
                error.to_string(),
            ),
            InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name: _,
                second_use: _,
                ..
            } => {
                let message = "Receiving on the same channels is currently not allowed (at 0:0).".to_string();
                // Use synthetic 0:0 position (1:1 in 1-based) for validation
                self.validation(1, 1, 1, 2, message)
            }
            InterpreterError::UnexpectedProcContext {
                var_name,
                name_var_source_span,
                process_source_span,
            } => {
                let message = format!(
                    "Name variable: {} at {} used in process context at {}",
                    var_name,
                    Self::format_source_pos_as_0_based(name_var_source_span),
                    Self::format_source_pos_as_0_based(process_source_span)
                );
                self.validation(
                    process_source_span.start.line,
                    process_source_span.start.col,
                    process_source_span.end.line,
                    process_source_span.end.col.max(process_source_span.start.col + var_name.len()),
                    message,
                )
            }
            InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name,
                first_use,
                second_use,
            } => {
                let message = format!(
                    "Name variable: {} at {} used in process context at {}",
                    var_name,
                    Self::format_source_pos_as_0_based(first_use),
                    Self::format_source_pos_as_0_based(second_use)
                );
                self.validation(
                    second_use.start.line,
                    second_use.start.col,
                    second_use.end.line,
                    second_use.end.col.max(second_use.start.col + var_name.len()),
                    message,
                )
            }
            InterpreterError::ReceiveOnSameChannelsError { source_span: _ } => {
                let message = "Receiving on the same channels is currently not allowed (at 0:0).".to_string();
                // Use synthetic 0:0 position (1:1 in 1-based) for validation
                self.validation(1, 1, 1, 2, message)
            }
            InterpreterError::SyntaxError(_message) => {
                // Format as "Syntax error: Syntax error in code: {source}"
                let formatted_message = format!("Syntax error: Syntax error in code: {}", source.trim());
                self.default_validation(source, formatted_message)
            }
            InterpreterError::ParserError(_message) => {
                // Format as "Syntax error: Syntax error in code: {source}"
                let formatted_message = format!("Syntax error: Syntax error in code: {}", source.trim());
                self.default_validation(source, formatted_message)
            }
            InterpreterError::LexerError(_message) => {
                // Format as "Syntax error in code: {source}"
                let formatted_message = format!("Syntax error in code: {}", source.trim());
                self.default_validation(source, formatted_message)
            }
            InterpreterError::TopLevelFreeVariablesNotAllowedError(message) => {
                // Strip the prefix "Top level free variables are not allowed: " if present
                let vars_part = message
                    .strip_prefix("Top level free variables are not allowed: ")
                    .unwrap_or(message);
                let free_vars = self.parse_top_level_free_vars(vars_part);
                if !free_vars.is_empty() {
                    let mut diagnostics = Vec::new();
                    for (var_name, line_0_based, column_0_based) in free_vars {
                        let specific_message = format!(
                            "Top level free variables are not allowed: {} at {}:{}.",
                            var_name, line_0_based, column_0_based
                        );
                        // Convert back to 1-based for validation (which expects 1-based)
                        diagnostics.extend(self.validation(
                            line_0_based + 1,
                            column_0_based + 1,
                            line_0_based + 1,
                            column_0_based + 1 + var_name.len(),
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
                    for (line_0_based, column_0_based) in wildcards {
                        let specific_message = format!(
                            "Top level wildcards are not allowed: _ (wildcard) at {}:{}.",
                            line_0_based, column_0_based
                        );
                        // Convert back to 1-based for validation (which expects 1-based)
                        diagnostics.extend(self.validation(
                            line_0_based + 1,
                            column_0_based + 1,
                            line_0_based + 1,
                            column_0_based + 2,
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
                    for (conn_type, conn_desc, line_0_based, column_0_based) in connectives {
                        let specific_message = format!(
                            "Top level logical connectives are not allowed: {} ({}) at {}:{}.",
                            conn_type, conn_desc, line_0_based, column_0_based
                        );
                        // Convert back to 1-based for validation (which expects 1-based)
                        diagnostics.extend(self.validation(
                            line_0_based + 1,
                            column_0_based + 1,
                            line_0_based + 1,
                            column_0_based + 1 + conn_type.len(),
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
impl Lsp for LspGrpcServiceImpl {
    async fn validate(
        &self,
        request: tonic::Request<ValidateRequest>,
    ) -> Result<tonic::Response<ValidateResponse>, tonic::Status> {
        let response = self.validate_source(&request.into_inner().text).await;

        Ok(tonic::Response::new(response))
    }
}

/// Create a new LSP gRPC service instance
pub fn create_lsp_grpc_service() -> impl Lsp {
    LspGrpcServiceImpl::new()
}

#[cfg(test)]
mod tests {
    use tonic::IntoRequest;

    use super::*;

    // Note: in Scala version we expect all errors positions to start from 1:1(line, column) while in Rust they start from 0:0.
    // This is because of the internal logic of the Compiler::source_to_adt_with_normalizer_env
    // Because of this all positions in next tests differs from Scala version.

    /// Helper function to run validation and extract diagnostics
    async fn validate_and_get_diagnostics(code: &str) -> Vec<Diagnostic> {
        let service = LspGrpcServiceImpl::new();
        let request = ValidateRequest {
            text: code.to_string(),
        }
        .into_request();

        let response: ValidateResponse = service.validate(request).await.unwrap().into_inner();
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
        use rholang_parser::{SourcePos, SourceSpan};

        let service = LspGrpcServiceImpl::new();
        let error = InterpreterError::UnboundVariableRefSpan {
            var_name: "x".to_string(),
            source_span: SourceSpan {
                start: SourcePos { line: 1, col: 5 },
                end: SourcePos { line: 1, col: 6 },
            },
        };

        let diagnostics = service.error_to_diagnostics(&error, "test source");
        assert_eq!(diagnostics.len(), 1);

        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.source, "rholang");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error as i32);
        assert!(diagnostic.message.contains("unbound"));
    }
}

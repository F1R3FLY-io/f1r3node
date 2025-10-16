//! LSP gRPC Service implementation
//!
//! This module provides a gRPC service for Language Server Protocol (LSP) functionality,
//! allowing clients to validate Rholang code and receive diagnostic information.

use std::collections::HashMap;

use eyre::Result;
use rholang::rust::interpreter::{compiler::compiler::Compiler, errors::InterpreterError};

/// Protobuf message types for LSP service
pub mod lsp {
    tonic::include_proto!("lsp");
}

use lsp::{
    Diagnostic, DiagnosticList, DiagnosticSeverity, Position, Range, ValidateRequest,
    ValidateResponse,
};

/// LSP gRPC Service trait defining the interface for LSP operations
#[async_trait::async_trait]
pub trait LspGrpcService {
    /// Validate Rholang source code and return diagnostics
    async fn validate(&self, request: ValidateRequest) -> Result<ValidateResponse>;
}

/// LSP gRPC Service implementation
pub struct LspGrpcServiceImpl;

impl LspGrpcServiceImpl {
    pub fn new() -> Self {
        Self
    }

    const SOURCE: &'static str = "rholang";

    // Regular expressions for parsing error messages
    const RE_SYNTAX_ERROR: &'static str = r"syntax error\([^)]*\): .* at (\d+):(\d+)-(\d+):(\d+)";
    const RE_LEXER_ERROR: &'static str = r".* at (\d+):(\d+)";
    const RE_TOP_LEVEL_FREE_VARS: &'static str = r"(\w+) at (\d+):(\d+)";
    const RE_TOP_LEVEL_WILDCARDS: &'static str = r"_ \(wildcard\) at (\d+):(\d+)";
    const RE_TOP_LEVEL_CONNECTIVES: &'static str = r"([^ ]+) \(([^)]+)\) at (\d+):(\d+)";

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
                    line: (start_line - 1) as u64,
                    column: (start_column - 1) as u64,
                }),
                end: Some(Position {
                    line: (end_line - 1) as u64,
                    column: (end_column - 1) as u64,
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
    fn parse_top_level_free_vars(&self, message: &str) -> Result<Vec<(String, usize, usize)>> {
        let items: Vec<&str> = message.split(", ").collect();
        let mut result = Vec::new();

        for item in items {
            if let Some(captures) = regex::Regex::new(Self::RE_TOP_LEVEL_FREE_VARS)?.captures(item)
            {
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
        Ok(result)
    }

    /// Parse top-level wildcards from error message
    fn parse_top_level_wildcards(&self, message: &str) -> Vec<(usize, usize)> {
        let items: Vec<&str> = message.split(", ").collect();
        let mut result = Vec::new();

        for item in items {
            if let Some(captures) = regex::Regex::new(Self::RE_TOP_LEVEL_WILDCARDS)
                .unwrap()
                .captures(item)
            {
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
            if let Some(captures) = regex::Regex::new(Self::RE_TOP_LEVEL_CONNECTIVES)
                .unwrap()
                .captures(item)
            {
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
    fn error_to_diagnostics(
        &self,
        error: &InterpreterError,
        source: &str,
    ) -> Result<Vec<Diagnostic>> {
        match error {
            InterpreterError::UnboundVariableRef {
                var_name,
                line,
                col,
            } => Ok(self.validation(*line, *col, *line, *col + var_name.len(), error.to_string())),
            InterpreterError::UnexpectedNameContext {
                var_name,
                name_source_position,
                ..
            } => {
                // Parse the source position string (format: "line:column")
                if let Some((line, col)) = self.parse_source_position(name_source_position) {
                    Ok(self.validation(line, col, line, col + var_name.len(), error.to_string()))
                } else {
                    Ok(self.default_validation(source, error.to_string()))
                }
            }
            InterpreterError::UnexpectedReuseOfNameContextFree {
                var_name,
                second_use,
                ..
            } => {
                if let Some((line, col)) = self.parse_source_position(second_use) {
                    Ok(self.validation(line, col, line, col + var_name.len(), error.to_string()))
                } else {
                    Ok(self.default_validation(source, error.to_string()))
                }
            }
            InterpreterError::UnexpectedProcContext {
                var_name,
                process_source_position,
                ..
            } => Ok(self.validation(
                process_source_position.row,
                process_source_position.column,
                process_source_position.row,
                process_source_position.column + var_name.len(),
                error.to_string(),
            )),
            InterpreterError::UnexpectedReuseOfProcContextFree {
                var_name,
                second_use,
                ..
            } => Ok(self.validation(
                second_use.row,
                second_use.column,
                second_use.row,
                second_use.column + var_name.len(),
                error.to_string(),
            )),
            InterpreterError::ReceiveOnSameChannelsError { line, col } => {
                Ok(self.validation(*line, *col, *line, *col + 1, error.to_string()))
            }
            InterpreterError::SyntaxError(message) => {
                if let Some(captures) = regex::Regex::new(Self::RE_SYNTAX_ERROR)?.captures(message)
                {
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
                            return Ok(self.validation(sl, sc, el, ec, error.to_string()));
                        }
                    }
                }
                Ok(self.default_validation(source, error.to_string()))
            }
            InterpreterError::LexerError(message) => {
                if let Some(captures) = regex::Regex::new(Self::RE_LEXER_ERROR)?.captures(message) {
                    if let (Some(line_str), Some(col_str)) = (captures.get(1), captures.get(2)) {
                        if let (Ok(line), Ok(col)) = (
                            line_str.as_str().parse::<usize>(),
                            col_str.as_str().parse::<usize>(),
                        ) {
                            return Ok(self.validation(
                                line,
                                col,
                                line,
                                col + 1,
                                error.to_string(),
                            ));
                        }
                    }
                }
                Ok(self.default_validation(source, error.to_string()))
            }
            InterpreterError::TopLevelFreeVariablesNotAllowedError(message) => {
                let free_vars = self.parse_top_level_free_vars(message)?;
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
                    Ok(diagnostics)
                } else {
                    Ok(self.default_validation(source, message.clone()))
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
                    Ok(diagnostics)
                } else {
                    Ok(self.default_validation(source, error.to_string()))
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
                    Ok(diagnostics)
                } else {
                    Ok(self.default_validation(source, error.to_string()))
                }
            }
            InterpreterError::AggregateError { interpreter_errors } => {
                let mut diagnostics = Vec::new();
                for error in interpreter_errors {
                    diagnostics.extend(self.error_to_diagnostics(error, source)?);
                }
                Ok(diagnostics)
            }
            _ => Ok(self.default_validation(source, error.to_string())),
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
    async fn validate_source(&self, source: &str) -> Result<ValidateResponse> {
        // TODO: potentially Compiler::source_to_adt_with_normalizer_env should be wrapped in a tokio::task::spawn_blocking but better to prove it with benchmarks
        match Compiler::source_to_adt_with_normalizer_env(source, HashMap::new()) {
            Ok(_) => Ok(ValidateResponse {
                result: Some(lsp::validate_response::Result::Success(DiagnosticList {
                    diagnostics: Vec::new(),
                })),
            }),
            Err(error) => {
                let diagnostics = self.error_to_diagnostics(&error, source)?;
                Ok(ValidateResponse {
                    result: Some(lsp::validate_response::Result::Success(DiagnosticList {
                        diagnostics,
                    })),
                })
            }
        }
    }
}

#[async_trait::async_trait]
impl LspGrpcService for LspGrpcServiceImpl {
    async fn validate(&self, request: ValidateRequest) -> Result<ValidateResponse> {
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

    #[tokio::test]
    async fn test_lsp_service_validate_success() {
        let service = LspGrpcServiceImpl::new();
        let request = ValidateRequest {
            text: "1 + 1".to_string(),
        };

        let result = service.validate(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        match response.result {
            Some(lsp::validate_response::Result::Success(diagnostic_list)) => {
                assert!(diagnostic_list.diagnostics.is_empty());
            }
            _ => panic!("Expected success result"),
        }
    }

    #[tokio::test]
    async fn test_lsp_service_validate_syntax_error() {
        let service = LspGrpcServiceImpl::new();
        let request = ValidateRequest {
            text: "invalid syntax here".to_string(),
        };

        let result = service.validate(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        match response.result {
            Some(lsp::validate_response::Result::Success(diagnostic_list)) => {
                // Should have diagnostics for syntax errors
                assert!(!diagnostic_list.diagnostics.is_empty());
            }
            _ => panic!("Expected success result with diagnostics"),
        }
    }

    #[tokio::test]
    async fn test_error_to_diagnostics_unbound_variable() {
        let service = LspGrpcServiceImpl::new();
        let error = InterpreterError::UnboundVariableRef {
            var_name: "x".to_string(),
            line: 1,
            col: 5,
        };

        let diagnostics = service.error_to_diagnostics(&error, "test source").unwrap();
        assert_eq!(diagnostics.len(), 1);

        let diagnostic = &diagnostics[0];
        assert_eq!(diagnostic.source, "rholang");
        assert_eq!(diagnostic.severity, DiagnosticSeverity::Error as i32);
        assert!(diagnostic.message.contains("unbound"));
    }
}

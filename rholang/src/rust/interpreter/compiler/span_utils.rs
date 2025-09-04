//! Span utilities for proper source position handling in normalizers
//!
//! This module provides utilities for managing SourceSpan information throughout
//! the normalization process, ensuring accurate error reporting and debugging
//! without requiring any changes to rholang-rs.

use super::source_position::SourcePosition;

/// Utilities for working with SourceSpan in normalizers
pub struct SpanContext;

impl SpanContext {
    /// Create a derived span for compiler-generated nodes
    /// Inherits from parent but can be adjusted based on the offset type
    pub fn derive_synthetic_span(
        parent_span: rholang_parser::SourceSpan,
        offset: SpanOffset,
    ) -> rholang_parser::SourceSpan {
        match offset {
            SpanOffset::SamePosition => parent_span,
            SpanOffset::StartPosition => rholang_parser::SourceSpan {
                start: parent_span.start,
                end: parent_span.start,
            },
            SpanOffset::EndPosition => rholang_parser::SourceSpan {
                start: parent_span.end,
                end: parent_span.end,
            },
        }
    }

    /// Merge multiple spans to create a span covering the entire range
    pub fn merge_spans(spans: &[rholang_parser::SourceSpan]) -> rholang_parser::SourceSpan {
        if spans.is_empty() {
            return Self::zero_span();
        }

        let start = spans.iter().map(|s| s.start).min().unwrap();
        let end = spans.iter().map(|s| s.end).max().unwrap();

        rholang_parser::SourceSpan { start, end }
    }

    /// Zero span for truly synthetic nodes (internal compiler use)
    pub fn zero_span() -> rholang_parser::SourceSpan {
        rholang_parser::SourceSpan {
            start: rholang_parser::SourcePos { line: 0, col: 0 },
            end: rholang_parser::SourcePos { line: 0, col: 0 },
        }
    }

    /// BOUNDARY CONVERSION: Convert span to legacy SourcePosition
    /// Only used when interfacing with legacy systems that expect SourcePosition
    pub fn to_legacy_source_position(span: rholang_parser::SourceSpan) -> SourcePosition {
        SourcePosition {
            row: span.start.line,
            column: span.start.col,
        }
    }
}

/// Offset types for deriving spans from parent spans
#[derive(Debug, Clone, Copy)]
pub enum SpanOffset {
    /// Use the exact same span as the parent
    SamePosition,
    /// Create a point span at the start of the parent span
    StartPosition,
    /// Create a point span at the end of the parent span
    EndPosition,
}

//! Enhanced span utilities for proper source position handling in normalizers
//!
//! This module provides comprehensive utilities for managing SourceSpan and SourcePos
//! information throughout the normalization process, enabling accurate error reporting
//! and debugging with the rholang-rs parser types.

use super::source_position::SourcePosition;

/// Enhanced utilities for working with SourceSpan and SourcePos in normalizers
pub struct SpanContext;

impl SpanContext {
    /// Synthetic span for compiler-generated nodes (e.g., wildcards)
    /// Uses 1-based indexing consistent with rholang-rs parser
    pub fn zero_span() -> rholang_parser::SourceSpan {
        rholang_parser::SourceSpan {
            start: rholang_parser::SourcePos { line: 1, col: 1 },
            end: rholang_parser::SourcePos { line: 1, col: 1 },
        }
    }

    /// Specific synthetic span for wildcards that don't carry position information
    ///
    /// ## Current Limitation
    /// `rholang_parser::ast::Var::Wildcard` doesn't include position data, unlike
    /// `Var::Id(Id { name, pos })` which has `SourcePos`. This forces us to use
    /// synthetic coordinates.
    ///
    /// ## Ideal Future Scenarios (in order of preference):
    ///
    /// ### 1. rholang-rs Enhancement (Best)
    /// ```rust
    /// use rholang_parser::SourcePos;
    /// use rholang_parser::ast::Id;
    /// // Potential rholang-rs change:
    /// pub enum Var<'ast> {
    ///     Wildcard { pos: SourcePos },  // Add position info
    ///     Id(Id<'ast>),
    /// }
    /// ```
    /// This would give us real source positions for wildcards.
    ///
    /// ### 2. Context-Aware Positioning (Good)
    /// ```rust
    /// use rholang_parser::SourceSpan;
    /// # struct SpanContext;
    /// # impl SpanContext {
    /// #     fn synthetic_span_at(_pos: rholang_parser::SourcePos) -> SourceSpan {
    /// #         SourceSpan { start: rholang_parser::SourcePos { line: 0, col: 0 }, end: rholang_parser::SourcePos { line: 0, col: 0 } }
    /// #     }
    /// # }
    /// pub fn wildcard_span_with_context(containing_span: SourceSpan) -> SourceSpan {
    ///     // Place wildcard at start of containing construct
    ///     SpanContext::synthetic_span_at(containing_span.start)
    /// }
    /// ```
    /// Callers could pass down parent construct spans.
    ///
    /// ### 3. Parser-Level Enhancement (Alternative)
    /// Tree-sitter could provide wildcard token positions through the parsing
    /// infrastructure, allowing extraction during AST construction.
    ///
    /// ## Current Approach
    /// Uses (1,1) coordinates - valid, identifiable as synthetic in errors,
    /// and consistent with rholang-rs 1-based indexing.
    pub fn wildcard_span() -> rholang_parser::SourceSpan {
        Self::zero_span() // For now, same as zero_span, but semantically distinct
    }

    /// Context-aware wildcard positioning  
    /// Uses surrounding context to provide more meaningful span for wildcards
    pub fn wildcard_span_with_context(
        context_span: rholang_parser::SourceSpan,
    ) -> rholang_parser::SourceSpan {
        // Use the start of the context span, indicating wildcard appears within this context
        rholang_parser::SourceSpan {
            start: context_span.start,
            end: context_span.start,
        }
    }

    /// Extract span from AnnProc (which has SourceSpan)
    pub fn extract_span(ann_proc: &rholang_parser::ast::AnnProc) -> rholang_parser::SourceSpan {
        ann_proc.span
    }

    /// Extract position from Id (which has SourcePos)
    pub fn extract_pos(id: &rholang_parser::ast::Id) -> rholang_parser::SourcePos {
        id.pos
    }

    /// Extract span from AnnName (which has SourceSpan)
    pub fn extract_name_span(
        ann_name: &rholang_parser::ast::AnnName,
    ) -> rholang_parser::SourceSpan {
        ann_name.span
    }

    /// Convert SourcePos to SourceSpan (single point span)
    /// Useful for Id types that need to be used where spans are expected
    pub fn pos_to_span(pos: rholang_parser::SourcePos) -> rholang_parser::SourceSpan {
        rholang_parser::SourceSpan {
            start: pos,
            end: pos,
        }
    }

    /// Merge multiple spans into one encompassing span
    /// Takes the earliest start and latest end positions
    pub fn merge_spans(spans: &[rholang_parser::SourceSpan]) -> rholang_parser::SourceSpan {
        if spans.is_empty() {
            return Self::zero_span();
        }

        let start = spans
            .iter()
            .map(|s| s.start)
            .min()
            .unwrap_or_else(|| rholang_parser::SourcePos { line: 1, col: 1 });

        let end = spans
            .iter()
            .map(|s| s.end)
            .max()
            .unwrap_or_else(|| rholang_parser::SourcePos { line: 1, col: 1 });

        rholang_parser::SourceSpan { start, end }
    }

    /// Merge exactly two spans (common case for binary expressions)
    pub fn merge_two_spans(
        left: rholang_parser::SourceSpan,
        right: rholang_parser::SourceSpan,
    ) -> rholang_parser::SourceSpan {
        rholang_parser::SourceSpan {
            start: std::cmp::min(left.start, right.start),
            end: std::cmp::max(left.end, right.end),
        }
    }

    /// Create synthetic span based on existing span with optional offset
    /// Useful for compiler-generated nodes that should be near original source
    pub fn synthetic_span_from(base: rholang_parser::SourceSpan) -> rholang_parser::SourceSpan {
        base // For now, just return the base span
    }

    /// Create synthetic span at a specific position (for single point operations)
    pub fn synthetic_span_at(pos: rholang_parser::SourcePos) -> rholang_parser::SourceSpan {
        rholang_parser::SourceSpan {
            start: pos,
            end: pos,
        }
    }

    /// Extract start position from a span (for compatibility with single position needs)
    pub fn span_start_pos(span: rholang_parser::SourceSpan) -> rholang_parser::SourcePos {
        span.start
    }

    /// Extract end position from a span
    pub fn span_end_pos(span: rholang_parser::SourceSpan) -> rholang_parser::SourcePos {
        span.end
    }

    /// Check if a span represents a single position (start == end)
    pub fn is_single_position(span: rholang_parser::SourceSpan) -> bool {
        span.start == span.end
    }

    /// BOUNDARY CONVERSION: Convert span to legacy SourcePosition
    /// Only used when interfacing with legacy systems that expect SourcePosition
    /// Takes the start position of the span
    pub fn to_legacy_source_position(span: rholang_parser::SourceSpan) -> SourcePosition {
        SourcePosition {
            row: span.start.line,
            column: span.start.col,
        }
    }

    /// BOUNDARY CONVERSION: Convert SourcePos to legacy SourcePosition
    /// Only used when interfacing with legacy systems that expect SourcePosition
    pub fn pos_to_legacy_source_position(pos: rholang_parser::SourcePos) -> SourcePosition {
        SourcePosition {
            row: pos.line,
            column: pos.col,
        }
    }

    /// BOUNDARY CONVERSION: Convert legacy SourcePosition to SourcePos
    /// Used when converting from legacy to new span-based types
    pub fn legacy_to_pos(legacy: SourcePosition) -> rholang_parser::SourcePos {
        rholang_parser::SourcePos {
            line: legacy.row,
            col: legacy.column,
        }
    }

    /// BOUNDARY CONVERSION: Convert legacy SourcePosition to SourceSpan (single point)
    /// Used when converting from legacy to new span-based types
    pub fn legacy_to_span(legacy: SourcePosition) -> rholang_parser::SourceSpan {
        let pos = Self::legacy_to_pos(legacy);
        Self::pos_to_span(pos)
    }

    // ============================================================================
    // LET NORMALIZATION SPAN HELPERS
    // ============================================================================

    /// Create span for synthetic variable based on original binding
    /// Used for compiler-generated variables that need traceable source locations
    pub fn variable_span_from_binding(
        binding_span: rholang_parser::SourceSpan,
        var_index: usize,
    ) -> rholang_parser::SourceSpan {
        // Offset slightly to distinguish synthetic variables while maintaining context
        let mut start = binding_span.start;
        start.col = start.col.saturating_add(var_index as usize); // Prevent overflow
        rholang_parser::SourceSpan { start, end: start }
    }

    /// Create span for compiler-generated constructs with meaningful context
    /// Provides better debugging by linking synthetic constructs to their source context
    pub fn synthetic_construct_span(
        context_span: rholang_parser::SourceSpan,
        construct_offset: u32,
    ) -> rholang_parser::SourceSpan {
        // Use context but mark as synthetic with controlled offset
        let mut span = context_span;
        span.start.col = span.start.col.saturating_add(construct_offset as usize);
        span.end = span.start;
        span
    }

    /// Derive meaningful span for generated send processes
    /// Covers the range from binding variable to expression
    pub fn send_span_from_binding(
        lhs_span: rholang_parser::SourceSpan,
        rhs_span: rholang_parser::SourceSpan,
    ) -> rholang_parser::SourceSpan {
        Self::merge_two_spans(lhs_span, rhs_span)
    }

    /// Extract a name hint from various AST constructs for debugging
    /// Returns meaningful names for compiler-generated variables
    pub fn extract_name_hint_from_var(var: &rholang_parser::ast::Var) -> String {
        match var {
            rholang_parser::ast::Var::Id(id) => id.name.to_string(),
            rholang_parser::ast::Var::Wildcard => "wildcard".to_string(),
        }
    }
}

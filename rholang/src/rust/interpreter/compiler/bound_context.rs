use super::exports::SourcePosition;

#[derive(Debug, Clone, PartialEq)]
pub struct BoundContext<T> {
    pub index: usize,
    pub typ: T,
    pub source_position: SourcePosition,
}

// ===== SourceSpan-based parallel types =====

/// SourceSpan-based version of BoundContext for use with rholang-rs parser types
///
/// This provides a parallel implementation that uses rholang_parser::SourceSpan
/// instead of the legacy SourcePosition, enabling precise source range tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundContextSpan<T> {
    pub index: usize,
    pub typ: T,
    pub source_span: rholang_parser::SourceSpan,
}

use super::exports::SourcePosition;

#[derive(Debug, Clone, PartialEq)]
pub struct FreeContext<T: Clone> {
    pub level: usize,
    pub typ: T,
    pub source_position: SourcePosition,
}

// ===== SourceSpan-based parallel types =====

/// SourceSpan-based version of FreeContext for use with rholang-rs parser types
///
/// This provides a parallel implementation that uses rholang_parser::SourceSpan
/// instead of the legacy SourcePosition, enabling precise source range tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct FreeContextSpan<T: Clone> {
    pub level: usize,
    pub typ: T,
    pub source_span: rholang_parser::SourceSpan,
}

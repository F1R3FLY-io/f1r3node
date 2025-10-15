#[derive(Debug, Clone, PartialEq)]
pub struct BoundContext<T> {
    pub index: usize,
    pub typ: T,
    pub source_span: rholang_parser::SourceSpan,
}

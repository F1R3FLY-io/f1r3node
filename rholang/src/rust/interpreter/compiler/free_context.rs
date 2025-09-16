#[derive(Debug, Clone, PartialEq)]
pub struct FreeContextSpan<T: Clone> {
    pub level: usize,
    pub typ: T,
    pub source_span: rholang_parser::SourceSpan,
}

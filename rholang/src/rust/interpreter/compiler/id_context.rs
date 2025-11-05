/// IdContext variant that uses SourceSpan for full range information
/// Suitable for AnnProc, AnnName, and other constructs that have full spans
pub type IdContextSpan<T> = (String, T, rholang_parser::SourceSpan);

/// IdContext variant that uses SourcePos for single position information  
/// Suitable for Id types and other constructs that have single positions
pub type IdContextPos<T> = (String, T, rholang_parser::SourcePos);

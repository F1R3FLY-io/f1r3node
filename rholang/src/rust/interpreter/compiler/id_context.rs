use super::exports::SourcePosition;

// Define the IdContext type alias
pub type IdContext<T> = (String, T, SourcePosition);

// ===== SourceSpan-based parallel types =====

/// SourceSpan-based versions of IdContext for use with rholang-rs parser types
///
/// This provides parallel implementations that use rholang_parser types
/// instead of the legacy SourcePosition, enabling precise source location tracking.

/// IdContext variant that uses SourceSpan for full range information
/// Suitable for AnnProc, AnnName, and other constructs that have full spans
pub type IdContextSpan<T> = (String, T, rholang_parser::SourceSpan);

/// IdContext variant that uses SourcePos for single position information  
/// Suitable for Id types and other constructs that have single positions
pub type IdContextPos<T> = (String, T, rholang_parser::SourcePos);

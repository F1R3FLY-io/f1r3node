pub mod rust;

// Re-export public API for easy access
pub use rust::graphz::{
    apply,
    // Public functions (equivalent to Scala's object Graphz functions)
    default_shape,
    subgraph,
    FileSerializer,

    GraphArrowType,

    GraphRank,
    GraphRankDir,
    // Serializers
    GraphSerializer,
    GraphShape,
    GraphStyle,
    // Enums
    GraphType,
    // Main types
    Graphz,
    GraphzError,

    ListSerializer,
    StringSerializer,
};

use super::exports::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct BoundMap<T> {
    next_index: usize,
    index_bindings: HashMap<String, BoundContext<T>>,
}

impl<T: Clone> BoundMap<T> {
    pub fn new() -> Self {
        BoundMap {
            next_index: 0,
            index_bindings: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<BoundContext<T>>
    where
        T: Clone,
    {
        self.index_bindings.get(name).map(|context| BoundContext {
            index: self.next_index - context.index - 1,
            typ: context.typ.clone(),
            source_position: context.source_position.clone(),
        })
    }

    pub fn put(&self, binding: IdContext<T>) -> BoundMap<T> {
        let (name, typ, source_position) = binding;
        let mut new_bindings = self.index_bindings.clone();
        new_bindings.insert(
            name,
            BoundContext {
                index: self.next_index,
                typ,
                source_position,
            },
        );
        BoundMap {
            next_index: self.next_index + 1,
            index_bindings: new_bindings,
        }
    }

    pub fn put_all(&self, bindings: Vec<IdContext<T>>) -> BoundMap<T> {
        let mut new_map = self.clone();
        for binding in bindings {
            new_map = new_map.put(binding);
        }
        new_map
    }

    pub fn absorb_free(&self, free_map: FreeMap<T>) -> BoundMap<T> {
        let mut new_bindings = self.index_bindings.clone();
        for (name, context) in free_map.level_bindings {
            new_bindings.insert(
                name,
                BoundContext {
                    index: context.level + self.next_index,
                    typ: context.typ,
                    source_position: context.source_position,
                },
            );
        }
        BoundMap {
            next_index: self.next_index + free_map.next_level,
            index_bindings: new_bindings,
        }
    }

    // Rename this method to avoid conflict with Iterator::count
    pub fn get_count(&self) -> usize {
        self.next_index
    }
}

// ===== SourceSpan-based parallel types =====

use super::bound_context::BoundContextSpan;
use super::id_context::{IdContextSpan, IdContextPos};

/// SourceSpan-based version of BoundMap for use with rholang-rs parser types
///
/// This provides a parallel implementation that uses rholang_parser::SourceSpan
/// instead of the legacy SourcePosition, enabling precise source range tracking.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundMapSpan<T> {
    next_index: usize,
    index_bindings: HashMap<String, BoundContextSpan<T>>,
}

impl<T: Clone> BoundMapSpan<T> {
    pub fn new() -> Self {
        BoundMapSpan {
            next_index: 0,
            index_bindings: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<BoundContextSpan<T>>
    where
        T: Clone,
    {
        self.index_bindings.get(name).map(|context| BoundContextSpan {
            index: self.next_index - context.index - 1,
            typ: context.typ.clone(),
            source_span: context.source_span,
        })
    }

    /// Put binding with SourceSpan (for AnnProc, AnnName, etc.)
    pub fn put_span(&self, binding: IdContextSpan<T>) -> BoundMapSpan<T> {
        let (name, typ, source_span) = binding;
        let mut new_bindings = self.index_bindings.clone();
        new_bindings.insert(
            name,
            BoundContextSpan {
                index: self.next_index,
                typ,
                source_span,
            },
        );
        BoundMapSpan {
            next_index: self.next_index + 1,
            index_bindings: new_bindings,
        }
    }

    /// Put binding with SourcePos (for Id types) - converts to SourceSpan
    pub fn put_pos(&self, binding: IdContextPos<T>) -> BoundMapSpan<T> {
        let (name, typ, source_pos) = binding;
        // Convert SourcePos to SourceSpan (single point span)
        let source_span = rholang_parser::SourceSpan {
            start: source_pos,
            end: source_pos,
        };
        self.put_span((name, typ, source_span))
    }

    pub fn put_all_span(&self, bindings: Vec<IdContextSpan<T>>) -> BoundMapSpan<T> {
        let mut new_map = self.clone();
        for binding in bindings {
            new_map = new_map.put_span(binding);
        }
        new_map
    }

    pub fn put_all_pos(&self, bindings: Vec<IdContextPos<T>>) -> BoundMapSpan<T> {
        let mut new_map = self.clone();
        for binding in bindings {
            new_map = new_map.put_pos(binding);
        }
        new_map
    }

    pub fn absorb_free_span(&self, free_map: &FreeMapSpan<T>) -> BoundMapSpan<T> {
        let mut new_bindings = self.index_bindings.clone();
        for (name, context) in &free_map.level_bindings {
            new_bindings.insert(
                name.clone(),
                BoundContextSpan {
                    index: context.level + self.next_index,
                    typ: context.typ.clone(),
                    source_span: context.source_span,
                },
            );
        }
        BoundMapSpan {
            next_index: self.next_index + free_map.next_level,
            index_bindings: new_bindings,
        }
    }

    pub fn get_count(&self) -> usize {
        self.next_index
    }
}

impl<T: Clone> Default for BoundMapSpan<T> {
    fn default() -> Self {
        Self::new()
    }
}

// Forward declaration for FreeMapSpan
use super::free_map::FreeMapSpan;

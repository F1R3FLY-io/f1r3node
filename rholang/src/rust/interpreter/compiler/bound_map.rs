use super::bound_context::BoundContextSpan;
use super::id_context::{IdContextPos, IdContextSpan};
use std::collections::HashMap;

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
        self.index_bindings
            .get(name)
            .map(|context| BoundContextSpan {
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

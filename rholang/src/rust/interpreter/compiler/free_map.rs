use super::free_context::FreeContextSpan;
use super::id_context::{IdContextPos, IdContextSpan};
use models::rhoapi::connective::ConnectiveInstance;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct FreeMapSpan<T: Clone> {
    pub next_level: usize,
    pub level_bindings: HashMap<String, FreeContextSpan<T>>,
    pub wildcards: Vec<rholang_parser::SourceSpan>,
    pub connectives: Vec<(ConnectiveInstance, rholang_parser::SourceSpan)>,
}

impl<T: Clone> FreeMapSpan<T> {
    pub fn new() -> Self {
        FreeMapSpan {
            next_level: 0,
            level_bindings: HashMap::new(),
            wildcards: Vec::new(),
            connectives: Vec::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<FreeContextSpan<T>>
    where
        T: Clone,
    {
        self.level_bindings.get(name).cloned()
    }

    /// Put binding with SourceSpan (for AnnProc, AnnName, etc.)
    pub fn put_span(&self, binding: IdContextSpan<T>) -> Self {
        let (name, typ, source_span) = binding;

        let mut new_level_bindings = self.level_bindings.clone();
        new_level_bindings.insert(
            name,
            FreeContextSpan {
                level: self.next_level,
                typ,
                source_span,
            },
        );

        FreeMapSpan {
            next_level: self.next_level + 1,
            level_bindings: new_level_bindings,
            wildcards: self.wildcards.clone(),
            connectives: self.connectives.clone(),
        }
    }

    /// Put binding with SourcePos (for Id types) - converts to SourceSpan
    pub fn put_pos(&self, binding: IdContextPos<T>) -> Self {
        let (name, typ, source_pos) = binding;
        // Convert SourcePos to SourceSpan (single point span)
        let source_span = rholang_parser::SourceSpan {
            start: source_pos,
            end: source_pos,
        };
        self.put_span((name, typ, source_span))
    }

    pub fn put_all_span(&self, bindings: Vec<IdContextSpan<T>>) -> Self {
        let mut new_free_map = self.clone();
        for binding in bindings {
            new_free_map = new_free_map.put_span(binding);
        }
        new_free_map
    }

    pub fn put_all_pos(&self, bindings: Vec<IdContextPos<T>>) -> Self {
        let mut new_free_map = self.clone();
        for binding in bindings {
            new_free_map = new_free_map.put_pos(binding);
        }
        new_free_map
    }

    /// Returns the new map, and a list of the shadowed variables with their spans
    pub fn merge(
        &self,
        free_map: FreeMapSpan<T>,
    ) -> (FreeMapSpan<T>, Vec<(String, rholang_parser::SourceSpan)>) {
        let (acc_env, shadowed) = free_map.level_bindings.into_iter().fold(
            (self.level_bindings.clone(), Vec::new()),
            |(mut acc_env, mut shadowed), (name, free_context)| {
                acc_env.insert(
                    name.clone(),
                    FreeContextSpan {
                        level: free_context.level + self.next_level,
                        typ: free_context.typ,
                        source_span: free_context.source_span,
                    },
                );

                (acc_env, {
                    if self.level_bindings.contains_key(&name) {
                        shadowed.insert(0, (name, free_context.source_span));
                        shadowed
                    } else {
                        shadowed
                    }
                })
            },
        );

        let mut new_wildcards = self.wildcards.clone();
        new_wildcards.extend(free_map.wildcards.into_iter());
        let mut new_connectives = self.connectives.clone();
        new_connectives.extend(free_map.connectives);

        (
            FreeMapSpan {
                next_level: self.next_level + free_map.next_level,
                level_bindings: acc_env,
                wildcards: new_wildcards,
                connectives: new_connectives,
            },
            shadowed,
        )
    }

    pub fn add_wildcard(&self, source_span: rholang_parser::SourceSpan) -> Self {
        let mut updated_wildcards = self.wildcards.clone();
        updated_wildcards.push(source_span);

        FreeMapSpan {
            next_level: self.next_level,
            level_bindings: self.level_bindings.clone(),
            wildcards: updated_wildcards,
            connectives: self.connectives.clone(),
        }
    }

    pub fn add_connective(
        &self,
        connective: ConnectiveInstance,
        source_span: rholang_parser::SourceSpan,
    ) -> Self {
        let mut updated_connectives = self.connectives.clone();
        updated_connectives.push((connective, source_span));

        FreeMapSpan {
            next_level: self.next_level,
            level_bindings: self.level_bindings.clone(),
            wildcards: self.wildcards.clone(),
            connectives: updated_connectives,
        }
    }

    pub fn count(&self) -> usize {
        self.next_level + self.wildcards.len() + self.connectives.len()
    }

    pub fn count_no_wildcards(&self) -> usize {
        self.next_level
    }
}

impl<T: Clone> Default for FreeMapSpan<T> {
    fn default() -> Self {
        Self::new()
    }
}

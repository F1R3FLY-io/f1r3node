// Type mapper for PathMap - converts between EPathMap (protobuf) and ParPathMap (Rust)

use crate::rhoapi::EPathMap;
use super::path_map::{ParPathMap, PathMapTrie, PathMapNode};

pub struct ParPathMapTypeMapper;

impl ParPathMapTypeMapper {
    pub fn e_pathmap_to_par_pathmap(e_pathmap: EPathMap) -> ParPathMap {
        let trie = PathMapTrie::from_elements(e_pathmap.ps);
        ParPathMap::new(
            trie,
            e_pathmap.connective_used,
            e_pathmap.locally_free,
            e_pathmap.remainder,
        )
    }

    pub fn par_pathmap_to_e_pathmap(par_pathmap: ParPathMap) -> EPathMap {
        // Extract all elements from the trie
        let ps = extract_elements_from_trie(&par_pathmap.trie);
        
        EPathMap {
            ps,
            locally_free: par_pathmap.locally_free,
            connective_used: par_pathmap.connective_used,
            remainder: par_pathmap.remainder,
        }
    }
}

fn extract_elements_from_trie(trie: &PathMapTrie) -> Vec<crate::rhoapi::Par> {
    let mut elements = Vec::new();
    collect_elements(&trie.root, &mut elements);
    elements
}

fn collect_elements(node: &PathMapNode, elements: &mut Vec<crate::rhoapi::Par>) {
    if let Some(ref par) = node.value {
        elements.push(par.clone());
    }
    for child in node.children.values() {
        collect_elements(child, elements);
    }
}


use crate::rhoapi::{EPathMap, Var};
use super::pathmap_integration::{RholangPathMap, PathMapCreationResult, create_pathmap_from_elements};

pub struct PathMapCrateTypeMapper;

impl PathMapCrateTypeMapper {
    /// Convert from protobuf EPathMap to PathMap-based structure
    pub fn e_pathmap_to_rholang_pathmap(e_pathmap: &EPathMap) -> PathMapCreationResult {
        create_pathmap_from_elements(&e_pathmap.ps, e_pathmap.remainder.clone())
    }

    /// Convert from PathMap back to protobuf EPathMap
    pub fn rholang_pathmap_to_e_pathmap(
        map: &RholangPathMap,
        connective_used: bool,
        locally_free: &[u8],
        remainder: Option<Var>,
    ) -> EPathMap {
        // Extract all values (flattened) from the trie as elements for proto EPathMap
        let mut ps = Vec::new();
        for (_, par) in map.iter() {
            ps.push(par.clone());
        }

        EPathMap {
            ps,
            locally_free: locally_free.to_vec(),
            connective_used,
            remainder,
        }
    }
}

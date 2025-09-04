use crate::rust::TopoSort;
use models::rust::block_hash::BlockHash;
use models::rust::casper::pretty_printer::PrettyPrinter;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiableEdge {
    pub from: String,
    pub to: String,
}

impl VerifiableEdge {
    pub fn new(from: String, to: String) -> Self {
        Self { from, to }
    }
}

// Equivalent to Scala's Show[VerifiableEdge]
impl Display for VerifiableEdge {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{} {}", self.from, self.to)
    }
}

pub struct MachineVerifiableDag;

impl MachineVerifiableDag {
    pub fn apply<F>(toposort: TopoSort, fetch_parents: F) -> Vec<VerifiableEdge>
    where
        F: Fn(BlockHash) -> Vec<BlockHash>,
    {
        let mut result_parts = Vec::new();

        // Equivalent to toposort.foldM(List.empty[VerifiableEdge])
        for block_hashes in toposort {
            // Equivalent to blockHashes.toList.traverse { blockHash => ... }
            let mut blocks_and_parents = Vec::new();

            for block_hash in block_hashes {
                let parents = fetch_parents(block_hash.clone());
                let block_hash_str = PrettyPrinter::build_string_bytes(&block_hash);
                let parent_strings: Vec<String> = parents
                    .iter()
                    .map(|p| PrettyPrinter::build_string_bytes(p))
                    .collect();
                blocks_and_parents.push((block_hash_str, parent_strings));
            }

            // Equivalent to blocksAndParents.flatMap { case (b, bp) => bp.map(VerifiableEdge(b, _)) }
            let new_edges: Vec<VerifiableEdge> = blocks_and_parents
                .into_iter()
                .flat_map(|(block_hash_str, parent_strings)| {
                    parent_strings.into_iter().map(move |parent_str| {
                        VerifiableEdge::new(block_hash_str.clone(), parent_str)
                    })
                })
                .collect();

            // Collect each group separately to reverse group order later (not individual elements)
            result_parts.push(new_edges);
        }

        // Equivalent to _ ++ acc: reverse the order of groups and flatten
        // This gives us O(n) performance instead of O(n²) from splice(0..0, ...)
        result_parts.into_iter().rev().flatten().collect()
    }
}

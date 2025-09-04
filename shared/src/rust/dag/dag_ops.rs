// See shared/src/main/scala/coop/rchain/dag/DagOps.scala

use std::collections::{HashSet, VecDeque};
use std::hash::Hash;

pub fn bf_traverse<A, F>(start: Vec<A>, mut neighbors: F) -> Vec<A>
where
    A: Eq + Hash + Clone,
    F: FnMut(&A) -> Vec<A>,
{
    let mut result = Vec::new();
    let mut queue = VecDeque::new();
    let mut visited = HashSet::new();

    // Initialize queue with starting nodes
    for node in start {
        queue.push_back(node);
    }

    while let Some(curr) = queue.pop_front() {
        if visited.contains(&curr) {
            continue;
        }

        // Mark as visited and add to result
        visited.insert(curr.clone());
        result.push(curr.clone());

        // Get neighbors and add unvisited ones to queue
        let ns = neighbors(&curr);
        for n in ns {
            if !visited.contains(&n) {
                queue.push_back(n);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_simple_tree() {
        // Create a simple tree:
        //      1
        //    /   \
        //   2     3
        //  / \   / \
        // 4   5 6   7
        let mut graph = HashMap::new();
        graph.insert(1, vec![2, 3]);
        graph.insert(2, vec![4, 5]);
        graph.insert(3, vec![6, 7]);
        graph.insert(4, vec![]);
        graph.insert(5, vec![]);
        graph.insert(6, vec![]);
        graph.insert(7, vec![]);

        let neighbors = |n: &i32| graph.get(n).unwrap_or(&vec![]).clone();

        let result = bf_traverse(vec![1], neighbors);
        assert_eq!(result, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_cyclic_graph() {
        // Create a graph with cycles:
        //  1 --- 2
        //  |     |
        //  3 --- 4
        let mut graph = HashMap::new();
        graph.insert(1, vec![2, 3]);
        graph.insert(2, vec![1, 4]);
        graph.insert(3, vec![1, 4]);
        graph.insert(4, vec![2, 3]);

        let neighbors = |n: &i32| graph.get(n).unwrap_or(&vec![]).clone();

        let result = bf_traverse(vec![1], neighbors);
        // The exact order can vary, but we should visit each node exactly once
        assert_eq!(result.len(), 4);
        assert!(result.contains(&1));
        assert!(result.contains(&2));
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }

    #[test]
    fn test_empty_start() {
        let neighbors = |_: &i32| vec![];
        let result = bf_traverse(Vec::<i32>::new(), neighbors);
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_multiple_start_nodes() {
        let mut graph = HashMap::new();
        graph.insert(1, vec![3]);
        graph.insert(2, vec![4]);
        graph.insert(3, vec![]);
        graph.insert(4, vec![]);

        let neighbors = |n: &i32| graph.get(n).unwrap_or(&vec![]).clone();

        let result = bf_traverse(vec![1, 2], neighbors);
        // We should visit all nodes starting with the initial set
        assert_eq!(result.len(), 4);
        // The first two nodes should be our start nodes in order
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        // The remaining nodes should include 3 and 4 (order may vary)
        assert!(result.contains(&3));
        assert!(result.contains(&4));
    }
}

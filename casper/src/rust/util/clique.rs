// See casper/src/main/scala/coop/rchain/casper/util/Clique.scala

use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub struct Clique;

impl Clique {
    /// Find the weight of the maximum-weight clique in the graph defined by edges.
    /// Uses Bron-Kerbosch with pivot selection and weight-based branch pruning.
    /// Prunes branches where the remaining candidates cannot exceed the current best.
    pub fn find_maximum_clique_by_weight<A>(edges: &[(A, A)], weights: &HashMap<A, i64>) -> i64
    where
        A: Eq + Hash + Clone + Ord,
    {
        let adj = Self::get_adj(edges);
        let nodes = Self::get_node_set(edges);

        // Start with the maximum single-node weight as baseline
        let mut best_weight: i64 = weights.values().max().cloned().unwrap_or(0);

        Self::expand_max_weight(
            &Vec::new(),
            0, // current clique weight
            &nodes,
            &HashSet::new(),
            &adj,
            weights,
            &mut best_weight,
        );

        best_weight
    }

    /// Bron-Kerbosch with pivot selection and weight-based pruning.
    /// Instead of materializing all cliques, tracks the best weight found so far
    /// and prunes branches where the upper bound (current weight + max possible
    /// remaining weight) cannot exceed the best.
    fn expand_max_weight<A>(
        ans: &[A],
        ans_weight: i64,
        p: &HashSet<A>,
        x: &HashSet<A>,
        adj: &HashMap<A, HashSet<A>>,
        weights: &HashMap<A, i64>,
        best_weight: &mut i64,
    ) where
        A: Eq + Hash + Clone + Ord,
    {
        if p.is_empty() && x.is_empty() {
            // Found a maximal clique — update best if better
            if !ans.is_empty() && ans_weight > *best_weight {
                *best_weight = ans_weight;
            }
            return;
        }

        if p.is_empty() {
            return;
        }

        // Upper bound pruning: current weight + sum of all remaining candidate weights
        let remaining_weight: i64 = p.iter().map(|v| weights.get(v).unwrap_or(&0)).sum();
        if ans_weight + remaining_weight <= *best_weight {
            return;
        }

        // Pivot selection: choose vertex in P ∪ X with most neighbors in P
        let empty = HashSet::new();
        let pivot = p
            .union(x)
            .max_by_key(|v| p.intersection(adj.get(v).unwrap_or(&empty)).count())
            .unwrap()
            .clone();

        // Iterate over P \ N(pivot)
        let pivot_neighbors = adj.get(&pivot).unwrap_or(&empty);
        let candidates: Vec<A> = p.difference(pivot_neighbors).cloned().collect();

        let mut current_p = p.clone();
        let mut current_x = x.clone();

        for v in candidates {
            let v_weight = *weights.get(&v).unwrap_or(&0);
            let v_neighbors = adj.get(&v).unwrap_or(&empty);

            let new_p: HashSet<A> = current_p.intersection(v_neighbors).cloned().collect();
            let new_x: HashSet<A> = current_x.intersection(v_neighbors).cloned().collect();

            let mut new_ans = ans.to_vec();
            new_ans.push(v.clone());

            Self::expand_max_weight(
                &new_ans,
                ans_weight + v_weight,
                &new_p,
                &new_x,
                adj,
                weights,
                best_weight,
            );

            current_p.remove(&v);
            current_x.insert(v);
        }
    }

    // e is a list of undirected edges
    fn get_node_set<A>(e: &[(A, A)]) -> HashSet<A>
    where
        A: Eq + Hash + Clone,
    {
        e.iter()
            .flat_map(|it| vec![it.0.clone(), it.1.clone()])
            .collect()
    }

    // e is a list of undirected edges
    fn get_adj<A>(e: &[(A, A)]) -> HashMap<A, HashSet<A>>
    where
        A: Eq + Hash + Clone,
    {
        let directed_edges: Vec<(A, A)> = e
            .iter()
            .flat_map(|it| vec![(it.0.clone(), it.1.clone()), (it.1.clone(), it.0.clone())])
            .collect();

        directed_edges
            .into_iter()
            .filter(|(src, dst)| src != dst)
            .fold(HashMap::new(), |mut map, (src, dst)| {
                map.entry(src).or_insert_with(HashSet::new).insert(dst);
                map
            })
    }
}

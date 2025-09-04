// See casper/src/main/scala/coop/rchain/casper/util/Clique.scala

use std::cmp::max;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::Hash;

pub struct Clique;

impl Clique {
    pub fn find_maximum_clique_by_weight<A>(edges: &[(A, A)], weights: &BTreeMap<A, i64>) -> i64
    where
        A: Eq + Hash + Clone + Ord,
    {
        let max_weight = weights.values().max().cloned().unwrap_or(0);
        Self::find_cliques_recursive(edges)
            .iter()
            .fold(max_weight, |largest_weight, clique| {
                let weight: i64 = clique
                    .iter()
                    .map(|item| weights.get(item).unwrap_or(&0))
                    .sum();
                max(weight, largest_weight)
            })
    }

    /*
      Scala -> Rust
      &~ -> difference
      & -> intersection
      ++ -> union
    */

    // e is a list of undirected edges
    pub fn find_cliques_recursive<A>(e: &[(A, A)]) -> Vec<Vec<A>>
    where
        A: Eq + Hash + Clone,
    {
        let adj = Self::get_adj(e);

        fn expand<A>(
            ans: Vec<A>,
            p: HashSet<A>,
            x: HashSet<A>,
            adj: &HashMap<A, HashSet<A>>,
        ) -> Vec<Vec<A>>
        where
            A: Eq + Hash + Clone,
        {
            if p.is_empty() && x.is_empty() {
                if ans.is_empty() {
                    vec![]
                } else {
                    vec![ans]
                }
            } else {
                let u = p
                    .union(&x)
                    .cloned()
                    .collect::<HashSet<A>>()
                    .iter()
                    .max_by_key(|v| {
                        p.intersection(adj.get(v).unwrap_or(&HashSet::new()))
                            .count()
                    })
                    .unwrap()
                    .clone();

                let p_ext: HashSet<A> = p
                    .difference(adj.get(&u).unwrap_or(&HashSet::new()))
                    .cloned()
                    .collect();

                let runtime_parameter: Vec<(HashSet<A>, Option<A>)> = p_ext
                    .into_iter()
                    .scan((HashSet::new(), None::<A>), |(acc_set, _), elem| {
                        acc_set.insert(elem.clone());
                        Some((acc_set.clone(), Some(elem)))
                    })
                    .collect();

                runtime_parameter
                    .into_iter()
                    .flat_map(|param| match param {
                        (s, Some(elem)) => {
                            let adj_q = adj.get(&elem).cloned().unwrap_or_default();

                            let new_p = p
                                .difference(&s)
                                .cloned()
                                .collect::<HashSet<A>>()
                                .intersection(&adj_q)
                                .cloned()
                                .collect();

                            let new_x = x
                                .union(&s)
                                .cloned()
                                .collect::<HashSet<A>>()
                                .intersection(&adj_q)
                                .cloned()
                                .collect();

                            let mut new_ans = ans.clone();
                            new_ans.push(elem);

                            expand(new_ans, new_p, new_x, adj).into_iter()
                        }
                        _ => panic!("Could not calculate findCliquesRecursive"),
                    })
                    .collect()
            }
        }
        expand(Vec::new(), Self::get_node_set(e), HashSet::new(), &adj)
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

        let adj: HashMap<A, HashSet<A>> = directed_edges
            .into_iter()
            .filter(|(src, dst)| src != dst)
            .fold(HashMap::new(), |mut map, (src, dst)| {
                map.entry(src).or_insert_with(HashSet::new).insert(dst);
                map
            });

        adj
    }
}

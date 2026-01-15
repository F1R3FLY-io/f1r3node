use itertools::Itertools;
use models::rhoapi::Par;

use super::par_count::ParCount;

// See rholang/src/main/scala/coop/rchain/rholang/interpreter/matcher/ParSpatialMatcherUtils.scala - subPars
pub fn sub_pars(
    par: &Par,
    min: &ParCount,
    max: &ParCount,
    min_prune: &ParCount,
    max_prune: &ParCount,
) -> impl Iterator<Item = (Par, Par)> {
    let send_max = std::cmp::min(
        max.sends as isize,
        par.sends.len() as isize - min_prune.sends as isize,
    );
    let receive_max = std::cmp::min(
        max.receives as isize,
        par.receives.len() as isize - min_prune.receives as isize,
    );
    let new_max = std::cmp::min(
        max.news as isize,
        par.news.len() as isize - min_prune.news as isize,
    );
    let expr_max = std::cmp::min(
        max.exprs as isize,
        par.exprs.len() as isize - min_prune.exprs as isize,
    );
    let match_max = std::cmp::min(
        max.matches as isize,
        par.matches.len() as isize - min_prune.matches as isize,
    );
    let unf_max = std::cmp::min(
        max.unforgeables as isize,
        par.unforgeables.len() as isize - min_prune.unforgeables as isize,
    );
    let bundle_max = std::cmp::min(
        max.bundles as isize,
        par.bundles.len() as isize - min_prune.bundles as isize,
    );

    let send_min = std::cmp::max(
        min.sends as isize,
        par.sends.len() as isize - max_prune.sends as isize,
    );
    let receive_min = std::cmp::max(
        min.receives as isize,
        par.receives.len() as isize - max_prune.receives as isize,
    );
    let new_min = std::cmp::max(
        min.news as isize,
        par.news.len() as isize - max_prune.news as isize,
    );
    let expr_min = std::cmp::max(
        min.exprs as isize,
        par.exprs.len() as isize - max_prune.exprs as isize,
    );
    let match_min = std::cmp::max(
        min.matches as isize,
        par.matches.len() as isize - max_prune.matches as isize,
    );
    let unf_min = std::cmp::max(
        min.unforgeables as isize,
        par.unforgeables.len() as isize - max_prune.unforgeables as isize,
    );
    let bundle_min = std::cmp::max(
        min.bundles as isize,
        par.bundles.len() as isize - max_prune.bundles as isize,
    );

    // This ideally should return type 'Iterator' instead of type 'Vec'
    fn min_max_subsets<A: Clone + std::fmt::Debug>(
        _as: &Vec<A>,
        min_size: isize,
        max_size: isize,
    ) -> Vec<(Vec<A>, Vec<A>)> {
        fn counted_max_subsets<A: Clone>(
            _as: Vec<A>,
            max_size: isize,
        ) -> Vec<(Vec<A>, Vec<A>, isize)> {
            match _as.split_first() {
                None => vec![(_as.to_vec(), _as.to_vec(), 0)],
                Some((head, rem)) => {
                    let mut results = vec![(_as[0..0].to_vec(), _as.clone(), 0)];

                    let counted_tail = counted_max_subsets(rem.to_vec(), max_size);
                    for (mut tail, mut complement, count) in counted_tail {
                        if count == max_size {
                            complement.insert(0, head.clone());
                            results.push((tail, complement, count));
                        } else if tail.is_empty() {
                            tail.insert(0, head.clone());
                            results.push((tail, complement, 1));
                        } else {
                            complement.insert(0, head.clone());
                            tail.insert(0, head.clone());

                            results.push((tail.clone(), complement.clone(), count));
                            results.push((tail, complement, count + 1));
                        }
                    }
                    results
                }
            }
        }

        // This ideally should return type 'Iterator' instead of type 'Vec'
        fn worker<A: Clone + std::fmt::Debug>(
            _as: Vec<A>,
            min_size: isize,
            max_size: isize,
        ) -> Vec<(Vec<A>, Vec<A>, isize)> {
            if max_size < 0 {
                vec![]
            } else if min_size > max_size {
                vec![]
            } else if min_size <= 0 {
                if max_size == 0 {
                    vec![(_as[0..0].to_vec(), _as.clone(), 0)]
                } else {
                    counted_max_subsets(_as, max_size)
                }
            } else {
                match _as.split_first() {
                    None => vec![],
                    Some((head, rem)) => {
                        let decr = min_size - 1;
                        let mut results = vec![];

                        let counted_tail = worker(rem.to_vec(), decr, max_size);
                        for (mut tail, mut complement, count) in counted_tail {
                            if count == max_size {
                                complement.insert(0, head.clone());
                                results.push((tail, complement, count));
                            } else if count == decr {
                                tail.insert(0, head.clone());
                                results.push((tail, complement, min_size));
                            } else {
                                complement.insert(0, head.clone());
                                tail.insert(0, head.clone());

                                results.push((tail.clone(), complement.clone(), count));
                                results.push((tail, complement, count + 1));
                            }
                        }
                        results
                    }
                }
            }
        }

        worker(_as.to_vec(), min_size, max_size)
            .iter()
            .map(|x| (x.0.clone(), x.1.clone()))
            .collect()
    }

    let result = min_max_subsets(&par.sends, send_min, send_max)
        .into_iter()
        .cartesian_product(min_max_subsets(&par.receives, receive_min, receive_max).into_iter())
        .cartesian_product(min_max_subsets(&par.news, new_min, new_max).into_iter())
        .cartesian_product(min_max_subsets(&par.exprs, expr_min, expr_max).into_iter())
        .cartesian_product(min_max_subsets(&par.matches, match_min, match_max).into_iter())
        .cartesian_product(min_max_subsets(&par.unforgeables, unf_min, unf_max).into_iter())
        .cartesian_product(min_max_subsets(&par.bundles, bundle_min, bundle_max).into_iter())
        .map(
            |(
                (((((sub_sends, sub_receives), sub_news), sub_exprs), sub_matches), sub_unfs),
                sub_bundles,
            )| {
                (
                    Par {
                        sends: sub_sends.0,
                        receives: sub_receives.0,
                        news: sub_news.0,
                        exprs: sub_exprs.0,
                        matches: sub_matches.0,
                        unforgeables: sub_unfs.0,
                        bundles: sub_bundles.0,
                        connectives: Vec::default(),
                        locally_free: Vec::default(),
                        connective_used: false,
                        use_blocks: Vec::default(),  // Reifying RSpaces
                    },
                    Par {
                        sends: sub_sends.1,
                        receives: sub_receives.1,
                        news: sub_news.1,
                        exprs: sub_exprs.1,
                        matches: sub_matches.1,
                        unforgeables: sub_unfs.1,
                        bundles: sub_bundles.1,
                        connectives: Vec::default(),
                        locally_free: Vec::default(),
                        connective_used: false,
                        use_blocks: Vec::default(),  // Reifying RSpaces
                    },
                )
            },
        );

    result
}

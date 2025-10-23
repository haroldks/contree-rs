use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::caching::{Cache, Entry};
use crate::common::{classification_error, PointSelector, SearchConfig, Statistics};
use crate::data::view::DataView;
use crate::data::Dataset;
use std::collections::VecDeque;
use std::time::Instant;
use rand::Rng;
use rand::rngs::ThreadRng;
use crate::algorithms::depth2::ConTreeDepth2;

pub struct ConTree<const USE_CACHE: bool> {
    config: SearchConfig,
    statistics: Statistics,
    cache: Cache,
    specialized: ConTreeDepth2,
    runtime: Instant,
    rng: ThreadRng
}

impl <const USE_CACHE: bool>ConTree<USE_CACHE> {
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: usize,
        split_selection_strategy: PointSelector,
        max_gap: usize,
        use_heuristic: bool,
        fast_d2: bool,
    ) -> Self {
        Self {
            // TODO : gap is set as large enough
            cache: Cache::default(),
            config: SearchConfig::new(min_sup, max_depth, max_time, max_gap, max_error, use_heuristic, fast_d2, split_selection_strategy),
            statistics: Statistics::default(),
            specialized: ConTreeDepth2::default(),
            runtime: Instant::now(),
            rng: rand::rng(),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        let root_view = DataView::root(dataset, self.config.use_heuristic);

        if USE_CACHE {
            self.cache = Cache::new(self.config.max_depth, root_view.total_instances);
        }
        let root_index = self.cache.init();

        self.statistics.num_features = root_view.get_feature_number();
        self.statistics.num_samples = root_view.total_instances;

        let (error, label) = classification_error(root_view.get_labels_freqs());
        let mut entry = Entry::default();
        entry.error = error;
        entry.label = label;

        if USE_CACHE {
            if let Some(entry) = self.cache.get_mut(root_index) {
                self.config.max_error = self.config.max_error.min(error);
                entry.error = error;
                entry.label = label;
            }
        }

        let root_config = self.config;
        self.runtime = Instant::now();
        self.expand_node_with_view(&root_view, &root_config, &mut entry, 0, true, root_config.max_error);
        self.statistics.error = entry.error;
        self.statistics.cache_size = self.cache.len();
        self.statistics.duration = self.elapsed_time();
    }

    fn expand_node_with_view(
        &mut self,
        view: &DataView,
        config: &SearchConfig,
        current_best: &mut Entry,
        parent_index: usize,
        is_new: bool,
        upper_bound: usize,
    ) {
        if  USE_CACHE && (current_best.error == 0 || view.len() == 0) {
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.is_leaf = true;
                entry.is_optimal = true;
                *current_best = *entry;
            }
            return;
        }

        if USE_CACHE && !is_new  {
            self.statistics.cache_hits +=1;
            return;
        }



        // Check if the node exists already in the cache. If so check if it is the same cache entry otherwise update it

        // let view_index = self.cache.find(&view.bitset, config.max_depth);
        // if let Some(&index) = view_index {
        //     dbg!("Bitset stored : at {:?}", index, &current_best);
        //     self.statistics.cache_hits += 1;
        //     *current_best = *self.cache.get(index).unwrap();
        //     return;
        // }


        if config.max_depth == 0 {
            current_best.is_optimal = true;
            current_best.is_leaf = true;

            if USE_CACHE {
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    entry.is_optimal = true;
                    entry.is_leaf = true;
                    *current_best = *entry;

                }
            }

            return;
        }

        if current_best.error <= config.max_gap || view.len() <= 1 {
            return;
        }

        if config.fast_d2 && config.max_depth <= 2 {
            let _tree = self.specialized.fit(view, &config, current_best, upper_bound, &mut self.statistics);
            // tree.print();
            if USE_CACHE {
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    *entry = *current_best
                }
                self.statistics.specialized_solver_call += 1;
            }
            return;
        }


        //TODO: Here explore each feature and the predetermined order

        let num_features = view.get_feature_number();
        let heuristics_data = view.features_best_score();
        debug_assert!(
            num_features == heuristics_data.len(),
            "Missmatch with heuristics and features number"
        );
        for it in 0..num_features {
            let (_, feat) = heuristics_data[it];
            self.expand_on_feature(view, feat, parent_index, config, current_best, upper_bound.min(current_best.error));

            if current_best.error == 0 {
                if USE_CACHE {
                    if let Some(entry) = self.cache.get_mut(parent_index) {
                        entry.is_optimal = true;
                    }
                }

                return;
            }

            if USE_CACHE {
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    entry.is_optimal = true;
                }
            }

            if !self.time_remains() {
                return;
            }
            // TODO : Check runtime
        }
    }

    fn expand_on_feature(
        &mut self,
        view: &DataView,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) {


        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        if config.point_selector == PointSelector::First {
            self.expand_on_feature_gini_priority(view, feature_index, cache_index, config, current_best, upper_bound);
            return;
        }


        // if feature_index == 0 {
        //     println!("No order : {:?}", view.get_possible_split_indices(feature_index));
        //     println!("order : {:?}", view.ordered_possible_splits(feature_index));
        //     println!("54 =  {}", view.get_possible_split_indices(feature_index)[54]);
        //     println!("56 =  {}", view.get_possible_split_indices(feature_index)[56]);
        // }

        let possible_index_split = view.get_possible_split_indices(feature_index);
        if possible_index_split.len() == 0 {
            return;
        }

        let mut pruner = IntervalsPruner::new(&possible_index_split, config.max_gap);
        let mut queue = VecDeque::new();
        let init_bound = Bound::new(0, possible_index_split.len() - 1, None, None);
        queue.push_back(init_bound);


        while !queue.is_empty() {
            if !self.time_remains() {
                return;
            }

            let mut current_bound = queue.pop_front().unwrap();
            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("Left bound: {} Right bound: {}", current_bound.left_bound, current_bound.right_bound);
            //
            // }
            if pruner.subinterval_pruning(&current_bound, current_best.error) {
                // if self.config.max_depth == config.max_depth && feature_index == 0{
                //     println!("\tpruned by subinterval");
                //
                // }
                continue;
            }

            pruner.interval_shrinking(&mut current_bound, current_best.error);
            if !current_bound.is_valid() {
                // if self.config.max_depth == config.max_depth && feature_index == 0{
                //     println!("\tpruned by shrinking");
                //
                // }
                continue;
            }

            // TODO : Allow to use other points as the best split and random



            let selected_point = self.select_point(config, &current_bound);
            let split_point = possible_index_split[selected_point];
            let int_half_distance = split_point
                .saturating_sub(possible_index_split[current_bound.left_bound])
                .max(possible_index_split[current_bound.right_bound].saturating_sub(split_point));

            let threshold_value = if selected_point > 0 {
                let previous = feature_column_ids[possible_index_split[selected_point - 1]];
                let point = feature_column_ids[split_point];
                (feature_column[previous].value()
                    + feature_column[point].value())
                    / 2.0
            } else {

                let point = feature_column_ids[split_point];
                (feature_column[point].value() + feature_column[feature_column_ids[0]].value()) / 2.0
            };

            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("\tSplit point {split_point} split unique value index {} threshold {}", feature_column[feature_column_ids[split_point]].unique_value_id(), threshold_value);
            //
            // }

            let (left_view, right_view) =  view.split(feature_index, split_point);
            // println!("Left view size {:?} and right view size : {:?} when feature {} split at {}", left_view.len(), right_view.len(), feature_index, split_point);

            // TODO : Do larger and smaller tree comparison and take the first

            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                continue;
            }

            // TODO : check first if I want to use ub
            let left_upper_bound = current_best.error.min(upper_bound);
            self.statistics.general_solver_call += 1;

            let left_config = config.derive_left();
            let mut left_entry = Entry::default();

            let (mut left_index, mut left_is_new) = (0, true);

            if USE_CACHE {
                (left_is_new, left_index) = self.cache.insert(&left_view.bitset, left_config.max_depth);
                if let Some(entry) = self.cache.get_mut(left_index) {
                    if left_is_new {
                        let (error, label) = classification_error(left_view.get_labels_freqs());
                        entry.error = error;
                        entry.label = label;
                        entry.depth = self.config.max_depth - left_config.max_depth;
                    }
                    left_entry = * entry;
                }
            }
            else {
                let (error, label) = classification_error(left_view.get_labels_freqs());
                left_entry.error = error;
                left_entry.label = label;
                left_entry.depth = self.config.max_depth - left_config.max_depth;
            }

            self.expand_node_with_view(&left_view, &left_config, &mut left_entry, left_index, left_is_new, left_upper_bound);

            // FIXME : Saturing sub ??
            let right_upper_bound = current_best.error.min(upper_bound).saturating_sub(left_entry.error).max(int_half_distance);
            let mut right_error = current_best.error;


            if right_upper_bound > 0 || (right_upper_bound == 0 && current_best.error == left_entry.error) {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);
                let mut right_entry = Entry::default();
                right_entry.error = current_best.error;

                let (mut right_index, mut right_is_new) = (0,true);

                if USE_CACHE {

                    (right_is_new, right_index) = self.cache.insert(&right_view.bitset, left_config.max_depth);
                    if let Some(entry) = self.cache.get_mut(right_index) {
                        if right_is_new {
                            let (error, label) = classification_error(right_view.get_labels_freqs());
                            entry.error = error;
                            entry.label = label;
                            entry.depth = self.config.max_depth - right_config.max_depth;
                        }
                        right_entry = *entry;
                    }
                }
                else {
                    let (error, label) = classification_error(right_view.get_labels_freqs());
                    left_entry.error = error;
                    left_entry.label = label;
                    left_entry.depth = self.config.max_depth - right_config.max_depth;
                }

                self.expand_node_with_view(&right_view, &right_config, &mut right_entry, right_index, right_is_new, right_upper_bound);
                right_error = right_entry.error;

                let feature_best = left_entry.error + right_entry.error;
                if feature_best < current_best.error {
                    current_best.error = feature_best;
                    current_best.feature = feature_index;
                    current_best.split = threshold_value;
                    current_best.left = left_index;
                    current_best.right = right_index;

                   if USE_CACHE {
                       if let Some(entry) = self.cache.get_mut(cache_index) {
                           *entry = *current_best;
                       }
                   }
                }
            }
            else {
                right_error = usize::MAX;
            }

            pruner.add_result(selected_point, left_entry.error, right_error);
            if current_bound.left_bound == current_bound.right_bound {
                continue
            }

            let score_difference = (left_entry.error + right_error).saturating_sub(current_best.error);
            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("\tScore difference: {} left error: {} right error: {} current error: {}",score_difference, left_entry.error, right_error, current_best.error);
            // }
            let (left_bound, right_bound) = pruner.neighbourhood_pruning(score_difference, current_bound.left_bound, current_bound.right_bound, selected_point);

            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("\tNew bound {} {}",left_bound, right_bound);
            // }

            if left_bound <= current_bound.right_bound {
                queue.push_back(Bound {
                    left_bound,
                    right_bound: current_bound.right_bound,
                    last_split_left_index: Some(selected_point),
                    last_split_right_index: current_bound.last_split_right_index,
                });
            }

            if current_bound.left_bound <= right_bound {
                queue.push_back(Bound {
                    left_bound: current_bound.left_bound,
                    right_bound,
                    last_split_left_index: current_bound.last_split_left_index,
                    last_split_right_index: Some(selected_point),
                });
            }

        }
    }

    /// Explore splits prioritized by gini quality while using pruner
    fn expand_on_feature_gini_priority(
        &mut self,
        view: &DataView,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) {
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        // Get position-sorted splits for pruner
        let possible_splits = view.get_possible_split_indices(feature_index);

        if possible_splits.len() == 0 {
            return;
        }

        // Get splits with their positions, sorted by gini (best first)
        let sorted_by_heuristic_indices = view.ordered_possible_splits(feature_index);

        // Initialize pruner with position-sorted data
        let mut pruner = IntervalsPruner::new(&possible_splits, config.max_gap);

        let mut queue = VecDeque::new();
        let init_bound = Bound::new(0, possible_splits.len() - 1, None, None);
        queue.push_back(init_bound);

        // Track which split indices have been pruned
        let mut pruned = vec![false; possible_splits.len()];

        let mut iteration = 0;

        // while !queue.is_empty() {
        //     if !self.time_remains() {
        //         return;
        //     }
        //
        //     // Skip if this split has been pruned
        //     if pruned[iteration] {
        //         iteration +=1;
        //         continue;
        //     }
        //
        //     let mut current_bound = queue.pop_front().unwrap();
        //     if pruner.subinterval_pruning(&current_bound, current_best.error) {
        //         if self.config.max_depth == config.max_depth && feature_index == 0{
        //             println!("\tpruned by subinterval");
        //
        //         }
        //         continue;
        //     }
        //
        //     pruner.interval_shrinking(&mut current_bound, current_best.error);
        //     if !current_bound.is_valid() {
        //         if self.config.max_depth == config.max_depth && feature_index == 0{
        //             println!("\tpruned by shrinking");
        //
        //         }
        //         continue;
        //     }
        //
        //     let selected_point = possible_splits[sorted_by_heuristic_indices[iteration]];
        //     let split_point = possible_splits[selected_point];
        //     let int_half_distance = split_point
        //         .saturating_sub(possible_splits[current_bound.left_bound])
        //         .max(possible_splits[current_bound.right_bound].saturating_sub(split_point));
        //
        //
        //     let threshold_value = if selected_point > 0 {
        //         let previous = feature_column_ids[possible_splits[selected_point - 1]];
        //         let point = feature_column_ids[split_point];
        //         (feature_column[previous].value()
        //             + feature_column[point].value())
        //             / 2.0
        //     } else {
        //
        //         let point = feature_column_ids[split_point];
        //         (feature_column[point].value() + feature_column[feature_column_ids[0]].value()) / 2.0
        //     };
        //
        //     if self.config.max_depth == config.max_depth && feature_index == 0{
        //         println!("\tSplit point {split_point} split unique value index {} threshold {}", feature_column[feature_column_ids[split_point]].unique_value_id(), threshold_value);
        //
        //     }
        //
        //     let (left_view, right_view) =  view.split(feature_index, split_point);
        //     if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
        //         continue;
        //     }
        //
        //     // TODO : check first if I want to use ub
        //     let left_upper_bound = current_best.error.min(upper_bound);
        //     self.statistics.general_solver_call += 1;
        //
        //     let left_config = config.derive_left();
        //     let mut left_entry = Entry::default();
        //
        //     let (mut left_index, mut left_is_new) = (0, true);
        //
        //     if USE_CACHE {
        //         (left_is_new, left_index) = self.cache.insert(&left_view.bitset, left_config.max_depth);
        //         if let Some(entry) = self.cache.get_mut(left_index) {
        //             if left_is_new {
        //                 let (error, label) = classification_error(left_view.get_labels_freqs());
        //                 entry.error = error;
        //                 entry.label = label;
        //                 entry.depth = self.config.max_depth - left_config.max_depth;
        //             }
        //             left_entry = * entry;
        //         }
        //     }
        //     else {
        //         let (error, label) = classification_error(left_view.get_labels_freqs());
        //         left_entry.error = error;
        //         left_entry.label = label;
        //         left_entry.depth = self.config.max_depth - left_config.max_depth;
        //     }
        //
        //     self.expand_node_with_view(&left_view, &left_config, &mut left_entry, left_index, left_is_new, left_upper_bound);
        //
        //     // FIXME : Saturing sub ??
        //     let right_upper_bound = current_best.error.min(upper_bound).saturating_sub(left_entry.error).max(int_half_distance);
        //     let mut right_error = current_best.error;
        //
        //
        //     if right_upper_bound > 0 || (right_upper_bound == 0 && current_best.error == left_entry.error) {
        //         self.statistics.general_solver_call += 1;
        //         let right_config = config.derive_right(left_config.max_gap);
        //         let mut right_entry = Entry::default();
        //         right_entry.error = current_best.error;
        //
        //         let (mut right_index, mut right_is_new) = (0,true);
        //
        //         if USE_CACHE {
        //
        //             (right_is_new, right_index) = self.cache.insert(&right_view.bitset, left_config.max_depth);
        //             if let Some(entry) = self.cache.get_mut(right_index) {
        //                 if right_is_new {
        //                     let (error, label) = classification_error(right_view.get_labels_freqs());
        //                     entry.error = error;
        //                     entry.label = label;
        //                     entry.depth = self.config.max_depth - right_config.max_depth;
        //                 }
        //                 right_entry = *entry;
        //             }
        //         }
        //         else {
        //             let (error, label) = classification_error(right_view.get_labels_freqs());
        //             left_entry.error = error;
        //             left_entry.label = label;
        //             left_entry.depth = self.config.max_depth - right_config.max_depth;
        //         }
        //
        //         self.expand_node_with_view(&right_view, &right_config, &mut right_entry, right_index, right_is_new, right_upper_bound);
        //         right_error = right_entry.error;
        //
        //         let feature_best = left_entry.error + right_entry.error;
        //         if feature_best < current_best.error {
        //             current_best.error = feature_best;
        //             current_best.feature = feature_index;
        //             current_best.split = threshold_value;
        //             current_best.left = left_index;
        //             current_best.right = right_index;
        //
        //             if USE_CACHE {
        //                 if let Some(entry) = self.cache.get_mut(cache_index) {
        //                     *entry = *current_best;
        //                 }
        //             }
        //         }
        //     }
        //     else {
        //         right_error = usize::MAX;
        //     }
        //     pruner.add_result(selected_point, left_entry.error, right_error);
        //     if current_bound.left_bound == current_bound.right_bound {
        //         continue
        //     }
        //
        //     let score_difference = (left_entry.error + right_error).saturating_sub(current_best.error);
        //     if self.config.max_depth == config.max_depth && feature_index == 0{
        //         println!("\tScore difference: {} left error: {} right error: {} current error: {}",score_difference, left_entry.error, right_error, current_best.error);
        //     }
        //     let (left_bound, right_bound) = pruner.neighbourhood_pruning(score_difference, current_bound.left_bound, current_bound.right_bound, selected_point);
        //     if self.config.max_depth == config.max_depth && feature_index == 0{
        //         println!("\tNew bound {} {}",left_bound, right_bound);
        //     }
        //
        //     if left_bound <= current_bound.right_bound {
        //         queue.push_back(Bound {
        //             left_bound,
        //             right_bound: current_bound.right_bound,
        //             last_split_left_index: Some(selected_point),
        //             last_split_right_index: current_bound.last_split_right_index,
        //         });
        //         for i in current_bound.left_bound..left_bound {
        //             pruned[i] = true;
        //         }
        //     }
        //
        //     if current_bound.left_bound <= right_bound {
        //         queue.push_back(Bound {
        //             left_bound: current_bound.left_bound,
        //             right_bound,
        //             last_split_left_index: current_bound.last_split_left_index,
        //             last_split_right_index: Some(selected_point),
        //         });
        //         for i in (right_bound + 1)..current_bound.right_bound {
        //             pruned[i] = true;
        //         }
        //     }
        //
        // }





        // Iterate through splits in gini order (best first)
        for &split_idx in sorted_by_heuristic_indices {
            if !self.time_remains() {
                return;
            }


            // Skip if this split has been pruned
            if pruned[split_idx] {
                continue;
            }

            let mut current_left = split_idx;
            while current_left > 0 && pruned[current_left - 1] {
                current_left -= 1;
            }
            if current_left > 0 {
                current_left -= 1;
            }

            let mut current_right = split_idx;
            while current_right < possible_splits.len() - 1 && pruned[current_right + 1] {
                current_right += 1;
            }
            if current_right < possible_splits.len() - 1 {
                current_right += 1;
            }

            let split_point = possible_splits[split_idx];


            // Calculate threshold value
            let threshold_value = if split_idx > 0 {
                let previous = feature_column_ids[possible_splits[split_idx - 1]];
                let point = feature_column_ids[split_point];
                (feature_column[previous].value() + feature_column[point].value()) / 2.0
            } else {
                let point = feature_column_ids[split_point];
                (feature_column[point].value() + feature_column[feature_column_ids[0]].value()) / 2.0
            };

            // Split the view
            let (left_view, right_view) = view.split(feature_index, split_point);

            // Check minimum support
            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                pruned[split_idx] = true;
                continue;
            }

            // Evaluate left subtree
            let left_upper_bound = current_best.error.min(upper_bound);
            self.statistics.general_solver_call += 1;

            let left_config = config.derive_left();
            let mut left_entry = Entry::default();
            let (mut left_index, mut left_is_new) = (0, true);

            if USE_CACHE {
                (left_is_new, left_index) = self.cache.insert(&left_view.bitset, left_config.max_depth);
                if let Some(entry) = self.cache.get_mut(left_index) {
                    if left_is_new {
                        let (error, label) = classification_error(left_view.get_labels_freqs());
                        entry.error = error;
                        entry.label = label;
                        entry.depth = self.config.max_depth - left_config.max_depth;
                    }
                    left_entry = *entry;
                }
            } else {
                let (error, label) = classification_error(left_view.get_labels_freqs());
                left_entry.error = error;
                left_entry.label = label;
                left_entry.depth = self.config.max_depth - left_config.max_depth;
            }

            self.expand_node_with_view(&left_view, &left_config, &mut left_entry, left_index, left_is_new, left_upper_bound);

            // Evaluate right subtree
            let int_half_distance = split_point
                .saturating_sub(possible_splits[0])
                .max(possible_splits[possible_splits.len() - 1].saturating_sub(split_point));
            let right_upper_bound = current_best.error.min(upper_bound).saturating_sub(left_entry.error).max(int_half_distance);
            let mut right_error = current_best.error;

            if right_upper_bound > 0 || (right_upper_bound == 0 && current_best.error == left_entry.error) {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);
                let mut right_entry = Entry::default();
                right_entry.error = current_best.error;
                let (mut right_index, mut right_is_new) = (0, true);

                if USE_CACHE {
                    (right_is_new, right_index) = self.cache.insert(&right_view.bitset, left_config.max_depth);
                    if let Some(entry) = self.cache.get_mut(right_index) {
                        if right_is_new {
                            let (error, label) = classification_error(right_view.get_labels_freqs());
                            entry.error = error;
                            entry.label = label;
                            entry.depth = self.config.max_depth - right_config.max_depth;
                        }
                        right_entry = *entry;
                    }
                } else {
                    let (error, label) = classification_error(right_view.get_labels_freqs());
                    right_entry.error = error;
                    right_entry.label = label;
                    right_entry.depth = self.config.max_depth - right_config.max_depth;
                }

                self.expand_node_with_view(&right_view, &right_config, &mut right_entry, right_index, right_is_new, right_upper_bound);
                right_error = right_entry.error;
                let feature_best = left_entry.error + right_error;
                if feature_best < current_best.error {
                    current_best.error = feature_best;
                    current_best.feature = feature_index;
                    current_best.split = threshold_value;
                    current_best.left = left_index;
                    current_best.right = right_index;

                    if USE_CACHE {
                        if let Some(entry) = self.cache.get_mut(cache_index) {
                            *entry = *current_best;
                        }
                    }

                }


            }
            else {
                right_error = usize::MAX;
            }


            // Record result in pruner
            pruner.add_result(split_idx, left_entry.error, right_error);

            // Use pruner to mark neighbors as pruned
            let score_difference = (left_entry.error + right_error).saturating_sub(current_best.error);
            let (new_left_bound, new_right_bound) = pruner.neighbourhood_pruning(
                score_difference,
                0,
                possible_splits.len() - 1,
                split_idx
            );

            // Mark pruned regions
            for i in current_left..new_left_bound {
                pruned[i] = true;
            }
            for i in (new_right_bound + 1)..=current_right {
                pruned[i] = true;
            }

            if current_best.error == 0 {
                break;
            }

        }
    }


    pub fn select_point(&mut self, config: &SearchConfig, bound: &Bound) -> usize {
        match config.point_selector {
            PointSelector::Mid => (bound.left_bound + bound.right_bound) / 2,
            PointSelector::First => bound.left_bound,
            PointSelector::Random => {
                self.rng.random_range(bound.left_bound..=bound.right_bound)
            }
        }
    }


    fn time_remains(&self) -> bool {
        self.elapsed_time() < self.config.max_time
    }

    fn elapsed_time(&self) -> f64 {
        self.runtime.elapsed().as_secs_f64()
    }

    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }


}

#[cfg(test)]
mod contree_test {
    use std::path::Path;
    use crate::algorithms::continuous_tree::ConTree;
    use crate::common::PointSelector;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;

    #[test]
    fn test_run() -> Result<(), DataReaderError> {

        let reader = DataReader::default();
        let path = Path::new("test_data/avila.txt");
        let mut dataset = reader.read_file(path)?;
        dataset.sort_features();


        let mut contree: ConTree<true> = ConTree::new(1, 3, 100.0, usize::MAX,  PointSelector::Mid, 0, false, true, );

        contree.fit(&dataset);

        println!("Cache Size : {:?}", contree.cache.len());
        println!("Stats : {:#?}", contree.statistics);



        Ok(())

    }

}

use crate::algorithms::depth2::ConTreeDepth2;
use crate::algorithms::interval_pruner::{Bound, IntervalsPruner};
use crate::caching::{Cache, Entry};
use crate::common::{classification_error, PointSelector, SearchConfig, Statistics};
use crate::data::view::DataView;
use crate::data::{Dataset, Feature};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::VecDeque;
use std::iter::Peekable;
use std::time::Instant;
use serde::Deserialize;
use crate::common::budget_iterator::BudgetIterator;
use crate::tree::{NodeInfos, Tree, TreeNode};

pub struct ConTreeLds<const USE_CACHE: bool> {
    config: SearchConfig,
    statistics: Statistics,
    cache: Cache,
    specialized: ConTreeDepth2,
    runtime: Instant,
    rng: ThreadRng,
    max_discrepancy: usize,
    max_split_number: usize,
    split_budget: usize,
    budget_strategy: BudgetStrategy,
    budget_iterator: Peekable<BudgetIterator>,
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Default)]
pub enum BudgetStrategy {
    Diagonal,
    Lexicographic,
    GeometricBoth,
    #[default]
    FeaturePriority,
}

impl<const USE_CACHE: bool> ConTreeLds<USE_CACHE> {
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
        let mut config = SearchConfig::new(
            min_sup,
            max_depth,
            max_time,
            max_gap,
            max_error,
            use_heuristic,
            fast_d2,
            split_selection_strategy,
        );
        config.use_discrepancy = true;

        Self {
            cache: Cache::default(),
            config,
            statistics: Statistics::default(),
            specialized: ConTreeDepth2::default(),
            runtime: Instant::now(),
            rng: rand::rng(),
            max_discrepancy: usize::MAX,
            max_split_number: usize::MAX,
            split_budget: 1,
            budget_strategy: BudgetStrategy::GeometricBoth,
            budget_iterator: BudgetIterator::new(0, 0).peekable(),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {

        let root_view = DataView::root(dataset, self.config.use_heuristic);
        // let root_view = DataView::root_with_split_budget(dataset, 90);

        // let max_discrepancy = self.max_discrepancy.min(Self::discrepancy_limit(
        //     root_view.get_feature_number(),
        //     self.config.max_depth,
        // ));

        // self.max_split_number = root_view.get_max_splits();
        let mut is_done = false;
        while !is_done {
            is_done = self.partial_fit(&root_view);
        }
        // for _ in 0..=max_discrepancy {
        //     let is_done = self.partial_fit(&root_view);
        //     if  is_done {
        //         break;
        //     }
        // }

    }

    pub fn partial_fit(&mut self, root_view: &DataView) -> bool {
        self.config.nb_runs += 1;
        let mut root_index = 0;
        let mut entry = Entry::default();
        if self.config.nb_runs <= 1 {
            if USE_CACHE {
                self.cache = Cache::new(self.config.max_depth, root_view.total_instances);
            }

            self.max_discrepancy = self.max_discrepancy.min(Self::discrepancy_limit(
                root_view.get_feature_number(),
                self.config.max_depth,
            ));
            self.max_split_number = root_view.get_max_splits();

            self.budget_iterator = BudgetIterator::new(self.max_discrepancy, self.max_split_number).peekable();
            // println!("Max discrepancy : {:?}", self.max_discrepancy);
            // println!("Max Split budget : {:?}", self.max_split_number);
            root_index = self.cache.init();

            self.statistics.num_features = root_view.get_feature_number();
            self.statistics.num_samples = root_view.total_instances;

            let (error, label) = classification_error(root_view.get_labels_freqs());
            entry.error = error;
            entry.label = label;

            if USE_CACHE {
                if let Some(entry) = self.cache.get_mut(root_index) {
                    self.config.max_error = self.config.max_error.min(error);
                    entry.error = error;
                    entry.label = label;
                    entry.ub = self.config.max_error;
                }
            }
            self.runtime = Instant::now();
        }

        let mut is_optimal = false;

        if let Some(e) = self.cache.get(root_index) {
            entry = *e
        }
        if entry.is_optimal {
            return true;
        }

        let mut config = self.config;

        config.discrepancy = 0;

        let error = entry.error;
        // println!("Initial discrepancy budget : {:?}", config.budget);
        let stopped = self.expand_node_with_view(&root_view, &config, &mut entry, 0, true, error);

        let search_exhausted = self.is_search_exhausted(self.config.budget, self.split_budget, self.budget_strategy);

        // println!(
        //     "For disc budget {} and split budget {} with runtime {}  cache len {} general solver {} and specialized solver {} give error {} and ub {} stopped : {}",
        //     self.config.budget,
        //      // self.max_discrepancy,
        //     self.split_budget,
        //     self.elapsed_time(),
        //     self.cache.len(),
        //     self.statistics.general_solver_call,
        //     self.statistics.specialized_solver_call,
        //     entry.error,
        //     error,
        //     stopped
        // );

        // let tree = self.get_solution_tree();

        if !stopped || !self.time_remains() ||  search_exhausted {
            entry.is_optimal = true;
            is_optimal = true;
            // return true
        }


        if self.config.nb_runs > 1 {
           if self.budget_iterator.peek().is_some() {
                (self.config.budget, self.split_budget) = self.budget_iterator.next().unwrap();
           }

            // let iteration = self.config.nb_runs - 1;
            //
            // let (next_feat, next_split) = self.next_budgets(
            //     self.config.budget,
            //     self.split_budget,
            //     iteration,
            //     self.budget_strategy,
            // );
            // self.config.budget = next_feat;
            // self.split_budget = next_split;
            // self.split_budget = next_split;

            // println!("Next budgets  {} {}", x, y);
        }


        // if self.config.nb_runs > 1 {
        //     if self.config.budget == 0 {
        //         self.config.budget = 1;
        //     }
        //     self.config.budget *= 2;
        //     self.split_budget += 1;
        //     if self.split_budget > self.max_split_number {
        //         self.split_budget = self.max_split_number;
        //     } else if self.split_budget == self.max_split_number {
        //         self.split_budget  = self.max_split_number + 1;
        //     }
        // }


        // if self.config.budget > self.max_discrepancy {
        //     self.split_budget = self.max_split_number;
        // }

        // println!("Self config {:?}", self.config);
        self.statistics.error = entry.error;
        self.statistics.cache_size = self.cache.len();
        self.statistics.duration = self.elapsed_time();
        is_optimal
    }

    fn next_budgets(
        &self,
        current_feat: usize,
        current_split: usize,
        iteration: usize,
        strategy: BudgetStrategy,
    ) -> (usize, usize) {
        match strategy {
            BudgetStrategy::Diagonal => self.diagonal_budgets(iteration),
            BudgetStrategy::Lexicographic => self.lexicographic_budgets(current_feat, current_split),
            BudgetStrategy::GeometricBoth => self.geometric_budgets(iteration),
            BudgetStrategy::FeaturePriority => self.feature_priority_budgets(current_feat, current_split),
        }
    }

    fn diagonal_budgets(&self, iteration: usize) -> (usize, usize) {
        // Step 1: Find which diagonal the iteration is on
        let mut diagonal = 0;
        let mut cumulative = 0;

        // Diagonal d has (d + 1) elements
        while iteration >= cumulative + (diagonal + 1) {
            cumulative += diagonal + 1;
            diagonal += 1;
        }

        // Step 2: Position inside the diagonal
        let pos = iteration - cumulative;

        // Step 3: Map to (dis, split)
        let dis = pos;
        let split = diagonal + 1 - pos;

        (dis.min(self.max_discrepancy), split.min(self.max_split_number))
    }


    fn lexicographic_budgets(&self, current_feat: usize, current_split: usize) -> (usize, usize) {
        let next_split = current_split + 1;

        if next_split > self.max_split_number {
            // Move to next feature, reset splits
            let next_feat = if current_feat == 0 { 1 } else { current_feat * 2 };
            (next_feat, 1)
        } else {
            (current_feat, next_split)
        }
    }

    fn geometric_budgets(&self, iteration: usize) -> (usize, usize) {
        // Features grow slower (every 2 iterations)
        // Splits grow faster (every iteration, then double)
        let feat_budget = iteration / 2;
        let split_budget = if iteration % 2 == 0 {
            1 << (iteration / 2)  // Powers of 2
        } else {
            1 << (iteration / 2)
        };

        (feat_budget, split_budget.min(self.max_split_number))
    }

    fn feature_priority_budgets(&self, current_feat: usize, current_split: usize) -> (usize, usize) {
        // TODO : Change increasing strategy
        let next_feat = if current_feat == 0 { 1 } else { current_feat * 2 };
        let next_split = (current_split + 1).min(self.max_split_number);

        (next_feat.min(self.max_discrepancy), next_split.min(self.max_split_number))
    }


    fn is_search_exhausted(&self, feat_budget: usize, split_budget: usize, strategy: BudgetStrategy) -> bool {
        feat_budget >= self.max_discrepancy && split_budget > self.max_split_number

    }

    fn expand_node_with_view(
        &mut self,
        view: &DataView,
        config: &SearchConfig,
        current_best: &mut Entry,
        parent_index: usize,
        is_new: bool,
        upper_bound: usize,
    ) -> bool {
        if USE_CACHE && (current_best.error == 0 || view.len() == 0) {
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.is_optimal = true;
                entry.ub = upper_bound;
                *current_best = *entry;
            }
            return false;
        }

        if USE_CACHE && !is_new {
            if config.use_discrepancy {
                // TODO : There is an issue with the caching
                if (current_best.age == self.config.nb_runs) || current_best.is_optimal {
                    self.statistics.cache_hits += 1;
                    return false;
                }
            } else {
                self.statistics.cache_hits += 1;
                return false;
            }
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
                    entry.ub = upper_bound;
                    entry.age = self.config.nb_runs;
                    *current_best = *entry;
                }
            }

            return false;
        }

        if current_best.error <= config.max_gap || view.len() <= 1 {
            return false;
        }

        if config.fast_d2 && config.max_depth <= 2  {

            let tree = self.specialized.fit(
                view,
                &config,
                current_best,
                upper_bound,
                &mut self.statistics,
            );
            // _tree.print();
            if USE_CACHE {
                let tree_index = self.cache.insert_tree(tree);
                if let Some(entry) = self.cache.get_mut(parent_index) {
                    current_best.ub = upper_bound;
                    *entry = *current_best;
                    entry.tree_idx = Some(tree_index);
                }
                self.statistics.specialized_solver_call += 1;
            }
            return false;
        }

        //TODO: Here explore each feature and the predetermined order

        let num_features = view.get_feature_number();
        let heuristics_data = view.features_best_score();
        debug_assert!(
            num_features == heuristics_data.len(),
            "Missmatch with heuristics and features number"
        );
        let mut stopped = false;
        for it in 0..num_features {
            // let depth_weight = (self.config.max_depth - config.max_depth) + 1;
            // let feat_discrepancy = config.discrepancy + (it * depth_weight);
            //
            let feat_discrepancy = config.discrepancy + it;
            if feat_discrepancy > config.budget {
                return true;
            }

            let mut node_config = *config;
            node_config.discrepancy = feat_discrepancy;

            let (_, feat) = heuristics_data[it];
            // println!("Expanding Feature  {} with config {:?}", feat, node_config );

            stopped |= self.expand_on_feature(
                view,
                feat,
                parent_index,
                &node_config,
                current_best,
                upper_bound.min(current_best.error),
            );

            // println!("Feature {feat} stopped {stopped} with config {:?}", node_config);
            if current_best.error == 0 {
                if USE_CACHE {
                    if let Some(entry) = self.cache.get_mut(parent_index) {
                        entry.is_optimal = true;
                        entry.ub = upper_bound;
                    }
                }

                return false;
            }

            if !self.time_remains() {
                return true;
            }
        }
        if USE_CACHE {
            if let Some(entry) = self.cache.get_mut(parent_index) {
                entry.ub = upper_bound;
                entry.is_optimal = !stopped;
            }
        }
        stopped
    }

    fn expand_on_feature(
        &mut self,
        view: &DataView,
        feature_index: usize,
        cache_index: usize,
        config: &SearchConfig,
        current_best: &mut Entry,
        upper_bound: usize,
    ) -> bool {
        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        if config.point_selector == PointSelector::First || self.config.nb_runs <= 1 {

            let stopped = self.expand_on_feature_gini_priority(
                view,
                feature_index,
                cache_index,
                config,
                current_best,
                upper_bound,
            );
            return stopped;
        }

        // if feature_index == 0 {
        //     println!("No order : {:?}", view.get_possible_split_indices(feature_index));
        //     println!("order : {:?}", view.ordered_possible_splits(feature_index));
        //     println!("54 =  {}", view.get_possible_split_indices(feature_index)[54]);
        //     println!("56 =  {}", view.get_possible_split_indices(feature_index)[56]);
        // }


        let possible_index_split = view.get_possible_split_indices(feature_index);
        if possible_index_split.len() == 0 {
            return false;
        }

        let mut pruner = IntervalsPruner::new(&possible_index_split, config.max_gap);
        let mut queue = VecDeque::new();
        let init_bound = Bound::new(0, possible_index_split.len() - 1, None, None);
        queue.push_back(init_bound);

        let mut stopped = false;



        while !queue.is_empty() {
            if !self.time_remains() {
                return true;
            }

            let mut current_bound = queue.pop_front().unwrap();

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
                (feature_column[previous].value() + feature_column[point].value()) / 2.0
            } else {
                let point = feature_column_ids[split_point];
                (feature_column[point].value() + feature_column[feature_column_ids[0]].value())
                    / 2.0
            };

            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("\tSplit point {split_point} split unique value index {} threshold {}", feature_column[feature_column_ids[split_point]].unique_value_id(), threshold_value);
            //
            // }

            let (left_view, right_view) = view.split(feature_index, split_point);
            // println!("Left view size {:?} and right view size : {:?} when feature {} split at {}", left_view.len(), right_view.len(), feature_index, split_point);

            // TODO : Do larger and smaller tree comparison and take the first

            if left_view.len() < self.config.min_sup || right_view.len() < self.config.min_sup {
                continue;
            }

            // Determine which dataset is larger
            let process_left_first = left_view.len() >= right_view.len();

            // Always derive both configs
            let left_config = config.derive_left();
            let mut left_entry = Entry::default();
            let mut right_entry = Entry::default();

            let (mut left_index, mut left_is_new) = (0, true);
            let (mut right_index, mut right_is_new) = (0, true);
            let mut stopped = false;

            // Process LARGER dataset first with left_config
            let larger_upper_bound = current_best.error.min(upper_bound);
            self.statistics.general_solver_call += 1;

            if process_left_first {
                // Left is larger - process it first with left_config
                if USE_CACHE {
                    (left_is_new, left_index) =
                        self.cache.insert(&left_view.bitset, left_config.max_depth);
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
                stopped |= self.expand_node_with_view(
                    &left_view,
                    &left_config,
                    &mut left_entry,
                    left_index,
                    left_is_new,
                    larger_upper_bound,
                );
            } else {
                // Right is larger - process it first with left_config
                if USE_CACHE {
                    (right_is_new, right_index) =
                        self.cache.insert(&right_view.bitset, left_config.max_depth);
                    if let Some(entry) = self.cache.get_mut(right_index) {
                        if right_is_new {
                            let (error, label) =
                                classification_error(right_view.get_labels_freqs());
                            entry.error = error;
                            entry.label = label;
                            entry.depth = self.config.max_depth - left_config.max_depth;
                        }
                        right_entry = *entry;
                    }
                } else {
                    let (error, label) = classification_error(right_view.get_labels_freqs());
                    right_entry.error = error;
                    right_entry.label = label;
                    right_entry.depth = self.config.max_depth - left_config.max_depth;
                }
                stopped |= self.expand_node_with_view(
                    &right_view,
                    &left_config,
                    &mut right_entry,
                    right_index,
                    right_is_new,
                    larger_upper_bound,
                );
            }

            // Calculate upper bound for SMALLER dataset
            let larger_error = if process_left_first {
                left_entry.error
            } else {
                right_entry.error
            };
            let smaller_upper_bound = current_best
                .error
                .min(upper_bound)
                .saturating_sub(larger_error)
                .max(int_half_distance);
            let mut right_error = current_best.error;

            // Process SMALLER dataset second with right_config
            if smaller_upper_bound > 0
                || (smaller_upper_bound == 0 && current_best.error == larger_error)
            {
                self.statistics.general_solver_call += 1;
                let right_config = config.derive_right(left_config.max_gap);

                if process_left_first {
                    // Right is smaller - process it second with right_config
                    right_entry.error = current_best.error;
                    if USE_CACHE {
                        (right_is_new, right_index) =
                            self.cache.insert(&right_view.bitset, left_config.max_depth);
                        if let Some(entry) = self.cache.get_mut(right_index) {
                            if right_is_new {
                                let (error, label) =
                                    classification_error(right_view.get_labels_freqs());
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
                    stopped |= self.expand_node_with_view(
                        &right_view,
                        &right_config,
                        &mut right_entry,
                        right_index,
                        right_is_new,
                        smaller_upper_bound,
                    );
                } else {
                    // Left is smaller - process it second with right_config
                    left_entry.error = current_best.error;
                    if USE_CACHE {
                        (left_is_new, left_index) =
                            self.cache.insert(&left_view.bitset, left_config.max_depth);
                        if let Some(entry) = self.cache.get_mut(left_index) {
                            if left_is_new {
                                let (error, label) =
                                    classification_error(left_view.get_labels_freqs());
                                entry.error = error;
                                entry.label = label;
                                entry.depth = self.config.max_depth - right_config.max_depth;
                            }
                            left_entry = *entry;
                        }
                    } else {
                        let (error, label) = classification_error(left_view.get_labels_freqs());
                        left_entry.error = error;
                        left_entry.label = label;
                        left_entry.depth = self.config.max_depth - right_config.max_depth;
                    }
                    stopped |= self.expand_node_with_view(
                        &left_view,
                        &right_config,
                        &mut left_entry,
                        left_index,
                        left_is_new,
                        smaller_upper_bound,
                    );
                }

                right_error = right_entry.error;

                let feature_best = left_entry.error + right_error;
                if feature_best < current_best.error {
                    current_best.age = self.config.nb_runs;
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

                    if feature_best == 0 {
                        return false;
                    }
                }
            } else {
                right_error = usize::MAX;
            }


            pruner.add_result(selected_point, left_entry.error, right_error);
            if current_bound.left_bound == current_bound.right_bound {
                continue;
            }

            let score_difference =
                (left_entry.error + right_error).saturating_sub(current_best.error);
            // if self.config.max_depth == config.max_depth && feature_index == 0{
            //     println!("\tScore difference: {} left error: {} right error: {} current error: {}",score_difference, left_entry.error, right_error, current_best.error);
            // }
            let (left_bound, right_bound) = pruner.neighbourhood_pruning(
                score_difference,
                current_bound.left_bound,
                current_bound.right_bound,
                selected_point,
            );

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
        stopped
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
    ) -> bool {

        let feature_column = view.get_sorted_feature(feature_index);
        let feature_column_ids = view.get_feature_indices(feature_index);

        // Get position-sorted splits for pruner
        let possible_splits = view.get_possible_split_indices(feature_index);

        if possible_splits.len() == 0 {
            return false;
        }

        // Initialize pruner with position-sorted data
        let mut pruner = IntervalsPruner::new(&possible_splits, config.max_gap);

        // Track which split indices have been pruned
        let mut pruned = vec![false; possible_splits.len()];
        let mut stopped = false;

        // Iterate through splits in order (gini if heuristic enabled, otherwise position order)
        let local_split_budget = possible_splits.len().min(self.split_budget);
        // println!("Depth  {} and budget {}", config.max_depth, local_split_budget);
        // if config.max_depth == self.config.max_depth {
        //     println!("Feature index: {} split number {}", feature_index, possible_splits.len());
        // }

        if config.use_heuristic {
            // Use gini-sorted order
            let sorted_indices = view.ordered_possible_splits(feature_index);
            for (idx, &split_idx) in sorted_indices.iter().enumerate() {
                if self.config.nb_runs <= 1 && idx > 0 {
                    return true;
                }

                if idx > local_split_budget - 1 {
                    return true;
                }

                if !self.time_remains() {
                    return true;
                }

                // Skip if this split has been pruned
                if pruned[split_idx] {
                    continue;
                }

                stopped |= self.evaluate_split_gini_priority(
                    view,
                    feature_index,
                    split_idx,
                    &possible_splits,
                    &feature_column,
                    &feature_column_ids,
                    config,
                    current_best,
                    cache_index,
                    upper_bound,
                    &mut pruner,
                    &mut pruned,
                );

                if current_best.error == 0 {
                    return false;
                }
            }
        } else {
            // Use normal position order
            for split_idx in 0..possible_splits.len() {
                if self.config.nb_runs <= 1 && split_idx > 0 {
                    return true;
                }

                if split_idx > local_split_budget - 1 {
                    return true;
                }

                // if split_idx > local_split_budget - 1 {
                //     return true;
                // }

                if !self.time_remains() {
                    return true;
                }

                // Skip if this split has been pruned
                if pruned[split_idx] {
                    continue;
                }

                stopped |= self.evaluate_split_gini_priority(
                    view,
                    feature_index,
                    split_idx,
                    &possible_splits,
                    &feature_column,
                    &feature_column_ids,
                    config,
                    current_best,
                    cache_index,
                    upper_bound,
                    &mut pruner,
                    &mut pruned,
                );

                if current_best.error == 0 {
                    return false;
                }
            }
        }

        stopped |= local_split_budget < possible_splits.len();


        stopped
    }

    #[inline]
    fn evaluate_split_gini_priority(
        &mut self,
        view: &DataView,
        feature_index: usize,
        split_idx: usize,
        possible_splits: &[usize],
        feature_column: &Feature,
        feature_column_ids: &[usize],
        config: &SearchConfig,
        current_best: &mut Entry,
        cache_index: usize,
        upper_bound: usize,
        pruner: &mut IntervalsPruner,
        pruned: &mut [bool],
    ) -> bool {
        // Find current bounds
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
            return false;
        }

        // Determine which dataset is larger
        let process_left_first = left_view.len() >= right_view.len();

        // Always derive both configs
        let left_config = config.derive_left();
        let mut left_entry = Entry::default();
        let mut right_entry = Entry::default();

        let (mut left_index, mut left_is_new) = (0, true);
        let (mut right_index, mut right_is_new) = (0, true);
        let mut stopped = false;

        // Calculate int_half_distance (needed for smaller upper bound)
        let int_half_distance = split_point
            .saturating_sub(possible_splits[0])
            .max(possible_splits[possible_splits.len() - 1].saturating_sub(split_point));

        // Process LARGER dataset first with left_config
        let larger_upper_bound = current_best.error.min(upper_bound);
        self.statistics.general_solver_call += 1;

        if process_left_first {
            // Left is larger - process it first with left_config
            if USE_CACHE {
                (left_is_new, left_index) =
                    self.cache.insert(&left_view.bitset, left_config.max_depth);
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
            stopped |= self.expand_node_with_view(
                &left_view,
                &left_config,
                &mut left_entry,
                left_index,
                left_is_new,
                larger_upper_bound,
            );
        } else {
            // Right is larger - process it first with left_config
            if USE_CACHE {
                (right_is_new, right_index) =
                    self.cache.insert(&right_view.bitset, left_config.max_depth);
                if let Some(entry) = self.cache.get_mut(right_index) {
                    if right_is_new {
                        let (error, label) = classification_error(right_view.get_labels_freqs());
                        entry.error = error;
                        entry.label = label;
                        entry.depth = self.config.max_depth - left_config.max_depth;
                    }
                    right_entry = *entry;
                }
            } else {
                let (error, label) = classification_error(right_view.get_labels_freqs());
                right_entry.error = error;
                right_entry.label = label;
                right_entry.depth = self.config.max_depth - left_config.max_depth;
            }
            stopped |= self.expand_node_with_view(
                &right_view,
                &left_config,
                &mut right_entry,
                right_index,
                right_is_new,
                larger_upper_bound,
            );
        }

        // Calculate upper bound for SMALLER dataset
        let larger_error = if process_left_first {
            left_entry.error
        } else {
            right_entry.error
        };
        let smaller_upper_bound = current_best
            .error
            .min(upper_bound)
            .saturating_sub(larger_error)
            .max(int_half_distance);
        let mut right_error = current_best.error;

        // Process SMALLER dataset second with right_config
        if smaller_upper_bound > 0
            || (smaller_upper_bound == 0 && current_best.error == larger_error)
        {
            self.statistics.general_solver_call += 1;
            let right_config = config.derive_right(left_config.max_gap);

            if process_left_first {
                // Right is smaller - process it second with right_config
                right_entry.error = current_best.error;
                if USE_CACHE {
                    (right_is_new, right_index) =
                        self.cache.insert(&right_view.bitset, left_config.max_depth);
                    if let Some(entry) = self.cache.get_mut(right_index) {
                        if right_is_new {
                            let (error, label) =
                                classification_error(right_view.get_labels_freqs());
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
                stopped |= self.expand_node_with_view(
                    &right_view,
                    &right_config,
                    &mut right_entry,
                    right_index,
                    right_is_new,
                    smaller_upper_bound,
                );
            } else {
                // Left is smaller - process it second with right_config
                left_entry.error = current_best.error;
                if USE_CACHE {
                    (left_is_new, left_index) =
                        self.cache.insert(&left_view.bitset, left_config.max_depth);
                    if let Some(entry) = self.cache.get_mut(left_index) {
                        if left_is_new {
                            let (error, label) = classification_error(left_view.get_labels_freqs());
                            entry.error = error;
                            entry.label = label;
                            entry.depth = self.config.max_depth - right_config.max_depth;
                        }
                        left_entry = *entry;
                    }
                } else {
                    let (error, label) = classification_error(left_view.get_labels_freqs());
                    left_entry.error = error;
                    left_entry.label = label;
                    left_entry.depth = self.config.max_depth - right_config.max_depth;
                }
                stopped |= self.expand_node_with_view(
                    &left_view,
                    &right_config,
                    &mut left_entry,
                    left_index,
                    left_is_new,
                    smaller_upper_bound,
                );
            }

            right_error = right_entry.error;

            let feature_best = left_entry.error + right_error;
            if feature_best < current_best.error {
                current_best.age = self.config.nb_runs;
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

                if feature_best == 0 {
                    return false;
                }
            }
        } else {
            right_error = usize::MAX;
        }

        // Record result in pruner
        pruner.add_result(split_idx, left_entry.error, right_error);
        pruned[split_idx] = true;

        // Use pruner to mark neighbors as pruned
        let score_difference = (left_entry.error + right_error).saturating_sub(current_best.error);
        let minimum_right_value = possible_splits[split_idx] + score_difference + 1;
        let right_value = possible_splits[current_right];
        // println!("Trying to prune using {current_left} and {current_right} bounds and score difference {score_difference} for feature {feature_index} and split index {split_idx} with point {split_point} and left most zero in pruning {:?} right most zero {:?} len {}. The min value for neighbour pruning is {minimum_right_value} compared to the right value of {right_value}", pruner.leftmost_zero_index, pruner.rightmost_zero_index, possible_splits.len());
        let (new_left_bound, new_right_bound) =
            pruner.neighbourhood_pruning(score_difference, current_left, current_right, split_idx);

        // Mark pruned regions
        for i in current_left..new_left_bound {
            pruned[i] = true;
        }
        for i in (new_right_bound + 1)..=current_right {
            pruned[i] = true;
        }
        return stopped;
    }

    pub fn select_point(&mut self, config: &SearchConfig, bound: &Bound) -> usize {
        match config.point_selector {
            PointSelector::Mid => (bound.left_bound + bound.right_bound) / 2,
            PointSelector::First => bound.left_bound,
            PointSelector::Random => self.rng.random_range(bound.left_bound..=bound.right_bound),
        }
    }

    fn discrepancy_limit(nb_candidates: usize, remaining_depth: usize) -> usize {
        let mut max_discrepancy = nb_candidates;
        for i in 1..remaining_depth {
            max_discrepancy += nb_candidates.saturating_sub(i);
        }

        // With depth-weighted discrepancy, at depth d, each feature i contributes i * (d + 1)
        // Maximum discrepancy at depth d is when exploring all features: sum(i * (d + 1) for i in 0..nb_candidates)
        // This equals (nb_candidates - 1) * nb_candidates / 2 * (d + 1)
        // let mut max_discrepancy = 0;
        // for depth in 0..remaining_depth {
        //     let depth_weight = depth + 1;
        //     // Maximum contribution at this depth: (0 + 1 + 2 + ... + (nb_candidates-1)) * depth_weight
        //     // = nb_candidates * (nb_candidates - 1) / 2 * depth_weight
        //     max_discrepancy += (nb_candidates.saturating_sub(1)) * nb_candidates / 2 * depth_weight;
        // }


        max_discrepancy
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

    pub fn get_solution_tree(&mut self) -> Tree {
        let mut solution = Tree::new();
        if let Some(root) = self.cache.root() {
            if let Some(tree_idx) = root.tree_idx {
                solution = self.cache.get_tree(tree_idx).unwrap().clone();
            } else {
                let infos = self.create_solution_tree_entry(root);
                let root = solution.add_root(TreeNode::new(infos));
                self.build_tree_recursion(&mut solution, root, self.cache.root_index());
            }
        }
        solution
    }

    fn create_solution_tree_entry(&self, cache_entry: &Entry) -> NodeInfos {
        let mut infos = NodeInfos {
            feature: Some(cache_entry.feature),
            split: Some(cache_entry.split),
            error: cache_entry.error,
            label: Some(cache_entry.label),
        };

        infos
    }

    fn build_tree_recursion(&self, solution: &mut Tree, parent: usize, cache_index: usize) {
        let branches = self.cache.get_children(cache_index);
        for (branch, &child_index) in branches.iter().enumerate() {
            if child_index > 0 {
                if let Some(entry) = self.cache.get(child_index) {
                    if let Some(tree_idx) = entry.tree_idx {
                        if let Some(sub_tree) = self.cache.get_tree(tree_idx) {
                            let infos = sub_tree.root_details();
                            let solution_child_index = solution.add_node(parent, branch==0, TreeNode::new(infos));
                            solution.update_subtree(solution_child_index, sub_tree, sub_tree.get_root_index());
                        }
                    } else {
                        let infos = self.create_solution_tree_entry(entry);
                        let solution_child_index = solution.add_node(parent, branch==0, TreeNode::new(infos));
                        if !entry.is_leaf {
                            self.build_tree_recursion(solution, solution_child_index, child_index);
                        }
                    }

                }
            }
        }
    }


}

#[cfg(test)]
mod contree_test {
    use crate::common::PointSelector;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;
    use std::path::Path;
    use crate::algorithms::ConTreeLds;

    #[test]
    fn test_run() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let path = Path::new("test_data/avila.txt");
        let mut dataset = reader.read_file(path)?;
        dataset.sort_features();

        let mut contree: ConTreeLds<true> =
            ConTreeLds::new(1, 3, 100.0, usize::MAX, PointSelector::Mid, 0, false, true);

        contree.fit(&dataset);

        println!("Cache Size : {:?}", contree.cache.len());
        println!("Stats : {:#?}", contree.statistics);

        Ok(())
    }
}

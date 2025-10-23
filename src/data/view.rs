use crate::bitsets::{BitCollection, Bitset, BitsetInit};
use crate::data::{Dataset, Feature};

#[derive(Clone, Debug)]
struct HeuristicValues {
    // For each feature, stores split indices sorted by gini value (best first)
    // split_index corresponds to indices in possible_split_indices
    gini_per_split: Vec<Vec<usize>>,
    // Cached best (min) Gini per feature for quick access
    best_gini_per_feature: Vec<(f64, usize)>, // (gini, feature_index)
}
impl HeuristicValues {
    pub fn new(num_features: usize) -> Self {
        Self {
            gini_per_split: vec![Vec::new(); num_features],
            best_gini_per_feature: (0..num_features).map(|i| (1.0, i)).collect(),
        }
    }

    /// Set sorted split indices and best gini for a specific feature
    /// split_indices should be sorted by gini score (best first)
    pub fn set_feature_ginis(&mut self, feature: usize, split_indices: Vec<usize>, best_gini: f64) {
        if feature < self.gini_per_split.len() {
            self.gini_per_split[feature] = split_indices;
            self.best_gini_per_feature[feature] = (best_gini, feature);
        }
    }

    /// Get sorted split indices for a feature (best splits first)
    pub fn feature_split_indices(&self, feature: usize) -> &[usize] {
        if feature < self.gini_per_split.len() {
            &self.gini_per_split[feature]
        } else {
            &[]
        }
    }

    /// Get the best Gini score for a specific feature
    pub fn best_score(&self, feature: usize) -> f64 {
        if feature < self.best_gini_per_feature.len() {
            self.best_gini_per_feature[feature].0
        } else {
            1.0
        }
    }

    /// Get the feature with the best (lowest) Gini score
    pub fn best_feature(&self) -> (usize, f64) {
        self.best_gini_per_feature
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(gini, feat)| (feat, gini))
            .unwrap_or((0, 1.0))
    }

    /// Sort features by best Gini index (best features first)
    pub fn sort_by_gini(&mut self) {
        self.best_gini_per_feature
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Get the sorted feature order (returns feature indices in order of quality)
    pub fn sorted_features(&self) -> impl Iterator<Item = usize> + '_ {
        self.best_gini_per_feature.iter().map(|&(_, feat)| feat)
    }

    /// Get number of features
    pub fn len(&self) -> usize {
        self.gini_per_split.len()
    }
}

pub struct DataView<'a> {
    pub dataset: &'a Dataset,
    pub total_instances: usize,                  // universe size
    pub feature_columns: Vec<Vec<usize>>,        // positions into dataset.features[f]
    pub possible_split_indices: Vec<Vec<usize>>, // per-feature value-change boundaries for this view
    pub label_freq: Vec<usize>,                  // per-class histogram for this view
    pub sort_by_heuristic: bool,
    pub heuristic_values: HeuristicValues, // Gini-based feature ordering with per-split values
    pub bitset: Bitset,                    // identity (filled on demand if you prefer)
}

impl<'a> DataView<'a> {
    pub fn root(dataset: &'a Dataset, sort_by_heuristic: bool) -> Self {
        let total_instances = dataset.count();
        let mut label_freq = vec![0; dataset.num_labels()];
        let mut feature_columns = Vec::new();
        let mut possible_split_indices = Vec::new();

        for (feature_idx, feature) in dataset.into_iter().enumerate() {
            let feature_len = feature.len();

            let mut splits = Vec::new();
            let mut last_unique_index: Option<usize> = None;

            for pos in 0..feature_len {
                let el = &feature[pos];

                if feature_idx == 0 {
                    label_freq[el.label as usize] += 1;
                }

                if let Some(last) = last_unique_index {
                    if el.unique_value_idx != last {
                        splits.push(pos);
                    }
                }
                last_unique_index = Some(el.unique_value_idx);
            }
            possible_split_indices.push(splits);
            feature_columns.push((0..feature_len).collect::<Vec<usize>>());
        }

        // Initialize heuristic values
        let mut heuristic_values = HeuristicValues::new(dataset.num_features());

        // Compute Gini index for each feature if sort_by_heuristic is enabled
        if sort_by_heuristic {
            for feature_idx in 0..dataset.num_features() {
                let feature = &dataset[feature_idx];
                let idx = &feature_columns[feature_idx];
                let (ordered_index, gini) = Self::compute_gini_for_all_splits(
                    feature,
                    idx,
                    &possible_split_indices[feature_idx],
                    &label_freq,
                    dataset.num_labels(),
                );
              
                heuristic_values.set_feature_ginis(feature_idx, ordered_index, gini);
            }
            // Sort features by Gini index
            heuristic_values.sort_by_gini();
        }


        let mut bitset = Bitset::new(BitsetInit::Full(total_instances));
        bitset.save_count();

        Self {
            dataset,
            total_instances,
            feature_columns,
            possible_split_indices,
            label_freq,
            sort_by_heuristic,
            heuristic_values,
            bitset,
        }
    }

    pub fn get_dataset_size(&self) -> usize {
        self.feature_columns[0].len()
    }

    pub fn get_feature_number(&self) -> usize {
        self.dataset.num_features()
    }

    pub fn get_sorted_feature(&self, f: usize) -> &Feature {
        &self.dataset[f]
    }

    pub fn get_feature_indices(&self, f: usize) -> &[usize] {
        &self.feature_columns[f]
    }

    pub fn get_labels_freqs(&self) -> &[usize] {
        &self.label_freq
    }

    pub fn get_num_labels(&self) -> usize {
        self.dataset.num_labels()
    }

    pub fn get_possible_split_indices(&self, f: usize) -> &[usize] {
        &self.possible_split_indices[f]
    }

    fn recompute_split_indices_all(&mut self) {
        for f in 0..self.get_feature_number() {
            self.possible_split_indices[f] =
                Self::compute_split_indices_for(&self.dataset[f], &self.feature_columns[f]);
        }
    }

    pub fn ordered_possible_splits(&self, feature: usize) -> &[usize] {
        &self.heuristic_values.gini_per_split[feature]
    }

    pub fn features_best_score(&self) -> &[(f64, usize)] {
        &self.heuristic_values.best_gini_per_feature
    }

    #[inline]
    fn compute_split_indices_for(col: &Feature, idxs: &[usize]) -> Vec<usize> {
        if idxs.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(idxs.len() / 10);
        let mut last = None::<usize>;

        for (i, &pos) in idxs.iter().enumerate() {
            let cur = col[pos].unique_value_idx;
            if i > 0 && Some(cur) != last {
                out.push(i);
            }
            last = Some(cur);
        }

        out.shrink_to_fit();
        out
    }

    pub fn initialize_split_parameters(
        &self,
        feature_index: usize,
        split_point: usize,
        left_freq: &mut Vec<usize>,
        right_freq: &mut Vec<usize>
    ) {
        let num_labels = self.dataset.num_labels();

        let total_size = self.get_dataset_size();
        let feature_ids = self.get_feature_indices(feature_index);
        let feature = self.get_sorted_feature(feature_index);



        // Count the smaller side, derive the other by subtraction
        if split_point < total_size - split_point {
            // Left is smaller: count left, derive right
            for i in 0..split_point {
                let data_point = &feature[feature_ids[i]];
                left_freq[data_point.label as usize] += 1;
            }
            for label in 0..num_labels {
                right_freq[label] = self.label_freq[label] - left_freq[label];
            }
        } else {
            // Right is smaller: count right, derive left
            for i in split_point..total_size {
                let data_point = &feature[feature_ids[i]];
                right_freq[data_point.label as usize] += 1;
            }
            for label in 0..num_labels {
                left_freq[label] = self.label_freq[label] - right_freq[label];
            }
        }
        
    }

    /// Compute Gini index for all possible split points of a feature
    /// Returns (sorted_split_indices, best_gini) where split indices are sorted by gini (best first)
    /// OPTIMIZED: Only computes at valid split boundaries, not at every element
    fn compute_gini_for_all_splits(
        feature: &Feature,
        idx: &[usize],
        possible_splits: &[usize],
        label_freq: &[usize],
        num_labels: usize,
    ) -> (Vec<usize>, f64) {
        if feature.is_empty() || possible_splits.is_empty() {
            return (Vec::new(), 1.0);
        }

        let mut gini_values = Vec::with_capacity(possible_splits.len());
        let mut left_label_freq = vec![0usize; num_labels];
        let mut right_label_freq = label_freq.to_vec();
        let mut best_gini = 1.0;

        let mut last_pos = 0;

        // Only compute Gini at valid split boundaries
        for (split_idx, &split_pos) in possible_splits.iter().enumerate() {
            // Update frequencies up to this split point
            for i in last_pos..split_pos {
                let data_idx = idx[i];
                let label = feature[data_idx].label as usize;
                right_label_freq[label] -= 1;
                left_label_freq[label] += 1;
            }

            let left_count = split_pos;
            let right_count = idx.len() - left_count;

            let gini = Self::compute_gini_from_frequencies(
                &left_label_freq,
                &right_label_freq,
                left_count,
                right_count,
            );

            // Track best gini
            if gini < best_gini {
                best_gini = gini;
            }

            gini_values.push((split_pos, gini));
            last_pos = split_pos;
        }

        // Sort by gini value (ascending) - best (lowest) gini first
        gini_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Extract just the split indices (already sorted by gini)
        let sorted_indices: Vec<usize> = gini_values.into_iter().map(|(idx, _)| idx).collect();

        (sorted_indices, best_gini)
    }

    /// Compute weighted Gini index from label frequency distributions
    #[inline]
    fn compute_gini_from_frequencies(
        left_freq: &[usize],
        right_freq: &[usize],
        left_count: usize,
        right_count: usize,
    ) -> f64 {
        let mut left_gini = 1.0;
        let mut right_gini = 1.0;

        if left_count > 0 {
            for &freq in left_freq {
                let prob = freq as f64 / left_count as f64;
                left_gini -= prob * prob;
            }
        }

        if right_count > 0 {
            for &freq in right_freq {
                let prob = freq as f64 / right_count as f64;
                right_gini -= prob * prob;
            }
        }

        let total = left_count + right_count;
        if total > 0 {
            (left_gini * left_count as f64 + right_gini * right_count as f64) / total as f64
        } else {
            1.0
        }
    }

    pub fn split(&self, sf: usize, split_point: usize) -> (Self, Self) {
        let col = &self.dataset[sf];
        let sf_idxs = &self.feature_columns[sf];

        let (sf_left_idxs, sf_right_idxs) = sf_idxs.split_at(split_point);

        let mut left_bitset = Bitset::new(BitsetInit::Empty(self.total_instances));
        let mut left_label_freq = vec![0; self.dataset.num_labels()];

        for &pos in sf_left_idxs {
            let el = &col[pos];
            left_bitset.set(el.tid);
            left_label_freq[el.label as usize] += 1;
        }
        left_bitset.save_count();

        // Derive right histogram by subtraction
        let mut right_label_freq = vec![0; self.dataset.num_labels()];
        for label in 0..self.dataset.num_labels() {
            right_label_freq[label] = self.label_freq[label] - left_label_freq[label];
        }

        let num_features = self.get_feature_number();
        let left_size_estimate = split_point;
        let right_size_estimate = self.get_dataset_size() - split_point;

        let mut left_pfi: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(left_size_estimate))
            .collect();
        let mut right_pfi: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(right_size_estimate))
            .collect();

        // Pre-allocate split indices vectors
        let mut left_split_indices: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(left_size_estimate / 10))
            .collect();
        let mut right_split_indices: Vec<Vec<usize>> = (0..num_features)
            .map(|_| Vec::with_capacity(right_size_estimate / 10))
            .collect();

        // Initialize Gini computation structures
        let mut left_heuristics = HeuristicValues::new(num_features);
        let mut right_heuristics = HeuristicValues::new(num_features);

        for f in 0..num_features {
            if f == sf {
                // Split feature already partitioned
                left_pfi[f] = sf_left_idxs.to_vec();
                right_pfi[f] = sf_right_idxs.to_vec();

                // Compute split indices for the split feature
                left_split_indices[f] =
                    Self::compute_split_indices_for(&self.dataset[f], sf_left_idxs);
                right_split_indices[f] =
                    Self::compute_split_indices_for(&self.dataset[f], sf_right_idxs);

                // Compute Gini for split feature if needed
                if self.sort_by_heuristic {
                    let (ordered_splits, best_gini) = Self::compute_gini_for_all_splits(
                        &self.dataset[f],
                        &left_pfi[f],
                        &left_split_indices[f],
                        &left_label_freq,
                        self.dataset.num_labels(),
                    );
                    left_heuristics.set_feature_ginis(f, ordered_splits, best_gini);

                    let  (ordered_splits, best_gini) = Self::compute_gini_for_all_splits(
                        &self.dataset[f],
                        &right_pfi[f],
                        &right_split_indices[f],
                        &right_label_freq,
                        self.dataset.num_labels(),
                    );
                    right_heuristics.set_feature_ginis(f, ordered_splits, best_gini);
                }
                continue;
            }

            let fcol = &self.dataset[f];
            let parent_idxs = &self.feature_columns[f];

            let mut left_last_unique: Option<usize> = None;
            let mut right_last_unique: Option<usize> = None;
            let mut left_counter = 0;
            let mut right_counter = 0;

            // For incremental Gini computation
            let num_labels = self.dataset.num_labels();
            let mut left_gini_values = Vec::new();
            let mut right_gini_values = Vec::new();
            let mut left_best_gini = 1.0;
            let mut right_best_gini = 1.0;

            let mut left_label_running = vec![0usize; num_labels];
            let mut right_label_running = vec![0usize; num_labels];
            let mut left_label_remaining = left_label_freq.clone();
            let mut right_label_remaining = right_label_freq.clone();

            for &pos in parent_idxs {
                let el = &fcol[pos];
                let row = el.tid;
                let label = el.label as usize;

                if left_bitset.contains(row) {
                    left_pfi[f].push(pos);

                    // Track split indices during partitioning
                    if let Some(last) = left_last_unique {
                        if el.unique_value_idx != last {
                            left_split_indices[f].push(left_counter);

                            // Compute Gini at this split boundary if enabled
                            if self.sort_by_heuristic {
                                let gini = Self::compute_gini_from_frequencies(
                                    &left_label_running,
                                    &left_label_remaining,
                                    left_counter,
                                    left_size_estimate - left_counter,
                                );
                                if gini < left_best_gini {
                                    left_best_gini = gini;
                                }
                                left_gini_values.push((left_split_indices[f].len() - 1, gini));
                            }
                        }
                    }
                    left_last_unique = Some(el.unique_value_idx);
                    left_counter += 1;

                    // Update running frequencies for Gini computation
                    if self.sort_by_heuristic {
                        left_label_remaining[label] -= 1;
                        left_label_running[label] += 1;
                    }
                } else {
                    right_pfi[f].push(pos);

                    // Track split indices during partitioning
                    if let Some(last) = right_last_unique {
                        if el.unique_value_idx != last {
                            right_split_indices[f].push(right_counter);

                            // Compute Gini at this split boundary if enabled
                            if self.sort_by_heuristic {
                                let gini = Self::compute_gini_from_frequencies(
                                    &right_label_running,
                                    &right_label_remaining,
                                    right_counter,
                                    right_size_estimate - right_counter,
                                );
                                if gini < right_best_gini {
                                    right_best_gini = gini;
                                }
                                right_gini_values.push((right_split_indices[f].len() - 1, gini));
                            }
                        }
                    }
                    right_last_unique = Some(el.unique_value_idx);
                    right_counter += 1;

                    // Update running frequencies for Gini computation
                    if self.sort_by_heuristic {
                        right_label_remaining[label] -= 1;
                        right_label_running[label] += 1;
                    }
                }
            }

            // Store computed Gini values
            if self.sort_by_heuristic {
                // Sort by gini value and extract indices
                left_gini_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                right_gini_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let left_sorted_indices: Vec<usize> = left_gini_values.into_iter().map(|(idx, _)| idx).collect();
                let right_sorted_indices: Vec<usize> = right_gini_values.into_iter().map(|(idx, _)| idx).collect();
                
                left_heuristics.set_feature_ginis(f, left_sorted_indices, left_best_gini);
                right_heuristics.set_feature_ginis(f, right_sorted_indices, right_best_gini);
            }

            // Shrink to fit to save memory
            left_split_indices[f].shrink_to_fit();
            right_split_indices[f].shrink_to_fit();
        }

        let mut right_bitset = self.bitset.intersect_with(&left_bitset, true);
        right_bitset.save_count();

        // Sort features by Gini index if enabled (Gini already computed during partitioning)
        if self.sort_by_heuristic {
            left_heuristics.sort_by_gini();
            right_heuristics.sort_by_gini();
        }

        let left = Self {
            dataset: self.dataset,
            total_instances: self.total_instances,
            feature_columns: left_pfi,
            possible_split_indices: left_split_indices,
            label_freq: left_label_freq,
            sort_by_heuristic: self.sort_by_heuristic,
            heuristic_values: left_heuristics,
            bitset: left_bitset,
        };

        let right = Self {
            dataset: self.dataset,
            total_instances: self.total_instances,
            feature_columns: right_pfi,
            possible_split_indices: right_split_indices,
            label_freq: right_label_freq,
            sort_by_heuristic: self.sort_by_heuristic,
            heuristic_values: right_heuristics,
            bitset: right_bitset,
        };

        (left, right)
    }

    pub fn len(&self) -> usize {
        debug_assert!(
            self.bitset.count() == self.feature_columns[0].len(),
            "Mismatched size in bitset and feature columns"
        );
        self.feature_columns[0].len()
    }
}

#[cfg(test)]
mod data_view_tests {
    use crate::bitsets::BitCollection;
    use crate::data::view::DataView;
    use crate::reader::data_reader::DataReader;
    use crate::reader::DataReaderError;
    use std::path::Path;

    #[test]
    fn test_small() -> Result<(), DataReaderError> {
        let reader = DataReader::default();
        let path = Path::new("test_data/anneal.txt");
        let mut dataset = reader.read_file(path)?;
        dataset.sort_features();

        println!("{:?}", dataset.num_features());
        println!("{:?}", dataset.count());
        println!("{:?}", dataset.num_labels());

        let view = DataView::root(&dataset, true);
        println!(
            "Possible split {:?}",
            view.heuristic_values.best_gini_per_feature
        );
        println!("{:?}", view.get_dataset_size());
        println!("Before label freq {:?}", view.get_labels_freqs());

        let (left, right) = view.split(4, 556);
        println!("left {}", left.bitset.count());
        println!("left {:?}", left.possible_split_indices);
        let (ll, lr) = left.split(9, 546);
        println!("ll {:?}", ll.heuristic_values.best_gini_per_feature);

        // println!("Right {}", right.bitset.count());
        // println!("l {:?}", left.heuristic_values.gini_per_split[0].len());
        // println!("l {:?}", left.possible_split_indices[0].len());
        //
        // let (l, r ) = left.split(3, 900);
        //
        // println!("left {}", l.bitset.count());
        // println!("Right {}", r.bitset.count());
        // println!("l {:?}", l.heuristic_values.gini_per_split[0].len());
        // println!("l {:?}", l.possible_split_indices[0].len());

        // println!("l {:?}", left.get_labels_freqs());
        // println!("r {:?}", right.get_labels_freqs());

        // println!("r {:?} 3, ", right.get_possible_split_indices(3));
        //
        // let (l, r) = right.split(2, 6);
        // println!("l : {}", l.bitset.count());
        // println!("l : {}", l.feature_columns[0].len());
        // let mut i = 0;
        // for feature_col in &dataset {
        //     print!("Feature {} ", i);
        //     for j in 0..l.bitset.count() {
        //         print!(" {:?} ", feature_col[l.feature_columns[i][j]]);
        //     }
        //     println!();
        //     println!();
        //     println!();
        //     i += 1;
        //
        // }

        Ok(())
    }
}

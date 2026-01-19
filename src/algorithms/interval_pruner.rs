use std::collections::HashMap;

/// Bound structure representing an interval with metadata about previous splits
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub left_bound: usize,
    pub right_bound: usize,
    pub last_split_left_index: Option<usize>,
    pub last_split_right_index: Option<usize>,
}

impl Bound {
    pub fn new(
        left: usize,
        right: usize,
        last_left: Option<usize>,
        last_right: Option<usize>,
    ) -> Self {
        Self {
            left_bound: left,
            right_bound: right,
            last_split_left_index: last_left,
            last_split_right_index: last_right,
        }
    }

    /// Check if the interval is valid (left <= right)
    pub fn is_valid(&self) -> bool {
        self.left_bound <= self.right_bound
    }

    /// Mark interval as invalid
    pub fn invalidate(&mut self) {
        self.left_bound = 1;
        self.right_bound = 0;
        self.last_split_left_index = None;
        self.last_split_right_index = None;
    }
}

/// IntervalsPruner performs branch-and-bound pruning on split intervals
/// to reduce the search space when finding optimal decision tree splits.
pub struct IntervalsPruner<'a> {
    possible_split_indexes: &'a [usize],
    possible_split_size: usize,
    pub rightmost_zero_index: Option<usize>,
    pub leftmost_zero_index: Option<usize>,
    max_gap: usize,
    /// Maps split index -> (left_score, right_score)
    evaluated_indices_record: HashMap<usize, (usize, usize)>,
}

impl<'a> IntervalsPruner<'a> {
    /// Creates a new IntervalsPruner
    ///
    /// # Arguments
    /// * `possible_split_indexes` - Reference to vector of possible split indices
    /// * `max_gap` - Maximum allowable gap for suboptimal solutions
    pub fn new(possible_split_indexes: &'a [usize], max_gap: usize) -> Self {
        let possible_split_size = possible_split_indexes.len();
        let mut evaluated_indices_record = HashMap::new();
        evaluated_indices_record.reserve(possible_split_size / 8);

        Self {
            possible_split_indexes,
            possible_split_size,
            rightmost_zero_index: None,
            leftmost_zero_index: None,
            max_gap,
            evaluated_indices_record,
        }
    }

    /// Performs subinterval pruning to determine if an interval can be entirely pruned
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds of the interval being evaluated
    /// * `current_best_score` - The best score obtained so far
    ///
    /// # Returns
    /// `true` if the subinterval can be pruned, `false` otherwise
    pub fn subinterval_pruning(&self, current_bounds: &Bound, current_best_score: usize) -> bool {
        let left_bound_score_left = current_bounds
            .last_split_left_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(left_score, _)| left_score)
            .unwrap_or(0);

        let right_bound_score_right = current_bounds
            .last_split_right_index
            .and_then(|idx| self.evaluated_indices_record.get(&idx))
            .map(|&(_, right_score)| right_score)
            .unwrap_or(0);

        left_bound_score_left + right_bound_score_right + self.max_gap >= current_best_score
    }

    /// Performs interval shrinking by narrowing the bounds based on current best score
    ///
    /// # Arguments
    /// * `current_bounds` - The current bounds to be updated by shrinking
    /// * `current_best_score` - The best score to compare against
    pub fn interval_shrinking(&self, current_bounds: &mut Bound, current_best_score: usize) {
        // Update bounds based on zero indices
        if let Some(leftmost) = self.leftmost_zero_index {
            current_bounds.left_bound = current_bounds.left_bound.max(leftmost + 1);
        }

        if let Some(rightmost) = self.rightmost_zero_index {
            current_bounds.right_bound =
                current_bounds.right_bound.min(rightmost.saturating_sub(1));
        }

        // Shrink from left side
        if let Some(last_left_idx) = current_bounds.last_split_left_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_left_idx)
            {
                let sum = left_score + right_score + self.max_gap;
                if sum >= current_best_score {
                    let updated_score_difference = (sum - current_best_score);
                    let extended_left_bound =
                        self.possible_split_indexes[last_left_idx] + updated_score_difference + 1;

                    if extended_left_bound
                        <= self.possible_split_indexes[current_bounds.right_bound]
                    {
                        let new_left_bound = self.lower_bound(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            extended_left_bound,
                        );
                        current_bounds.left_bound = current_bounds.left_bound.max(new_left_bound);
                    } else {
                        current_bounds.invalidate();
                        return;
                    }
                }
            }
        }

        // Shrink from right side
        if let Some(last_right_idx) = current_bounds.last_split_right_index {
            if let Some(&(left_score, right_score)) =
                self.evaluated_indices_record.get(&last_right_idx)
            {
                let sum = left_score + right_score + self.max_gap;
                if sum >= current_best_score {
                    let updated_score_difference = (sum - current_best_score);
                    let extended_right_bound = self.possible_split_indexes[last_right_idx]
                        .saturating_sub(updated_score_difference + 1);

                    if extended_right_bound
                        >= self.possible_split_indexes[current_bounds.left_bound]
                    {
                        let new_right_bound = self.upper_bound(
                            current_bounds.left_bound,
                            current_bounds.right_bound,
                            extended_right_bound,
                        );
                        current_bounds.right_bound =
                            current_bounds.right_bound.min(new_right_bound);
                    } else {
                        current_bounds.invalidate();
                        return;
                    }
                }
            }
        }
    }

    /// Performs neighborhood pruning by evaluating the interval around a split index
    ///
    /// # Arguments
    /// * `score_difference` - The difference in scores used to determine pruning
    /// * `left` - The left boundary of the interval
    /// * `right` - The right boundary of the interval
    /// * `split_index` - The index at which the split is evaluated
    ///
    /// # Returns
    /// A tuple `(new_left_bound, new_right_bound)` representing the pruned interval
    pub fn neighbourhood_pruning(
        &self,
        score_difference: usize,
        left: usize,
        right: usize,
        split_index: usize,
    ) -> (usize, usize) {
        let score_difference = score_difference + self.max_gap;

        if score_difference == 0 {
            return (split_index + 1, split_index);
        }

        // Calculate new left bound
        let mut new_bound_left = split_index + 1;
        if let Some(leftmost) = self.leftmost_zero_index {
            new_bound_left = new_bound_left.max(leftmost + 1);
        }

        let minimum_right_value = self.possible_split_indexes[split_index] + score_difference + 1;

        if minimum_right_value < self.possible_split_indexes[right] {
            new_bound_left = self.lower_bound(new_bound_left, right, minimum_right_value);
        } else {
            new_bound_left = right + 1;
        }

        // Calculate new right bound
        let mut new_bound_right = split_index.saturating_sub(1);
        if let Some(rightmost) = self.rightmost_zero_index {
            new_bound_right = new_bound_right.min(rightmost.saturating_sub(1));
        }

        let minimum_left_value =
            self.possible_split_indexes[split_index].saturating_sub(score_difference + 1);

        if minimum_left_value > self.possible_split_indexes[left] {
            new_bound_right = self.upper_bound(left, new_bound_right, minimum_left_value);
        } else {
            new_bound_right = left.saturating_sub(1);
        }

        (new_bound_left, new_bound_right)
    }

    pub fn add_result(&mut self, index: usize, mut left_score: usize, mut right_score: usize) {
        if left_score == 0 {
            self.leftmost_zero_index = Some(
                self.leftmost_zero_index
                    .map(|v| v.max(index))
                    .unwrap_or(index),
            );
        }

        if right_score == 0 {
            self.rightmost_zero_index = Some(
                self.rightmost_zero_index
                    .map(|v| v.min(index))
                    .unwrap_or(index),
            );
        }

        if left_score == usize::MAX {
            left_score = 0;
        }

        if right_score == usize::MAX {
            right_score = 0;
        }

        self.evaluated_indices_record
            .insert(index, (left_score, right_score));
    }

    /// Helper: Find first element >= value (like std::lower_bound)
    fn lower_bound(&self, left: usize, mut right: usize, value: usize) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        match slice.binary_search(&value) {
            Ok(pos) => left + pos,
            Err(pos) => left + pos,
        }
    }

    /// Helper: Find first element > value (like std::upper_bound)
    fn upper_bound(&self, left: usize, mut right: usize, value: usize) -> usize {
        if left >= right {
            right = self.possible_split_indexes.len() - 1
        }
        let slice = &self.possible_split_indexes[left..=right];
        let pos = slice.partition_point(|&x| x <= value);
        left + pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruner_creation() {
        let splits = vec![10, 20, 30, 40, 50];
        let pruner = IntervalsPruner::new(&splits, 2);

        assert_eq!(pruner.possible_split_size, 5);
        assert_eq!(pruner.max_gap, 2);
        assert_eq!(pruner.leftmost_zero_index, None);
        assert_eq!(pruner.rightmost_zero_index, None);
    }

    #[test]
    fn test_bound_creation() {
        let bound = Bound::new(0, 10, None, None);
        assert_eq!(bound.left_bound, 0);
        assert_eq!(bound.right_bound, 10);
        assert_eq!(bound.last_split_left_index, None);
        assert_eq!(bound.last_split_right_index, None);
        assert!(bound.is_valid());
    }

    #[test]
    fn test_bound_invalidation() {
        let mut bound = Bound::new(0, 10, Some(5), Some(7));
        assert!(bound.is_valid());

        bound.invalidate();
        assert!(!bound.is_valid());
        assert_eq!(bound.last_split_left_index, None);
        assert_eq!(bound.last_split_right_index, None);
    }
}

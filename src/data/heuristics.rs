/// Trait for computing split heuristics incrementally
pub trait SplitHeuristic {
    /// Create a new instance for a child view
    fn new_for_child(num_labels: usize, child_label_freq: &[usize]) -> Self;

    /// Update state when adding an element and return the heuristic at this position
    fn on_element_added(&mut self, label: usize, position: usize, total_size: usize)
        -> Option<f32>;

    /// Get the best (minimum) heuristic score seen
    fn best_score(&self) -> f32;

    /// Get all heuristic values computed (position, score)
    fn all_scores(&self) -> &[(usize, f32)];
}

/// Gini impurity heuristic with full history
pub struct GiniHeuristic {
    num_labels: usize,
    left_freq: Vec<usize>,
    right_freq: Vec<usize>,
    best_gini: f32,
    scores: Vec<(usize, f32)>, // (position, gini_value)
}

impl SplitHeuristic for GiniHeuristic {
    fn new_for_child(num_labels: usize, child_label_freq: &[usize]) -> Self {
        Self {
            num_labels,
            left_freq: vec![0; num_labels],
            right_freq: child_label_freq.to_vec(),
            best_gini: 1.0,
            scores: Vec::new(),
        }
    }

    fn on_element_added(
        &mut self,
        label: usize,
        position: usize,
        total_size: usize,
    ) -> Option<f32> {
        // Move element from right to left
        self.right_freq[label] -= 1;
        self.left_freq[label] += 1;

        let left_count = position + 1;
        let right_count = total_size - left_count;

        // Can't split after the last element
        if right_count == 0 {
            return None;
        }

        let mut left_gini = 1.0f32;
        let mut right_gini = 1.0f32;

        for l in 0..self.num_labels {
            if left_count > 0 {
                let left_prob = self.left_freq[l] as f32 / left_count as f32;
                left_gini -= left_prob * left_prob;
            }
            if right_count > 0 {
                let right_prob = self.right_freq[l] as f32 / right_count as f32;
                right_gini -= right_prob * right_prob;
            }
        }

        let gini =
            (left_gini * left_count as f32 + right_gini * right_count as f32) / total_size as f32;

        // Store this score
        self.scores.push((position, gini));

        // Update best
        if gini < self.best_gini {
            self.best_gini = gini;
        }

        Some(gini)
    }

    fn best_score(&self) -> f32 {
        self.best_gini
    }

    fn all_scores(&self) -> &[(usize, f32)] {
        &self.scores
    }
}

/// No-op heuristic
pub struct NoHeuristic;

impl SplitHeuristic for NoHeuristic {
    fn new_for_child(_num_labels: usize, _child_label_freq: &[usize]) -> Self {
        Self
    }

    fn on_element_added(
        &mut self,
        _label: usize,
        _position: usize,
        _total_size: usize,
    ) -> Option<f32> {
        None
    }

    fn best_score(&self) -> f32 {
        1.0
    }

    fn all_scores(&self) -> &[(usize, f32)] {
        &[]
    }
}

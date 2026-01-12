pub mod budget_iterator;

use clap::ValueEnum;

#[derive(Default, Copy, Debug, Clone, PartialOrd, PartialEq, ValueEnum)]
pub enum PointSelector {
    #[default]
    Mid,
    First,
    Random,
}
#[derive(Copy, Clone, Debug)]
pub struct SearchConfig {
    pub max_depth: usize,
    pub min_sup: usize,
    pub max_time: f64,
    pub max_gap: usize,
    pub max_error: usize,
    pub is_root: bool,
    pub use_heuristic: bool,
    pub use_discrepancy: bool,
    pub fast_d2: bool,
    pub point_selector: PointSelector,
    pub nb_runs: usize,
    pub discrepancy: usize,
    pub budget: usize,
}

impl SearchConfig {
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_gap: usize,
        max_error: usize,
        use_heuristic: bool,
        fast_d2: bool,
        split_strategy: PointSelector,
    ) -> Self {
        Self {
            max_depth,
            min_sup,
            max_time,
            max_gap,
            max_error,
            is_root: true,
            use_heuristic,
            use_discrepancy: false,
            fast_d2,
            point_selector: split_strategy,
            nb_runs: 0,
            discrepancy: 0,
            budget: 0,
        }
    }

    pub fn derive_left(&self) -> Self {
        let mut left_config = *self;
        left_config.max_depth -= 1;
        left_config.is_root = false;
        left_config.max_gap = (self.max_gap - (self.max_gap + 1) / 2) / 2;
        left_config
    }

    pub fn derive_right(&self, left_gap: usize) -> Self {
        let mut right_config = *self;
        right_config.max_depth -= 1;
        right_config.is_root = false;
        right_config.max_gap = (self.max_gap - (self.max_gap + 1) / 2).saturating_sub(left_gap);
        right_config
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Statistics {
    pub cache_size: usize,
    pub cache_hits: usize,
    pub general_solver_call: usize,
    pub specialized_solver_call: usize,
    pub num_samples: usize,
    pub num_features: usize,
    pub error: usize,
    pub duration: f64,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            cache_size: 0,
            cache_hits: 0,
            general_solver_call: 0,
            specialized_solver_call: 0,
            num_samples: 0,
            num_features: 0,
            error: 0,
            duration: 0.0,
        }
    }
}

pub fn classification_error(classes_support: &[usize]) -> (usize, usize) {
    let mut max_idx = 0;
    let mut max_value = 0;
    let mut total = 0;
    for (idx, value) in classes_support.iter().enumerate() {
        total += value;
        if *value >= max_value {
            max_value = *value;
            max_idx = idx;
        }
    }
    let error = total - max_value;
    (error, max_idx)
}

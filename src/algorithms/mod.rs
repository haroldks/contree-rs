mod continuous_tree;
mod contree_lds;
mod depth2;
mod interval_pruner;
mod contree_lds_prune;

use crate::common::{PointSelector, Statistics};
pub use continuous_tree::ConTree;
pub use contree_lds::ConTreeLds;
use crate::tree::Tree;
use crate::data::Dataset;
use crate::data::view::DataView;

pub enum GenericConTree<const USE_CACHE: bool> {
    Normal(ConTree<USE_CACHE>),
    LDS(ConTreeLds<USE_CACHE>),
}

impl<const USE_CACHE: bool> GenericConTree<USE_CACHE> {
    pub fn new(
        min_sup: usize,
        max_depth: usize,
        max_time: f64,
        max_error: usize,
        split_selection_strategy: PointSelector,
        max_gap: usize,
        use_heuristic: bool,
        fast_d2: bool,
        use_lds: bool,
    ) -> Self {
        match use_lds {
            true => Self::LDS(ConTreeLds::new(
                min_sup,
                max_depth,
                max_time,
                max_error,
                split_selection_strategy,
                max_gap,
                use_heuristic,
                fast_d2,
            )),
            false => Self::Normal(ConTree::new(
                min_sup,
                max_depth,
                max_time,
                max_error,
                split_selection_strategy,
                max_gap,
                use_heuristic,
                fast_d2,
            )),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        match self {
            GenericConTree::Normal(solver) => solver.fit(dataset),
            GenericConTree::LDS(solver) => solver.fit(dataset)
        }
    }
    
    
    pub fn partial_fit(&mut self, dataset: &DataView) -> bool {
        match self {
            GenericConTree::Normal(_) => true,
            GenericConTree::LDS(solver) => solver.partial_fit(dataset)
        }
    }
    
    pub fn stats(&self) -> Statistics {
        match self { 
            GenericConTree::Normal(solver) => solver.statistics(),
            GenericConTree::LDS(solver) => *solver.statistics()
        }
    }

    pub fn tree(&mut self) -> Tree {
        match self {
            GenericConTree::Normal(solver) => Tree::new(),
            GenericConTree::LDS(solver) => solver.get_solution_tree()
        }
    }


}

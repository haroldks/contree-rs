use std::path::PathBuf;
use clap::Parser;
use crate::common::PointSelector;

#[derive(Debug, Parser)]
#[clap(name = "con-tree", version, author, about)]
pub struct GeneralParser {
    /// Dataset input file path
    #[clap(short, long, value_parser)]
    pub input: PathBuf,

    /// Minimum support
    #[arg(short, long, default_value_t = 1)]
    pub support: usize,

    /// Maximum depth
    #[arg(short, long)]
    pub depth: usize,

    /// Maximum error allowed
    #[arg(long, default_value_t = usize::MAX)]
    pub max_error: usize,

    /// Maximum error gap allowed
    #[arg(long, default_value_t = 0)]
    pub max_gap: usize,

    /// Maximum execution time allowed
    #[arg(short, long, default_value_t = 600.0)]
    pub time_limit: f64,

    /// Sort split and feature using gini index
    #[arg(long, default_value_t = false)]
    pub sort_by_heuristic: bool,

    /// Split selection strategy to use in the search
    #[arg(long, value_enum, default_value_t = PointSelector::Mid)]
    pub split_selection_strategy: PointSelector,

    /// Use specialized algorithm for depth 2 tree
    #[arg(short, long, default_value_t = false)]
    pub fast_d2: bool,

    /// Printing Statistics and Constraints
    #[arg(long, default_value_t = false)]
    pub print_stats: bool,

    /// Printing Tree
    #[arg(long, default_value_t = false)]
    pub(crate) print_tree: bool,

}

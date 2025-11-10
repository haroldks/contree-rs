use crate::algorithms::{ConTree, ConTreeLds, GenericConTree};
use crate::common::PointSelector;
use crate::data::view::DataView;
use crate::parsers::GeneralParser;
use crate::reader::data_reader::DataReader;
use crate::reader::DataReaderError;
use clap::Parser;

mod algorithms;
mod bitsets;
mod caching;
mod common;
mod cover;
mod data;
mod globals;
mod parsers;
mod reader;
mod tree;

fn main() -> Result<(), DataReaderError> {
    coz::thread_init();
    let app = GeneralParser::parse();

    if !app.input.exists() {
        panic!("File does not exist");
    }

    let reader = DataReader::default();
    let mut dataset = reader.read_file(&app.input)?;
    dataset.sort_features();
    let mut solver: GenericConTree<true> = GenericConTree::from(&app);
    solver.fit(&dataset);
    if app.print_stats {
        println!("{:#?}", solver.stats());
    }

    Ok(())
}

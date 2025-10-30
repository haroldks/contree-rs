use clap::Parser;
use crate::algorithms::{ConTree, ConTreeLds};
use crate::common::PointSelector;
use crate::data::view::DataView;
use crate::parsers::GeneralParser;
use crate::reader::data_reader::DataReader;
use crate::reader::DataReaderError;

mod algorithms;
mod bitsets;
mod caching;
mod common;
mod cover;
mod data;
mod globals;
mod reader;
mod tree;
mod parsers;

fn main() -> Result<(), DataReaderError>{
    coz::thread_init();
    let app = GeneralParser::parse();
    if !app.input.exists() {
        panic!("File does not exist");
    }

    let reader = DataReader::default();
    let mut dataset = reader.read_file(&app.input)?;
    dataset.sort_features();

    // let mut search:  ConTree<true> = ConTree::new(
    //     app.support,
    //     app.depth,
    //     app.time_limit,
    //     app.max_error,
    //     PointSelector::Mid,
    //     app.max_gap,
    //     app.sort_by_heuristic,
    //     app.fast_d2
    // );

    let mut search:  ConTreeLds<true> = ConTreeLds::new(
        app.support,
        app.depth,
        app.time_limit,
        app.max_error,
        PointSelector::Mid,
        app.max_gap,
        app.sort_by_heuristic,
        app.fast_d2
    );



    search.fit(&dataset);
     let view = DataView::root(&dataset, app.sort_by_heuristic);
     for _ in 0..28 {
         search.partial_fit(&view)
     }


    if app.print_stats {
        println!("{:#?}", search.statistics());
    }


    Ok(())
}

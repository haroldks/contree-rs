## Installation and Execution
This project is implemented in Rust, and cargo is required to build and run it. Clone and build:

```bash
cargo build --release
```

## Running

```bash
cargo run --release --example lds -- -i datasets/avila.txt 
-d 4 
-s 1 
--fast-d2 
--split-selection-strategy first 
--use-lds --result-dir . 
--overwrite 
--sort-by-heuristic
```

## Benchmarking

Benchmarks for anytime performance and cross-validation are generated using 
`experiments/benchmarks.sh` and `experiments/crossval.sh`.

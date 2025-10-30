mod algorithms;
mod bitsets;
mod caching;
mod common;
mod data;
mod globals;
mod parsers;
mod reader;
mod tree;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

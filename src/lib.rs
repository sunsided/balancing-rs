extern crate core;

mod index;
mod simulation;
mod timing;

use index::{Index, IndexAssignment};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let index = Index::new(0, 100, 512);

        let indexes = vec![
            Index::new(0, 100, 512),
            Index::new(1, 100, 512),
            Index::new(2, 100, 512),
        ];
        assert!(!indexes.is_empty());
    }
}

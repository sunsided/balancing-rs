extern crate core;

mod index;
mod simulation;
mod timing;

use crate::simulation::SimulationBuilder;
use crate::timing::{Milliseconds, Nanoseconds};
use index::{Index, IndexAssignment};

pub fn main() {
    let simulation = SimulationBuilder::default()
        .with_index(Index::new_from_shards(0, &[20_000_000], 784))
        .with_index(Index::new_from_shards(1, &[10_000_000; 2], 784))
        .with_index(Index::new_from_shards(2, &[5_000_000; 4], 784))
        .with_index(Index::new_from_shards(
            3,
            &[
                3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000, 2_000_000,
            ],
            784,
        ))
        .with_index(Index::new_from_shards(4, &[1_000_000; 20], 784))
        // Double the vector, half the elements
        .with_index(Index::new_from_shards(5, &[10_000_000], 1536))
        .with_index(Index::new_from_shards(6, &[5_000_000; 2], 1536))
        .with_index(Index::new_from_shards(7, &[2_000_000; 5], 1536))
        .with_index(Index::new_from_shards(8, &[1_000_000; 10], 1536))
        // Half the vector, double the elements
        .with_index(Index::new_from_shards(9, &[40_000_000], 384))
        .with_index(Index::new_from_shards(10, &[20_000_000; 2], 384))
        .with_index(Index::new_from_shards(11, &[10_000_000; 4], 384))
        .with_index(Index::new_from_shards(12, &[2_000_000; 20], 384))
        .with_index(Index::new_from_shards(13, &[1_000_000; 40], 384))
        .with_search_cost(Nanoseconds(0.171326754), Nanoseconds(10.))
        .with_scatter_gather_cost(Milliseconds(20.), Milliseconds(0.))
        // .with_threads(4, Microseconds(10.))
        .build();

    for index_id in simulation.index_id() {
        let index = simulation.index(index_id);
        let duration = simulation.simulate_find(index.index_id);
        println!(
            "weight={weight}, num vectors={count}, dims={dims}, num shards={num_shards}, duration={duration}",
            weight = index.weight(),
            count = index.num_vectors,
            dims = index.vector_length,
            num_shards = index.num_shards(),
            duration = duration
        );
    }
}

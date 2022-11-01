extern crate core;

mod index;
mod simulation;
mod timing;

use crate::simulation::SimulationBuilder;
use crate::timing::{Microseconds, Milliseconds, Nanoseconds};
use index::{Index, IndexAssignment};
use rand::{thread_rng, Rng};

pub fn main() {
    let mut rng = thread_rng();

    println!("row,run,num_shards,num_threads,num_dims,num_vectors,weight,cost_per_vector,cost_per_scatter,cost_per_gather,thread_overhead,duration,total_duration");
    let mut row_id: usize = 0;
    for run in 0..1000 {
        let num_threads: usize = rng.gen_range(1..32);
        let search_cost_per_vector = Nanoseconds(rng.gen::<f64>() * 50.);
        let search_cost_per_scatter = Milliseconds(rng.gen::<f64>() * 100.);
        let search_cost_per_gather = Milliseconds(rng.gen::<f64>() * 100.);
        let thread_overhead = Microseconds(rng.gen::<f64>() * 100.);

        let simulation = SimulationBuilder::default()
            .with_index(Index::new_from_shards(0, &[20_000_000], 784))
            .with_index(Index::new_from_shards(1, &[10_000_000; 2], 784))
            .with_index(Index::new_from_shards(2, &[5_000_000; 4], 784))
            .with_index(Index::new_from_shards(3, &[2_000_000; 10], 784))
            .with_index(Index::new_from_shards(4, &[1_000_000; 20], 784))
            .with_index(Index::new_from_shards(
                5,
                &[15_000_000, 2_500_000, 1_000_000, 500_000],
                784,
            ))
            // Double the vector, half the elements
            .with_index(Index::new_from_shards(6, &[10_000_000], 1536))
            .with_index(Index::new_from_shards(7, &[5_000_000; 2], 1536))
            .with_index(Index::new_from_shards(8, &[2_000_000; 5], 1536))
            .with_index(Index::new_from_shards(9, &[1_000_000; 10], 1536))
            // Half the vector, double the elements
            .with_index(Index::new_from_shards(10, &[40_000_000], 384))
            .with_index(Index::new_from_shards(11, &[20_000_000; 2], 384))
            .with_index(Index::new_from_shards(12, &[10_000_000; 4], 384))
            .with_index(Index::new_from_shards(13, &[2_000_000; 20], 384))
            .with_index(Index::new_from_shards(14, &[1_000_000; 40], 384))
            // Small workload
            .with_index(Index::new_from_shards(15, &[100], 786))
            .with_index(Index::new_from_shards(16, &[50; 2], 786))
            .with_index(Index::new_from_shards(17, &[20; 5], 786))
            .with_index(Index::new_from_shards(18, &[10; 10], 786))
            .with_search_cost(Nanoseconds(0.171326754), search_cost_per_vector)
            .with_scatter_gather_cost(search_cost_per_scatter, search_cost_per_gather)
            .with_threads(num_threads, thread_overhead)
            .build();

        for index_id in simulation.index_id() {
            row_id += 1;
            let index = simulation.index(index_id);
            let result = simulation.simulate_find(index.index_id);
            println!(
                "{row_id},{run},{num_shards},{num_threads},{dims},{count},{weight},{cost_per_vector},{cost_per_scatter},{cost_per_gather},{thread_overhead},{duration},{duration_total}",
                row_id = row_id,
                run = run,
                num_shards = index.num_shards(),
                num_threads = simulation.thread_count,
                weight = index.weight(),
                count = index.num_vectors,
                dims = index.vector_length,
                duration = *result.duration,
                duration_total = *result.duration_total,
                cost_per_vector = *search_cost_per_vector,
                cost_per_scatter = *search_cost_per_scatter,
                cost_per_gather = *search_cost_per_gather,
                thread_overhead = *thread_overhead,
            );
        }
    }
}

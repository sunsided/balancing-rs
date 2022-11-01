use crate::index::Index;
use crate::timing::Seconds;
use std::collections::HashMap;

#[derive(Debug)]
pub struct Simulation {
    indexes: HashMap<usize, Index>,
    search_cost_per_vector_element: Seconds,
    search_cost_per_vector: Seconds,
    search_cost_per_scatter: Seconds,
    search_cost_per_gather: Seconds,
    thread_count: usize,
    threading_cost: Seconds,
}

pub struct SimulationBuilder {
    indexes: HashMap<usize, Index>,
    search_cost_per_vector_element: Seconds,
    search_cost_per_vector: Seconds,
    search_cost_per_scatter: Seconds,
    search_cost_per_gather: Seconds,
    thread_count: usize,
    threading_cost: Seconds,
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        Self {
            indexes: HashMap::default(),
            search_cost_per_vector_element: Seconds::default(),
            search_cost_per_vector: Seconds::default(),
            search_cost_per_scatter: Seconds::default(),
            search_cost_per_gather: Seconds::default(),
            thread_count: 1,
            threading_cost: Seconds::default(),
        }
    }
}

impl SimulationBuilder {
    pub fn with_index(mut self, index: Index) -> Self {
        self.indexes.insert(index.index_id, index);
        self
    }

    pub fn with_search_cost<E, V>(mut self, per_element: E, per_vector: V) -> Self
    where
        E: Into<Seconds>,
        V: Into<Seconds>,
    {
        self.search_cost_per_vector_element = per_element.into();
        self.search_cost_per_vector = per_vector.into();
        self
    }

    pub fn with_scatter_gather_cost<S, G>(mut self, scatter: S, gather: G) -> Self
    where
        S: Into<Seconds>,
        G: Into<Seconds>,
    {
        self.search_cost_per_scatter = scatter.into();
        self.search_cost_per_gather = gather.into();
        self
    }

    pub fn with_threads<T>(mut self, num_threads: usize, cost: T) -> Self
    where
        T: Into<Seconds>,
    {
        assert_ne!(num_threads, 0);
        self.thread_count = num_threads;
        self.threading_cost = cost.into();
        self
    }

    pub fn build(self) -> Simulation {
        Simulation {
            indexes: self.indexes,
            search_cost_per_vector_element: self.search_cost_per_vector_element,
            search_cost_per_vector: self.search_cost_per_vector,
            search_cost_per_scatter: self.search_cost_per_scatter,
            search_cost_per_gather: self.search_cost_per_gather,
            thread_count: self.thread_count,
            threading_cost: self.threading_cost,
        }
    }
}

impl Simulation {
    pub fn simulate_find(&self, index_id: usize) -> Seconds {
        let index = self.indexes.get(&index_id).expect("Index not found");

        let mut search_time = Seconds(0.);
        let mut scatter_time = Seconds(0.);
        let mut gather_time = Seconds(0.);

        for shard in index.into_iter() {
            scatter_time += self.search_cost_per_scatter;
            gather_time += self.search_cost_per_gather;
            let threading_cost = self.threading_cost * self.thread_count;

            let search_time_per_vector = self.search_cost_per_vector_element * shard.vector_length;
            let base_search_time =
                (search_time_per_vector + self.search_cost_per_vector) * shard.num_vectors;
            let threaded_search_time = base_search_time / self.thread_count + threading_cost;

            search_time = Seconds(search_time.0.max(threaded_search_time.0));
        }

        search_time + scatter_time + gather_time
    }
}

impl Into<Simulation> for SimulationBuilder {
    fn into(self) -> Simulation {
        self.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::{Microseconds, Milliseconds, Nanoseconds};

    #[test]
    fn it_works() {
        let simulation = SimulationBuilder::default()
            .with_index(Index::new_from_shards(0, &[20_000_000], 768))
            // Assuming 350.877193 ns for 2.85 million 2048-dimensional vectors per second per core
            // according to https://git.nyris.io/nyris/experiments/Nyris.Discovery.ChunkedVectors/-/blob/develop/benchmarks/BenchmarkSuite/BenchmarkDotNet.Artifacts/results/BenchmarkSuite.Benchmarks.Avx2CacheBlockedAndUnrolledProcessorDisassembly-report-github.md,
            // this gives  1 / ((2.85 million * 2048) / second) = 0.171326754 ns per element.
            .with_search_cost(Nanoseconds(0.171326754), Nanoseconds(10.))
            // A bad estimation for gRPC unary calls on a single host taken from
            // https://www.mpi-hd.mpg.de/personalhomes/fwerner/research/2021/09/grpc-for-ipc/
            // Those numbers of course don't reflect our setup, but they provide some ballpark at least.
            .with_scatter_gather_cost(Microseconds(200.), Microseconds(200.))
            .build();

        let _ = simulation.simulate_find(0);
    }

    #[test]
    fn single_shard() {
        let shard_assignment = vec![
            (768, vec![20_000_000]),
            (768, vec![10_000_000; 2]),
            (768, vec![5_000_000; 4]),
            (
                768,
                vec![
                    3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000, 2_000_000,
                ],
            ),
            (768, vec![1_000_000; 20]),
            // Double the vector, half the elements
            (1536, vec![10_000_000]),
            (1536, vec![5_000_000; 2]),
            (1536, vec![2_000_000; 5]),
            (1536, vec![1_000_000; 10]),
            // Half the vector, double the elements
            (384, vec![40_000_000]),
            (384, vec![20_000_000; 2]),
            (384, vec![10_000_000; 4]),
            (384, vec![2_000_000; 20]),
            (384, vec![1_000_000; 40]),
        ];

        for (dims, assignment) in shard_assignment {
            let simulation = SimulationBuilder::default()
                .with_index(Index::new_from_shards(0, &assignment, dims))
                .with_search_cost(Nanoseconds(0.171326754), Nanoseconds(10.))
                .with_scatter_gather_cost(Milliseconds(20.), Milliseconds(0.))
                // .with_threads(4, Microseconds(10.))
                .build();

            let duration = simulation.simulate_find(0);
            let count: usize = assignment.iter().sum();
            let weight: usize = count * dims;
            println!(
                "weight={}, num vectors={}, dims={}, num shards={}, duration={}",
                weight,
                count,
                dims,
                assignment.len(),
                duration
            );
            let duration = duration;
        }
    }
}

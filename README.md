# Balancing Strategy Timing Evaluation

Varies different parameters such as vector count, shard count, thread count and operation latencies to determine the impact on the overall search duration.

```bash
$ cargo build
$ target/debug/balancing-rs > results.csv
```

Example output:

```csv
row,run,num_shards,num_threads,num_dims,num_vectors,weight,cost_per_vector,cost_per_scatter,cost_per_gather,thread_overhead,duration,total_duration
1,0,1,19,784,20000000,15680000000,48.649171146352366,56.84328102832301,43.133176966937015,72.3280846783403,0.29395000348030886,3.760737617251196
2,0,2,19,784,20000000,15680000000,48.649171146352366,56.84328102832301,43.133176966937015,72.3280846783403,0.2976268055374887,3.8620883088553444
3,0,4,19,784,20000000,15680000000,48.649171146352366,56.84328102832301,43.133176966937015,72.3280846783403,0.4494298935589687,4.064789692063641
```


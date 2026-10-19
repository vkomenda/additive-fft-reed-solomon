# Benchmarks

## Benchmarking methodology

Benchmarks are run by `cargo bench`. Time and throughput are obtained on an AMD EPYC 9575F for GFNI and AVX2, and on an NVIDIA GH200 for NEON. The input-output buffers are either aligned to 64 bytes or unaligned. Throughput is measured in the number of message input payload bytes, not including parity. Aligned input-output buffers yield better results on average.

We run benchmarks for n a power of 2 from 2 to 256 and fix the number T of parity shards to n/2. Benchmarked shard lengths are 64 B, 1 KiB and 64 KiB. The same set of benchmarks is run for both the Cantor basis and for the standard basis.

### Results

- [Cantor basis, aligned buffers](bench_cantor_aligned.md)
- [Cantor basis, unaligned buffers](bench_cantor_unaligned.md)
- [Standard basis, aligned buffers](bench_standard_aligned.md)
- [Standard basis, unaligned buffers](bench_standard_unaligned.md)

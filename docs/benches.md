# Benchmarks

## Benchmarking methodology

Benchmarks are run by `cargo bench`. Time and throughput are obtained on an AMD EPYC 9575F for GFNI and AVX2, and on an NVIDIA GH200 for NEON, for input-output buffers aligned to 64 bytes. Throughput is measured in the number of message input payload bytes, not including parity. Aligned input-output buffers yield better results on average.

We run benchmarks for n a power of 2 from 2 to 256 and fix the number T of parity shards to n/2. Benchmarked shard lengths are 64 B, 1 KiB and 64 KiB.

## Main RS codec methods

### `Codec::encode_systematic_sharded`

![systematic encoding of T parity shards from T data shards, T=n/2, shard length 64 B](charts/encode_systematic_sharded_64.svg)
![systematic encoding of T parity shards from T data shards, T=n/2, shard length 1 KiB](charts/encode_systematic_sharded_1024.svg)
![systematic encoding of T parity shards from T data shards, T=n/2, shard length 64 KiB](charts/encode_systematic_sharded_65536.svg)

### `Codec::recover_erasures_sharded_clobber`

![recovery of random T erased shards (Leopard-style), T=n/2, shard length 64 B](charts/recover_erasures_sharded_clobber_64.svg)
![recovery of random T erased shards (Leopard-style), T=n/2, shard length 1 KiB](charts/recover_erasures_sharded_clobber_1024.svg)
![recovery of random T erased shards (Leopard-style), T=n/2, shard length 64 KiB](charts/recover_erasures_sharded_clobber_65536.svg)

### `Codec::recover_erasures_sharded`

![recovery of random T erased shards (LNH original), T=n/2, shard length 64 B](charts/recover_erasures_sharded_64.svg)
![recovery of random T erased shards (LNH original), T=n/2, shard length 1 KiB](charts/recover_erasures_sharded_1024.svg)
![recovery of random T erased shards (LNH original), T=n/2, shard length 64 KiB](charts/recover_erasures_sharded_65536.svg)

## Transforms

## `Kernel::fft_sharded`

![transform, fft_sharded, shard length 64 B](charts/transform_fft_sharded_64.svg)
![transform, fft_sharded, shard length 1 KiB](charts/transform_fft_sharded_1024.svg)
![transform, fft_sharded, shard length 64 KiB](charts/transform_fft_sharded_65536.svg)

## `Kernel::fft_sharded_dit2`

![transform, fft_sharded_dit2, shard length 64 B](charts/transform_fft_sharded_dit2_64.svg)
![transform, fft_sharded_dit2, shard length 1 KiB](charts/transform_fft_sharded_dit2_1024.svg)
![transform, fft_sharded_dit2, shard length 64 KiB](charts/transform_fft_sharded_dit2_65536.svg)

## `Kernel::fft_sharded_dit4`

![transform, fft_sharded_dit4, shard length 64 B](charts/transform_fft_sharded_dit4_64.svg)
![transform, fft_sharded_dit4, shard length 1 KiB](charts/transform_fft_sharded_dit4_1024.svg)
![transform, fft_sharded_dit4, shard length 64 KiB](charts/transform_fft_sharded_dit4_65536.svg)

## `Kernel::fft_sharded_radix2_last`

![transform, fft_sharded_radix2_last, shard length 64 B](charts/transform_fft_sharded_radix2_last_64.svg)
![transform, fft_sharded_radix2_last, shard length 1 KiB](charts/transform_fft_sharded_radix2_last_1024.svg)
![transform, fft_sharded_radix2_last, shard length 64 KiB](charts/transform_fft_sharded_radix2_last_65536.svg)

## `Kernel::ifft_sharded`

![transform, ifft_sharded, shard length 64 B](charts/transform_ifft_sharded_64.svg)
![transform, ifft_sharded, shard length 1 KiB](charts/transform_ifft_sharded_1024.svg)
![transform, ifft_sharded, shard length 64 KiB](charts/transform_ifft_sharded_65536.svg)

## `Kernel::ifft_sharded_dit2`

![transform, ifft_sharded_dit2, shard length 64 B](charts/transform_ifft_sharded_dit2_64.svg)
![transform, ifft_sharded_dit2, shard length 1 KiB](charts/transform_ifft_sharded_dit2_1024.svg)
![transform, ifft_sharded_dit2, shard length 64 KiB](charts/transform_ifft_sharded_dit2_65536.svg)

## `Kernel::ifft_sharded_dit4`

![transform, ifft_sharded_dit4, shard length 64 B](charts/transform_ifft_sharded_dit4_64.svg)
![transform, ifft_sharded_dit4, shard length 1 KiB](charts/transform_ifft_sharded_dit4_1024.svg)
![transform, ifft_sharded_dit4, shard length 64 KiB](charts/transform_ifft_sharded_dit4_65536.svg)

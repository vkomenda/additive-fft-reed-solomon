# Cantor basis benches with unaligned buffers


## Main RS codec methods


### `Codec::encode_systematic_sharded`

![systematic encoding of T parity shards from T data shards, T=n/2, shard length 64 B](charts/encode_systematic_sharded_cantor_unaligned_64.svg)
![systematic encoding of T parity shards from T data shards, T=n/2, shard length 1 KiB](charts/encode_systematic_sharded_cantor_unaligned_1024.svg)
![systematic encoding of T parity shards from T data shards, T=n/2, shard length 64 KiB](charts/encode_systematic_sharded_cantor_unaligned_65536.svg)


### `Codec::recover_erasures_sharded_clobber`

![recovery of random T erased shards (Leopard-style), T=n/2, shard length 64 B](charts/recover_erasures_sharded_clobber_cantor_unaligned_64.svg)
![recovery of random T erased shards (Leopard-style), T=n/2, shard length 1 KiB](charts/recover_erasures_sharded_clobber_cantor_unaligned_1024.svg)
![recovery of random T erased shards (Leopard-style), T=n/2, shard length 64 KiB](charts/recover_erasures_sharded_clobber_cantor_unaligned_65536.svg)


### `Codec::recover_erasures_sharded`

![recovery of random T erased shards (LNH original), T=n/2, shard length 64 B](charts/recover_erasures_sharded_cantor_unaligned_64.svg)
![recovery of random T erased shards (LNH original), T=n/2, shard length 1 KiB](charts/recover_erasures_shardedcantor_unaligned__1024.svg)
![recovery of random T erased shards (LNH original), T=n/2, shard length 64 KiB](charts/recover_erasures_sharded_cantor_unaligned_65536.svg)


## Transforms


### `Kernel::fft_sharded`

![transform, fft_sharded, shard length 64 B](charts/transform_fft_sharded_cantor_unaligned_64.svg)
![transform, fft_sharded, shard length 1 KiB](charts/transform_fft_sharded_cantor_unaligned_1024.svg)
![transform, fft_sharded, shard length 64 KiB](charts/transform_fft_sharded_cantor_unaligned_65536.svg)


### `Kernel::fft_sharded_dit2`

![transform, fft_sharded_dit2, shard length 64 B](charts/transform_fft_sharded_dit2_cantor_unaligned_64.svg)
![transform, fft_sharded_dit2, shard length 1 KiB](charts/transform_fft_sharded_dit2_cantor_unaligned_1024.svg)
![transform, fft_sharded_dit2, shard length 64 KiB](charts/transform_fft_sharded_dit2_cantor_unaligned_65536.svg)


### `Kernel::fft_sharded_dit4`

![transform, fft_sharded_dit4, shard length 64 B](charts/transform_fft_sharded_dit4_cantor_unaligned_64.svg)
![transform, fft_sharded_dit4, shard length 1 KiB](charts/transform_fft_sharded_dit4_cantor_unaligned_1024.svg)
![transform, fft_sharded_dit4, shard length 64 KiB](charts/transform_fft_sharded_dit4_cantor_unaligned_65536.svg)


### `Kernel::fft_sharded_radix2_last`

![transform, fft_sharded_radix2_last, shard length 64 B](charts/transform_fft_sharded_radix2_last_cantor_unaligned_64.svg)
![transform, fft_sharded_radix2_last, shard length 1 KiB](charts/transform_fft_sharded_radix2_last_cantor_unaligned_1024.svg)
![transform, fft_sharded_radix2_last, shard length 64 KiB](charts/transform_fft_sharded_radix2_last_cantor_unaligned_65536.svg)


### `Kernel::ifft_sharded`

![transform, ifft_sharded, shard length 64 B](charts/transform_ifft_sharded_cantor_unaligned_64.svg)
![transform, ifft_sharded, shard length 1 KiB](charts/transform_ifft_sharded_cantor_unaligned_1024.svg)
![transform, ifft_sharded, shard length 64 KiB](charts/transform_ifft_sharded_cantor_unaligned_65536.svg)


### `Kernel::ifft_sharded_dit2`

![transform, ifft_sharded_dit2, shard length 64 B](charts/transform_ifft_sharded_dit2_cantor_unaligned_64.svg)
![transform, ifft_sharded_dit2, shard length 1 KiB](charts/transform_ifft_sharded_dit2_cantor_unaligned_1024.svg)
![transform, ifft_sharded_dit2, shard length 64 KiB](charts/transform_ifft_sharded_dit2_cantor_unaligned_65536.svg)


### `Kernel::ifft_sharded_dit4`

![transform, ifft_sharded_dit4, shard length 64 B](charts/transform_ifft_sharded_dit4_cantor_unaligned_64.svg)
![transform, ifft_sharded_dit4, shard length 1 KiB](charts/transform_ifft_sharded_dit4_cantor_unaligned_1024.svg)
![transform, ifft_sharded_dit4, shard length 64 KiB](charts/transform_ifft_sharded_dit4_cantor_unaligned_65536.svg)

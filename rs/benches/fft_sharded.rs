mod common;

#[cfg(native_avx2)]
use additive_fft_reed_solomon::kernel::avx2_kernel::Avx2Kernel;
#[cfg(native_gfni)]
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
#[cfg(native_neon)]
use additive_fft_reed_solomon::kernel::neon_kernel::NeonKernel;
use additive_fft_reed_solomon::{
    kernel::{Kernel, lut_kernel::LutKernel},
    poly_11d_lut::CantorBasisLut11d,
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use common::*;
use criterion::{
    BatchSize, Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;

macro_rules! bench_params {
    ($group:expr,
     $shard_len:expr,
     $rng:expr,
     $kernel:ty,
     $kernel_name:expr,
     [$(($n:expr)),* $(,)?]) => {
        $({
            $group.throughput(Throughput::Bytes(($n * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_{}_aligned", $n, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_fft_sharded_inner::<$kernel, $n>(&mut b, shard_len, &mut $rng, true);
                },
            );
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_{}_unaligned", $n, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_fft_sharded_inner::<$kernel, $n>(&mut b, shard_len, &mut $rng, false);
                },
            );
        })*
    }
}

fn bench_fft_sharded_inner<K, const N: usize>(
    b: &mut Bencher<'_>,
    shard_len: usize,
    rng: &mut SmallRng,
    is_aligned: bool,
) where
    K: Kernel<Gf2p8_11d>,
{
    let (mut shards, _shards_backing) = create_buffer(N, shard_len, Some(rng), is_aligned);
    b.iter_batched(
        || {},
        |_| {
            K::fft_sharded(
                &CantorBasisLut11d,
                &mut shards,
                shard_len,
                N.trailing_zeros() as u8,
                Gf2p8_11d::zero(),
            );
        },
        BatchSize::LargeInput,
    );
}

fn bench_fft_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_sharded");
    let mut rng = SmallRng::seed_from_u64(42);

    for shard_len in [64, 1024, 65536] {
        bench_params!(
            group,
            shard_len,
            &mut rng,
            LutKernel<Gf2p8_11d>,
            "lut",
            [(2), (4), (8), (16), (32), (64), (128), (256)]
        );

        #[cfg(native_avx2)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Avx2Kernel<Gf2p8_11d>,
            "avx2",
            [(2), (4), (8), (16), (32), (64), (128), (256)]
        );

        #[cfg(native_gfni)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            GfniKernel<Gf2p8_11d>,
            "gfni",
            [(2), (4), (8), (16), (32), (64), (128), (256)]
        );

        #[cfg(native_neon)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            NeonKernel<Gf2p8_11d>,
            "neon",
            [(2), (4), (8), (16), (32), (64), (128), (256)]
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fft_sharded);
criterion_main!(benches);

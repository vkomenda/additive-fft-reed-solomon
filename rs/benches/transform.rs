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
use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use std::hint::black_box;

type Transform = fn(&CantorBasisLut11d, &mut [Gf2p8_11d], usize, u8, Gf2p8_11d);

fn transforms<K: Kernel<Gf2p8_11d>>() -> [(&'static str, Transform); 6] {
    [
        ("fft_sharded", K::fft_sharded),
        ("fft_sharded_dit2", K::fft_sharded_dit2),
        ("fft_sharded_dit4", K::fft_sharded_dit4),
        ("ifft_sharded", K::ifft_sharded),
        ("ifft_sharded_dit2", K::ifft_sharded_dit2),
        ("ifft_sharded_dit4", K::ifft_sharded_dit4),
    ]
}

macro_rules! bench_params {
    ($group:expr,
     $shard_len:expr,
     $rng:expr,
     $kernel_name:expr,
     $func:ident,
     $func_name:expr,
     [$($k:literal),* $(,)?]) => {
        $({
            let n = 1 << $k;
            $group.throughput(Throughput::Bytes((n * $shard_len) as u64));
            for (aligned, tag) in [(true, "aligned"), (false, "unaligned")] {
                $group.bench_with_input(
                    BenchmarkId::new(
                        format!("{}_{}_k{}_{}", $kernel_name, $func_name, $k, tag),
                        $shard_len,
                    ),
                    &$shard_len,
                    |b, &shard_len| {
                        bench_transform_inner(b, shard_len, $k, &mut *$rng, aligned, |s, l, k| {
                            $func(&CantorBasisLut11d, s, l, k, Gf2p8_11d::zero())
                        });
                    },
                );
            }
        })*
    }
}

fn bench_transform_inner(
    b: &mut Bencher<'_>,
    shard_len: usize,
    k: u8,
    rng: &mut SmallRng,
    is_aligned: bool,
    f: impl Fn(&mut [Gf2p8_11d], usize, u8),
) {
    let n = 1 << k;
    let (mut shards, _shards_backing) = create_buffer(n, shard_len, Some(rng), is_aligned);
    b.iter(|| f(black_box(&mut shards), shard_len, k));
}

fn bench_transform(c: &mut Criterion) {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut group = c.benchmark_group("transform");

    // for (name, f) in transforms::<LutKernel<Gf2p8_11d>>() {
    //     for shard_len in [64, 1024, 65536] {
    //         bench_params!(
    //             group,
    //             shard_len,
    //             &mut rng,
    //             "lut",
    //             f,
    //             name,
    //             [1, 2, 3, 4, 5, 6, 7, 8]
    //         );
    //     }
    // }

    #[cfg(native_avx2)]
    for (name, f) in transforms::<Avx2Kernel<Gf2p8_11d>>() {
        for shard_len in [64, 1024, 65536] {
            bench_params!(
                group,
                shard_len,
                &mut rng,
                "avx2",
                f,
                name,
                [1, 2, 3, 4, 5, 6, 7, 8]
            );
        }
    }

    #[cfg(native_gfni)]
    for (name, f) in transforms::<GfniKernel<Gf2p8_11d>>() {
        for shard_len in [64, 1024, 65536] {
            bench_params!(
                group,
                shard_len,
                &mut rng,
                "gfni",
                f,
                name,
                [1, 2, 3, 4, 5, 6, 7, 8]
            );
        }
    }

    #[cfg(native_neon)]
    for (name, f) in transforms::<NeonKernel<Gf2p8_11d>>() {
        for shard_len in [64, 1024, 65536] {
            bench_params!(
                group,
                shard_len,
                &mut rng,
                "neon",
                f,
                name,
                [1, 2, 3, 4, 5, 6, 7, 8]
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_transform);
criterion_main!(benches);

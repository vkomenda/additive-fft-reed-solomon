mod common;

#[cfg(native_avx2)]
use additive_fft_reed_solomon::kernel::avx2_kernel::Avx2Kernel;
#[cfg(native_gfni)]
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
#[cfg(native_neon)]
use additive_fft_reed_solomon::kernel::neon_kernel::NeonKernel;
#[cfg(native_sse2)]
use additive_fft_reed_solomon::kernel::sse2_kernel::Sse2Kernel;
#[cfg(native_ssse3)]
use additive_fft_reed_solomon::kernel::ssse3_kernel::Ssse3Kernel;
use additive_fft_reed_solomon::{
    codec::{Codec, ParityFirst, ShardLayout},
    gf2p8lut::SubspaceBasisLut,
    kernel::{Kernel, lut_kernel::LutKernel},
    poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d},
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use common::*;
use criterion::{Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::{Duration, Instant};

macro_rules! bench_params {
    ($group:expr,
     $shard_len:expr,
     $rng:expr,
     $kernel:ty,
     $kernel_name:expr,
     $l:ty,
     [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            let rs = Codec::<_, _, $kernel, $l, $n, $t>::new();
            $group.throughput(Throughput::Bytes((($n - $t) * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("{}_N{}_T{}_aligned", $kernel_name, $n, $t), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_recover_erasures_sharded_clobber_inner(&mut b, &rs, shard_len, &mut $rng, true);
                },
            );
            $group.bench_with_input(
                BenchmarkId::new(format!("{}_N{}_T{}_unaligned", $kernel_name, $n, $t), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_recover_erasures_sharded_clobber_inner(&mut b, &rs, shard_len, &mut $rng, false);
                },
            );
        })*
    }
}

fn bench_recover_erasures_sharded_clobber_inner<B, K, L, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, B, K, L, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
    is_aligned: bool,
) where
    B: SubspaceBasisLut<Gf2p8_11d> + Default,
    K: Kernel<Gf2p8_11d, B>,
    L: ShardLayout,
{
    let (pristine_backing, start) = generate_random_codeword(rs, shard_len, rng, is_aligned);
    let pristine = &pristine_backing[start..][..N * shard_len];

    // Allocated once, so every iteration runs on the same addresses rather than on a freshly
    // mapped buffer whose physical placement differs from iteration to iteration.
    let (mut work_backing, work_start) =
        create_buffer(N, shard_len, None::<&mut SmallRng>, is_aligned);
    let work = &mut work_backing[work_start..][..N * shard_len];

    // Choose T random distinct positions to erase
    let mut positions: Vec<usize> = (0..N).collect();
    // partial Fisher-Yates shuffle for T elements
    for i in 0..T {
        let j = Uniform::new(i, N).unwrap().sample(rng);
        positions.swap(i, j);
    }
    let mut erasure_positions: Vec<u8> = positions[..T].iter().map(|&p| p as u8).collect();
    erasure_positions.sort_unstable();

    b.iter_custom(|iters| {
        let mut total = Duration::ZERO;
        for _ in 0..iters {
            // Untimed: clobber recovery overwrites the buffer, so restore it each iteration.
            work.copy_from_slice(pristine);
            for &pos in &erasure_positions {
                work[pos as usize * shard_len..][..shard_len].fill(Gf2p8_11d::zero());
            }
            let t0 = Instant::now();
            rs.recover_erasures_sharded_clobber(
                black_box(&mut *work),
                shard_len,
                black_box(&erasure_positions),
            );
            total += t0.elapsed();
        }
        total
    });
}

fn bench_recover_erasures_sharded_clobber(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_erasures_sharded_clobber");
    let mut rng = SmallRng::seed_from_u64(42);

    // 65600 = 64 x 1025 rather than 2^16: a power-of-two stride makes shards share cache sets
    // and 4K-alias, so timings depend on page placement instead of on the code.
    for shard_len in [64, 1024, 65600] {
        #[cfg(native_gfni)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            GfniKernel<Gf2p8_11d, CantorBasisLut11d>,
            "gfni_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_gfni)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            GfniKernel<Gf2p8_11d, StandardBasisLut11d>,
            "gfni_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_avx2)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>,
            "avx2_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_avx2)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>,
            "avx2_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_ssse3)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d>,
            "ssse3_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_ssse3)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d>,
            "ssse3_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_sse2)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Sse2Kernel<Gf2p8_11d, CantorBasisLut11d>,
            "sse2_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_sse2)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            Sse2Kernel<Gf2p8_11d, StandardBasisLut11d>,
            "sse2_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_neon)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            NeonKernel<Gf2p8_11d, CantorBasisLut11d>,
            "neon_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        #[cfg(native_neon)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            NeonKernel<Gf2p8_11d, StandardBasisLut11d>,
            "neon_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        bench_params!(
            group,
            shard_len,
            &mut rng,
            LutKernel<Gf2p8_11d, CantorBasisLut11d>,
            "lut_cantor",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );

        bench_params!(
            group,
            shard_len,
            &mut rng,
            LutKernel<Gf2p8_11d, StandardBasisLut11d>,
            "lut_standard",
            ParityFirst,
            [
                (2, 1),
                (4, 1),
                (4, 2),
                (8, 2),
                (8, 4),
                (16, 4),
                (16, 8),
                (32, 8),
                (32, 16),
                (64, 16),
                (64, 32),
                (128, 32),
                (128, 64),
                (256, 64),
                (256, 128),
            ]
        );
    }
    group.finish();
}

criterion_group!(benches, bench_recover_erasures_sharded_clobber);
criterion_main!(benches);

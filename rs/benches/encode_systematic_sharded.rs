mod common;

#[cfg(native_avx2)]
use additive_fft_reed_solomon::kernel::avx2_kernel::Avx2Kernel;
#[cfg(native_gfni)]
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
#[cfg(native_neon)]
use additive_fft_reed_solomon::kernel::neon_kernel::NeonKernel;
use additive_fft_reed_solomon::{
    codec::{Codec, ParityFirst, ShardLayout},
    gf2p8lut::SubspaceBasisLut,
    kernel::{Kernel, lut_kernel::LutKernel},
    poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d},
};
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;
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
     $l:ty,
     [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            let rs = Codec::<_, _, $kernel, $l, $n, $t>::new();
            $group.throughput(Throughput::Bytes((($n - $t) * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("{}_N{}_T{}_aligned", $kernel_name, $n, $t), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_encode_systematic_sharded_inner(&mut b, &rs, shard_len, &mut $rng, true);
                },
            );
            $group.bench_with_input(
                BenchmarkId::new(format!("{}_N{}_T{}_unaligned", $kernel_name, $n, $t), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_encode_systematic_sharded_inner(&mut b, &rs, shard_len, &mut $rng, false);
                },
            );
        })*
    }
}

fn bench_encode_systematic_sharded_inner<B, K, L, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, B, K, L, N, T>,
    shard_len: usize,
    rng: &mut SmallRng,
    is_aligned: bool,
) where
    B: SubspaceBasisLut<Gf2p8_11d> + Default,
    K: Kernel<Gf2p8_11d, B>,
    L: ShardLayout,
{
    let (message, _message_backing) = create_buffer(N - T, shard_len, Some(rng), is_aligned);
    b.iter_batched(
        || {
            let (parity_backing, parity_start) =
                create_buffer(T, shard_len, None::<&mut SmallRng>, is_aligned);
            let workspace = aligned_buffer(shard_len * T);
            (parity_backing, parity_start, workspace)
        },
        |(mut parity_backing, parity_start, mut workspace)| {
            rs.encode_systematic_sharded(
                &message,
                &mut parity_backing[parity_start..][..T * shard_len],
                &mut workspace,
                shard_len,
            );
        },
        BatchSize::LargeInput,
    );
}

fn bench_encode_systematic_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_systematic_sharded");
    let mut rng = SmallRng::seed_from_u64(42);

    for shard_len in [64, 1024, 65536] {
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
    }
    group.finish();
}

criterion_group!(benches, bench_encode_systematic_sharded);
criterion_main!(benches);

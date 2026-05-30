#[cfg(native_gfni)]
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
use additive_fft_reed_solomon::{
    codec::Codec,
    gf2p8lut::CantorBasisLut,
    kernel::{Kernel, lut_kernel::LutKernel},
    poly_11d_lut::CantorBasisLut11d,
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use criterion::{
    BatchSize, Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

macro_rules! bench_params {
    ($group:expr,
     $shard_len:expr,
     $rng:expr,
     $kernel:ty,
     $kernel_name:expr,
     [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            let rs = Codec::<Gf2p8_11d, CantorBasisLut11d, $kernel, $n, $t>::new();
            $group.throughput(Throughput::Bytes(($n * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}_{}", $n, $t, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_recover_erasure_shards_inner(&mut b, &rs, shard_len, &mut $rng);
                },
            );
        })*
    }
}

fn generate_random_codeword<B, K, const N: usize, const T: usize>(
    rs: &Codec<Gf2p8_11d, B, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
) -> Vec<Gf2p8_11d>
where
    B: CantorBasisLut<Gf2p8_11d> + Default,
    K: Kernel<Gf2p8_11d>,
{
    let k = N - T;
    let mut message = vec![Gf2p8_11d::zero(); k * shard_len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(message.as_mut_ptr() as *mut u8, message.len()) };
    rng.fill_bytes(bytes);

    let mut parity = vec![Gf2p8_11d::zero(); shard_len * T];
    let mut workspace = vec![Gf2p8_11d::zero(); shard_len * T];

    rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, shard_len);

    let mut codeword = vec![Gf2p8_11d::zero(); shard_len * N];
    codeword[..T * shard_len].clone_from_slice(&parity);
    codeword[T * shard_len..].clone_from_slice(&message);
    codeword
}

fn bench_recover_erasure_shards_inner<K, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, CantorBasisLut11d, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
) where
    K: Kernel<Gf2p8_11d>,
{
    let original = generate_random_codeword(rs, shard_len, rng);
    b.iter_batched(
        || {
            let mut received = original.clone();

            // Choose T random distinct positions to erase
            let mut positions: Vec<usize> = (0..N).collect();
            // partial Fisher-Yates shuffle for T elements
            for i in 0..T {
                let j = Uniform::new(i, N).unwrap().sample(rng);
                positions.swap(i, j);
            }
            let mut erasure_positions = positions[..T].to_vec();
            erasure_positions.sort_unstable();

            for &pos in &erasure_positions {
                received[pos * shard_len..(pos + 1) * shard_len].fill(Gf2p8_11d::zero());
            }

            let workspace = vec![Gf2p8_11d::zero(); shard_len * N];

            (
                received,
                workspace,
                erasure_positions
                    .iter()
                    .map(|&p| p as u8)
                    .collect::<Vec<u8>>(),
            )
        },
        |(mut received, mut workspace, erasure_positions)| {
            rs.recover_erasure_shards(&mut received, &mut workspace, shard_len, &erasure_positions);
        },
        BatchSize::LargeInput,
    );
}

fn bench_recover_erasure_shards(c: &mut Criterion) {
    let mut group = c.benchmark_group("recover_erasure_shards");
    let mut rng = SmallRng::seed_from_u64(42);

    for shard_len in [64, 1024, 65536] {
        bench_params!(
            group,
            shard_len,
            &mut rng,
            LutKernel<Gf2p8_11d>,
            "lut",
            [
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
            GfniKernel<Gf2p8_11d>,
            "gfni",
            [
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

criterion_group!(benches, bench_recover_erasure_shards);
criterion_main!(benches);

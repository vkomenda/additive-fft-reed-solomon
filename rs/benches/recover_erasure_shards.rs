use additive_fft_reed_solomon::RsLut;
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use criterion::{
    BatchSize, Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

macro_rules! bench_params {
    ($group:expr, $shard_len:expr, $rng:expr, [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            $group.throughput(Throughput::Bytes(($n * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}", $n, $t), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    let rs = RsLut::<$n, $t>::new();
                    bench_recover_erasure_shards_inner(&mut b, &rs, shard_len, &mut $rng);
                },
            );
        })*
    }
}

fn generate_random_codeword<const N: usize, const T: usize>(
    rs: &RsLut<N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<Gf2p8_11d>> {
    let k = N - T;
    let message: Vec<Vec<Gf2p8_11d>> = (0..k)
        .map(|_| {
            (0..shard_len)
                .map(|_| Gf2p8_11d(rng.next_u32() as u8))
                .collect()
        })
        .collect();
    let mut parity = vec![vec![Gf2p8_11d::zero(); shard_len]; T];

    let message_slices: Vec<&[Gf2p8_11d]> = message.iter().map(|s| s.as_ref()).collect();
    let mut parity_slices: Vec<&mut [Gf2p8_11d]> = parity.iter_mut().map(|s| s.as_mut()).collect();

    rs.encode_systematic_sharded(&message_slices, &mut parity_slices);

    let mut codeword = vec![vec![Gf2p8_11d::zero(); shard_len]; N];
    codeword[..T].clone_from_slice(&parity);
    codeword[T..].clone_from_slice(&message);
    codeword
}

fn bench_recover_erasure_shards_inner<const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &RsLut<N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
) {
    let original = generate_random_codeword(rs, shard_len, rng);
    b.iter_batched(
        || {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut received = original.clone();

            // Choose T random distinct positions to erase
            let mut positions: Vec<usize> = (0..N).collect();
            // partial Fisher-Yates shuffle for T elements
            for i in 0..T {
                let j = Uniform::new(i, N).unwrap().sample(&mut rng);
                positions.swap(i, j);
            }
            let mut erasure_positions: Vec<usize> = positions[..T].to_vec();
            erasure_positions.sort_unstable();

            for &pos in &erasure_positions {
                received[pos].fill(Gf2p8_11d::zero());
            }

            (received, erasure_positions)
        },
        |(mut received, erasure_positions)| {
            let mut received_slices: Vec<&mut [Gf2p8_11d]> =
                received.iter_mut().map(|s| s.as_mut()).collect();
            rs.recover_erasure_shards(&mut received_slices, &erasure_positions);
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

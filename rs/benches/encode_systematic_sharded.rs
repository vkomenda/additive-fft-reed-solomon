#[cfg(native_gfni)]
use additive_fft_reed_solomon::kernel::gfni_kernel::GfniKernel;
use additive_fft_reed_solomon::{
    codec::Codec,
    kernel::{Kernel, lut_kernel::LutKernel},
    poly_11d_lut::CantorBasisLut11d,
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use criterion::{
    BatchSize, Bencher, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
};
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
                    bench_encode_systematic_sharded_inner(&mut b, &rs, shard_len, &mut $rng);
                },
            );
        })*
    }
}

fn generate_random_message(
    n: usize,
    t: usize,
    shard_len: usize,
    rng: &mut impl Rng,
) -> Vec<Gf2p8_11d>
where
{
    let k = n - t;
    let mut message = vec![Gf2p8_11d::zero(); k * shard_len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(message.as_mut_ptr() as *mut u8, message.len()) };
    rng.fill_bytes(bytes);
    message
}

fn bench_encode_systematic_sharded_inner<K, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, CantorBasisLut11d, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
) where
    K: Kernel<Gf2p8_11d>,
{
    let message = generate_random_message(N, T, shard_len, rng);
    b.iter_batched(
        || {
            let parity = vec![Gf2p8_11d::zero(); shard_len * T];
            let workspace = vec![Gf2p8_11d::zero(); shard_len * T];
            (parity, workspace)
        },
        |(mut parity, mut workspace)| {
            rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, shard_len);
        },
        BatchSize::LargeInput,
    );
}

fn bench_encode_systematic_sharded(c: &mut Criterion) {
    let mut group = c.benchmark_group("encode_systematic_sharded");
    let mut rng = SmallRng::seed_from_u64(42);

    for shard_len in [64, 1024, 65536] {
        // bench_params!(
        //     group,
        //     shard_len,
        //     &mut rng,
        //     LutKernel<Gf2p8_11d>,
        //     "lut",
        //     [
        //         (2, 1),
        //         (4, 1),
        //         (4, 2),
        //         (8, 2),
        //         (8, 4),
        //         (16, 4),
        //         (16, 8),
        //         (32, 8),
        //         (32, 16),
        //         (64, 16),
        //         (64, 32),
        //         (128, 32),
        //         (128, 64),
        //         (256, 64),
        //         (256, 128),
        //     ]
        // );

        #[cfg(native_gfni)]
        bench_params!(
            group,
            shard_len,
            &mut rng,
            GfniKernel<Gf2p8_11d>,
            "gfni",
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

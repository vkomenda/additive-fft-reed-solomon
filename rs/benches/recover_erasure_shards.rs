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
use std::alloc::{Layout, alloc};

macro_rules! bench_params {
    ($group:expr,
     $shard_len:expr,
     $rng:expr,
     $kernel:ty,
     $kernel_name:expr,
     [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            let rs = Codec::<Gf2p8_11d, CantorBasisLut11d, $kernel, $n, $t>::default();
            $group.throughput(Throughput::Bytes(($n * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}_{}_aligned", $n, $t, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_recover_erasure_shards_inner(&mut b, &rs, shard_len, &mut $rng, true);
                },
            );
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}_{}_unaligned", $n, $t, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_recover_erasure_shards_inner(&mut b, &rs, shard_len, &mut $rng, false);
                },
            );
        })*
    }
}

fn aligned_buffer(len: usize) -> Vec<Gf2p8_11d> {
    let layout = Layout::from_size_align(len, 64).unwrap();
    let buf = unsafe { alloc(layout) };
    let codeword: Vec<Gf2p8_11d> = unsafe { Vec::from_raw_parts(buf as *mut Gf2p8_11d, len, len) };
    codeword
}

fn create_buffer(
    num_shards: usize,
    shard_len: usize,
    rng: &mut impl Rng,
    is_aligned: bool,
) -> (Vec<Gf2p8_11d>, usize) {
    let mut backing = vec![Gf2p8_11d::zero(); (num_shards + 1) * shard_len];
    let aligned_off = (64 - (backing.as_ptr() as usize % 64)) % 64;
    let start = if is_aligned {
        aligned_off
    } else {
        aligned_off + 1
    };

    let buffer = &mut backing[start..][..num_shards * shard_len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, buffer.len()) };
    rng.fill_bytes(bytes);

    (backing, start)
}

fn generate_random_codeword<B, K, const N: usize, const T: usize>(
    rs: &Codec<Gf2p8_11d, B, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
    is_aligned: bool,
) -> (Vec<Gf2p8_11d>, usize)
where
    B: CantorBasisLut<Gf2p8_11d> + Default,
    K: Kernel<Gf2p8_11d>,
{
    let k = N - T;
    let message_len = k * shard_len;
    let mut message = vec![Gf2p8_11d::zero(); message_len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(message.as_mut_ptr() as *mut u8, message.len()) };
    rng.fill_bytes(bytes);

    let parity_len = T * shard_len;
    let mut parity = vec![Gf2p8_11d::zero(); parity_len];
    let mut workspace = vec![Gf2p8_11d::zero(); parity_len];

    rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, shard_len);

    let (mut codeword, start) = create_buffer(N, shard_len, rng, is_aligned);

    codeword[start..start + parity_len].clone_from_slice(&parity);
    codeword[start + parity_len..start + parity_len + message_len].clone_from_slice(&message);
    (codeword, start)
}

fn bench_recover_erasure_shards_inner<K, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, CantorBasisLut11d, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
    is_aligned: bool,
) where
    K: Kernel<Gf2p8_11d>,
{
    b.iter_batched(
        || {
            let (mut codeword_backing, codeword_start) =
                generate_random_codeword(rs, shard_len, rng, is_aligned);
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
                codeword_backing
                    [codeword_start + pos * shard_len..codeword_start + (pos + 1) * shard_len]
                    .fill(Gf2p8_11d::zero());
            }

            let workspace = aligned_buffer(N * shard_len);

            (
                codeword_backing,
                codeword_start,
                workspace,
                erasure_positions
                    .iter()
                    .map(|&p| p as u8)
                    .collect::<Vec<u8>>(),
            )
        },
        |(mut codeword_backing, codeword_start, mut workspace, erasure_positions)| {
            rs.recover_erasure_shards(
                &mut codeword_backing[codeword_start..][..N * shard_len],
                &mut workspace,
                shard_len,
                &erasure_positions,
            );
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

criterion_group!(benches, bench_recover_erasure_shards);
criterion_main!(benches);

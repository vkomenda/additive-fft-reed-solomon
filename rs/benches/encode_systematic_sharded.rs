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
            $group.throughput(Throughput::Bytes(($t * $shard_len) as u64));
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}_{}_aligned", $n, $t, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_encode_systematic_sharded_inner(&mut b, &rs, shard_len, &mut $rng, true);
                },
            );
            $group.bench_with_input(
                BenchmarkId::new(format!("N{}_T{}_{}_unaligned", $n, $t, $kernel_name), $shard_len),
                &$shard_len,
                |mut b, &shard_len| {
                    bench_encode_systematic_sharded_inner(&mut b, &rs, shard_len, &mut $rng, false);
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
    rng: Option<&mut SmallRng>,
    is_aligned: bool,
) -> (Vec<Gf2p8_11d>, usize) {
    let mut backing = vec![Gf2p8_11d::zero(); (num_shards + 1) * shard_len];
    let aligned_off = (64 - (backing.as_ptr() as usize % 64)) % 64;
    let start = if is_aligned {
        aligned_off
    } else {
        aligned_off + 1
    };

    if let Some(rng) = rng {
        let buffer = &mut backing[start..][..num_shards * shard_len];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, buffer.len()) };
        rng.fill_bytes(bytes);
    }

    (backing, start)
}

fn bench_encode_systematic_sharded_inner<K, const N: usize, const T: usize>(
    b: &mut Bencher<'_>,
    rs: &Codec<Gf2p8_11d, CantorBasisLut11d, K, N, T>,
    shard_len: usize,
    rng: &mut SmallRng,
    is_aligned: bool,
) where
    K: Kernel<Gf2p8_11d>,
{
    let (message, _message_backing) = create_buffer(N - T, shard_len, Some(rng), is_aligned);
    b.iter_batched(
        || {
            let (parity_backing, parity_start) = create_buffer(T, shard_len, None, is_aligned);
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

criterion_group!(benches, bench_encode_systematic_sharded);
criterion_main!(benches);

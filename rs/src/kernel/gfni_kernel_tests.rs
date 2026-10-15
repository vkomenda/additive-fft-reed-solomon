use super::*;
use crate::kernel::lut_kernel;
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

fn make_shards(n: usize, shard_len: usize) -> Vec<Gf2p8_11d> {
    (0..n)
        .flat_map(|i| {
            (0..shard_len)
                .map(|j| Gf2p8_11d::from((i * 37 + j * 13 + 1) as u8))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// GFNI FFT produces the same evaluations as the LUT butterfly.
/// shard_len covers: pure tail (63), exact ZMM (64), ZMM + tail (65), two ZMMs (128).
#[test]
fn fft_gfni_matches_lut() {
    for shard_len in [1, 63, 64, 65, 128] {
        for k in 1u8..=8 {
            let n = 1 << k;
            let max_log_offset = k.min(7);
            for log_offset in 0..=max_log_offset {
                let beta = B::get_subspace_point_lut(1 << log_offset);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                lut_kernel::fft_sharded_recursive::<_, B>(&mut lut, shard_len, k, beta);
                GfniKernel::<_, B>::fft_sharded(&mut gfni, shard_len, k, beta);

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }
}

/// GFNI IFFT produces the same coefficients as the LUT butterfly.
#[test]
fn ifft_matches_lut() {
    for shard_len in [1, 63, 64, 65, 128] {
        for k in 1u8..=8 {
            let n = 1 << k;
            let max_log_offset = k.min(7);
            for log_offset in 0..=max_log_offset {
                let beta = B::get_subspace_point_lut(1 << log_offset);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                lut_kernel::ifft_sharded_recursive::<_, B>(&mut lut, shard_len, k, beta);
                GfniKernel::<_, B>::ifft_sharded(&mut gfni, shard_len, k, beta);

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }
}

/// IFFT;FFT ~= Id.
#[test]
fn ifft_then_fft_is_identity() {
    for shard_len in [1, 63, 64, 65] {
        for k in 1u8..8 {
            let n = 1 << k;
            let beta = B::get_subspace_point_lut(n as u8);
            let original = make_shards(n, shard_len);
            let mut data = original.clone();

            GfniKernel::<_, B>::ifft_sharded(&mut data, shard_len, k, beta);
            GfniKernel::<_, B>::fft_sharded(&mut data, shard_len, k, beta);

            assert_eq!(data, original, "k={k} shard_len={shard_len}");
        }
    }
}

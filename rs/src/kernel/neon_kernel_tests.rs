use super::*;
use crate::{gf2p8lut::SubspaceBasisLut, kernel::lut_kernel};
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

#[test]
fn debug_neon_cfg() {
    let target_aarch64 = cfg!(target_arch = "aarch64");

    assert!(target_aarch64);
}

#[test]
fn mul_vec_matches_mul_lut() {
    for p in 0..=255u8 {
        let m = &NIBBLE_MUL_TABLE[p as usize];
        for x in 0..=255u8 {
            let expected = Gf2p8_11d(x).mul_lut(Gf2p8_11d(p));

            let actual = unsafe {
                let v = vdupq_n_u8(x); // copy x cross all 16 lanes
                Gf2p8_11d(vgetq_lane_u8::<0>(mul_vec(v, m)))
            };
            assert_eq!(expected, actual, "p={p:02x} x={x:02x}");
        }
    }
}

fn make_shards(n: usize, shard_len: usize) -> Vec<Gf2p8_11d> {
    (0..n)
        .flat_map(|i| {
            (0..shard_len)
                .map(|j| Gf2p8_11d::from((i * 37 + j * 13 + 1) as u8))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// NEON FFT produces the same evaluations as the LUT butterfly.
/// shard_len covers: pure tail (15), aligned (16), aligned + tail (17), two aligned (32).
#[test]
fn fft_neon_matches_lut_rec() {
    for shard_len in [1, 15, 16, 17, 32] {
        for k in 1u8..=4 {
            let n = 1 << k;
            // Non-zero beta so twiddles are not trivially zero.
            let beta = B::get_subspace_point_lut(n as u8);
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::fft_sharded_recursive::<_, B>(&mut expected, shard_len, k, beta);
            NeonKernel::<_, B>::fft_sharded(&mut actual, shard_len, k, beta);

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

/// NEON IFFT produces the same coefficients as the LUT butterfly.
#[test]
fn ifft_neon_matches_lut_rec() {
    for shard_len in [1, 15, 16, 17, 32] {
        for k in 1u8..=4 {
            let n = 1 << k;
            let beta = B::get_subspace_point_lut(n as u8);
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::ifft_sharded_recursive::<_, B>(&mut expected, shard_len, k, beta);
            NeonKernel::<_, B>::ifft_sharded(&mut actual, shard_len, k, beta);

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

/// IFFT;FFT ~= Id.
#[test]
fn ifft_then_fft_neon_is_identity() {
    for shard_len in [1, 15, 16, 17] {
        for k in 1u8..=4 {
            let n = 1 << k;
            let beta = B::get_subspace_point_lut(n as u8);
            let original = make_shards(n, shard_len);
            let mut data = original.clone();

            NeonKernel::<_, B>::ifft_sharded(&mut data, shard_len, k, beta);
            NeonKernel::<_, B>::fft_sharded(&mut data, shard_len, k, beta);

            assert_eq!(data, original, "k={k} shard_len={shard_len}");
        }
    }
}

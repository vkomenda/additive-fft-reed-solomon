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

/// IFFT;FFT ~= Id.
#[test]
fn ifft_then_fft_is_identity() {
    for shard_len in [1, 31, 32, 33, 64] {
        for k in 1u8..8 {
            let n = 1 << k;
            let beta = B::get_subspace_point_lut(n as u8);
            let original = make_shards(n, shard_len);
            let mut data = original.clone();

            Avx2Kernel::<_, B>::ifft_sharded(&mut data, shard_len, k, beta);
            Avx2Kernel::<_, B>::fft_sharded(&mut data, shard_len, k, beta);

            assert_eq!(data, original, "k={k} shard_len={shard_len}");
        }
    }
}

#[test]
fn fft_sharded_radix2_last_matches_fft_sharded_rec() {
    for shard_len in [1, 31, 32, 33, 64] {
        for k in 1u8..=8 {
            let n = 1 << k;
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::fft_sharded_recursive::<_, B>(
                &mut expected,
                shard_len,
                k,
                Gf2p8_11d::zero(),
            );
            Avx2Kernel::<_, B>::fft_sharded_radix2_last(
                &mut actual,
                shard_len,
                k,
                Gf2p8_11d::zero(),
            );

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

#[test]
fn fft_sharded_matches_fft_sharded_rec() {
    for shard_len in [1, 31, 32, 33, 64] {
        for k in 1u8..=8 {
            let n = 1 << k;
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::fft_sharded_recursive::<_, B>(
                &mut expected,
                shard_len,
                k,
                Gf2p8_11d::zero(),
            );
            Avx2Kernel::<_, B>::fft_sharded(&mut actual, shard_len, k, Gf2p8_11d::zero());

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

#[test]
fn ifft_sharded_matches_ifft_sharded_rec() {
    for shard_len in [1, 31, 32, 33, 64] {
        for k in 1u8..=8 {
            let n = 1 << k;
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::ifft_sharded_recursive::<_, B>(
                &mut expected,
                shard_len,
                k,
                Gf2p8_11d::zero(),
            );
            Avx2Kernel::<_, B>::ifft_sharded(&mut actual, shard_len, k, Gf2p8_11d::zero());

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

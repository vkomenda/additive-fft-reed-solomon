use super::{Kernel, lut_kernel};
use crate::gf2p8lut::SubspaceBasisLut;
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};

fn fft_evaluates_x<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() {
    let n = 256;
    let mut a = vec![Gf2p8_11d::zero(); n];
    a[1] = Gf2p8_11d::one(); // coefficient of X̂_1 = x
    K::fft_sharded(&mut a, 1, 8, 0);
    for i in 0..n {
        assert_eq!(a[i], B::get_subspace_point_lut(i as u8), "i={i}");
    }
}

fn fft_evaluates_subspace_polys<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() {
    for l in 0..8u8 {
        let mut a = vec![Gf2p8_11d::zero(); 256];
        a[1 << l] = Gf2p8_11d::one();
        K::fft_sharded(&mut a, 1, 8, 0);
        for i in 0..256 {
            let w = B::get_subspace_point_lut(i as u8);
            assert_eq!(a[i], B::eval_subspace_poly_lut(l, w), "l={l} i={i}");
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

/// Shard lengths around the kernel's vector width: pure tail, exact, exact + tail, two vectors.
fn shard_lens<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() -> [usize; 5] {
    let a = K::ALIGN.max(16);
    [1, a - 1, a, a + 1, 2 * a]
}

/// Kernel FFT produces the same evaluations as the LUT butterfly.
/// shard_len covers: pure tail (K::ALIGN - 1), aligned (K::ALIGN), aligned + tail (K::ALIGN + 1),
/// two aligned (2 * K::ALIGN).
pub(super) fn fft_matches_lut<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() {
    for shard_len in shard_lens::<K, B>() {
        for k in 1u8..=8 {
            let n = 1usize << k;
            // Shift to the coset starting at ω_n where it fits. The full field has no shift.
            let coords = if k < 8 { n } else { 0 };
            let beta = B::get_subspace_point_lut(coords as u8);
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::ifft_sharded_recursive::<_, B>(&mut expected, shard_len, k, beta);
            K::ifft_sharded(&mut actual, shard_len, k, n);

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

/// Kernel IFFT produces the same coefficients as the LUT butterfly.
pub(super) fn ifft_matches_lut<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() {
    for shard_len in shard_lens::<K, B>() {
        for k in 1u8..=8 {
            let n = 1usize << k;
            // Shift to the coset starting at ω_n where it fits. The full field has no shift.
            let coords = if k < 8 { n } else { 0 };
            let beta = B::get_subspace_point_lut(coords as u8);

            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::ifft_sharded_recursive::<_, B>(&mut expected, shard_len, k, beta);
            K::ifft_sharded(&mut actual, shard_len, k, coords);

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

/// IFFT;FFT ~= Id.
pub(super) fn ifft_then_fft_is_identity<K: Kernel<Gf2p8_11d, B>, B: SubspaceBasisLut<Gf2p8_11d>>() {
    for shard_len in shard_lens::<K, B>() {
        for k in 1u8..=8 {
            let n = 1usize << k;
            // Shift to the coset starting at ω_n where it fits. The full field has no shift.
            let coords = if k < 8 { n } else { 0 };
            let beta = B::get_subspace_point_lut(coords as u8);
            let mut expected = make_shards(n, shard_len);
            let mut actual = expected.clone();

            lut_kernel::ifft_sharded_recursive::<_, B>(&mut expected, shard_len, k, beta);
            K::ifft_sharded(&mut actual, shard_len, k, coords);

            assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
        }
    }
}

pub(super) fn fft_sharded_dit4_inner_tail_matches_fft<
    K: Kernel<Gf2p8_11d, B>,
    B: SubspaceBasisLut<Gf2p8_11d>,
>() {
    for shard_len in shard_lens::<K, B>() {
        for k in 1u8..=8 {
            let n = 1usize << k;
            // Shift to the coset starting at ω_n where it fits. The full field has no shift.
            let coords = if k < 8 { n } else { 0 };
            let original = make_shards(n, shard_len);
            let mut data = original.clone();

            K::ifft_sharded(&mut data, shard_len, k, coords);
            K::fft_sharded(&mut data, shard_len, k, coords);

            assert_eq!(data, original, "k={k} shard_len={shard_len}");
        }
    }
}

/// Instantiates the generic kernel tests for both bases of `$kernel`.
macro_rules! kernel_tests {
    ($kernel:ident) => {
        mod cantor {
            type B = crate::poly_11d_lut::CantorBasisLut11d;
            type K = super::$kernel<additive_fft_reed_solomon_gf2p8::Gf2p8_11d, B>;
            #[test]
            fn fft_matches_lut() {
                crate::kernel::tests::fft_matches_lut::<K, B>()
            }
            #[test]
            fn ifft_matches_lut() {
                crate::kernel::tests::ifft_matches_lut::<K, B>()
            }
            #[test]
            fn ifft_then_fft_is_identity() {
                crate::kernel::tests::ifft_then_fft_is_identity::<K, B>()
            }
            #[test]
            fn fft_sharded_dit4_inner_tail_matches_fft() {
                crate::kernel::tests::fft_sharded_dit4_inner_tail_matches_fft::<K, B>()
            }
        }
        mod standard {
            type B = crate::poly_11d_lut::StandardBasisLut11d;
            type K = super::$kernel<additive_fft_reed_solomon_gf2p8::Gf2p8_11d, B>;
            #[test]
            fn fft_matches_lut() {
                crate::kernel::tests::fft_matches_lut::<K, B>()
            }
            #[test]
            fn ifft_matches_lut() {
                crate::kernel::tests::ifft_matches_lut::<K, B>()
            }
            #[test]
            fn ifft_then_fft_is_identity() {
                crate::kernel::tests::ifft_then_fft_is_identity::<K, B>()
            }
            #[test]
            fn fft_sharded_dit4_inner_tail_matches_fft() {
                crate::kernel::tests::fft_sharded_dit4_inner_tail_matches_fft::<K, B>()
            }
        }
    };
}

#[cfg(avx512_gfni)]
mod avx512_gfni {
    use crate::GfniKernel as K;
    kernel_tests!(K);
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use crate::Avx2Kernel as K;
    kernel_tests!(K);
}

#[cfg(target_feature = "ssse3")]
mod ssse3 {
    use crate::Ssse3Kernel as K;
    kernel_tests!(K);
}

#[cfg(target_feature = "sse2")]
mod sse2 {
    use crate::Sse2Kernel as K;
    kernel_tests!(K);
}

#[cfg(target_feature = "neon")]
mod neon {
    use crate::NeonKernel as K;
    kernel_tests!(K);
}

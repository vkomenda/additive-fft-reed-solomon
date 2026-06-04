use super::Kernel;
use crate::{
    gf2p8lut::{CantorBasisLut, Gf2p8Lut},
    poly_11d_lut::generated::CANTOR_SUBSPACE,
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use core::arch::x86_64::*;
use std::marker::PhantomData;

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_gfni_kernel_11d.rs"));
}

/// Forward butterfly transforming (a, b) into (a + T·b, b + a + T·b).
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd_gfni<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, mat: __m512i) {
    let a = a.as_mut_ptr();
    let b = b.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·b
            let va = _mm512_xor_si512(va, t); // a + T·b  = g0
            let vb = _mm512_xor_si512(vb, va); // b + g0   = g1
            _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
        }
        i += 64;
    }
    if i < len {
        let k = (1u64 << (len - i)) - 1;
        unsafe {
            let va = _mm512_maskz_loadu_epi8(k, a.add(i) as *const i8);
            let vb = _mm512_maskz_loadu_epi8(k, b.add(i) as *const i8);
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0);
            let va = _mm512_xor_si512(va, t);
            let vb = _mm512_xor_si512(vb, va);
            _mm512_mask_storeu_epi8(a.add(i) as *mut i8, k, va);
            _mm512_mask_storeu_epi8(b.add(i) as *mut i8, k, vb);
        }
    }
}

/// Inverse butterfly transforming (g0, g1) into (g0 + T·(g0+g1), g0+g1).
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_inv_gfni<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, mat: __m512i) {
    let a = a.as_mut_ptr();
    let b = b.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let vb = _mm512_xor_si512(vb, va); // d' = g0 + g1
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·d'
            let va = _mm512_xor_si512(va, t); // d  = g0 + T·d'
            _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
        }
        i += 64;
    }
    if i < len {
        let k = (1u64 << (len - i)) - 1;
        unsafe {
            let va = _mm512_maskz_loadu_epi8(k, a.add(i) as *const i8);
            let vb = _mm512_maskz_loadu_epi8(k, b.add(i) as *const i8);
            let vb = _mm512_xor_si512(vb, va);
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0);
            let va = _mm512_xor_si512(va, t);
            _mm512_mask_storeu_epi8(a.add(i) as *mut i8, k, va);
            _mm512_mask_storeu_epi8(b.add(i) as *mut i8, k, vb);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_gfni<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G],
    shard_len: usize,
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1usize << (k - 1);
    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let mat = _mm512_set1_epi64(twiddle.gfni_mul_matrix() as i64);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_fwd_gfni(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            mat,
        );
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    fft_sharded_gfni(basis, &mut shards[..h], shard_len, k - 1, beta);
    fft_sharded_gfni(basis, &mut shards[h..], shard_len, k - 1, next_beta);
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn ifft_sharded_gfni<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G],
    shard_len: usize,
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1usize << (k - 1);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    ifft_sharded_gfni(
        basis,
        &mut shards[..half * shard_len],
        shard_len,
        k - 1,
        beta,
    );
    ifft_sharded_gfni(
        basis,
        &mut shards[half * shard_len..],
        shard_len,
        k - 1,
        next_beta,
    );

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let mat = _mm512_set1_epi64(twiddle.gfni_mul_matrix() as i64);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_inv_gfni(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            mat,
        )
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn scale_gfni<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, mat: __m512i) {
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let v = _mm512_loadu_si512(src.add(i) as *const __m512i);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, r);
        }
        i += 64;
    }
    if i < len {
        let k = (1u64 << (len - i)) - 1;
        unsafe {
            let v = _mm512_maskz_loadu_epi8(k, src.add(i) as *const i8);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_mask_storeu_epi8(dst.add(i) as *mut i8, k, r);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, mat: __m512i) {
    let dst = dst.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let v = _mm512_loadu_si512(dst.add(i) as *const __m512i);
            let v = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, v);
        }
        i += 64;
    }
    if i < len {
        let k = (1u64 << (len - i)) - 1;
        unsafe {
            let v = _mm512_maskz_loadu_epi8(k, dst.add(i) as *const i8);
            let v = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_mask_storeu_epi8(dst.add(i) as *mut i8, k, v);
        }
    }
}

#[derive(Default)]
pub struct GfniKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for GfniKernel<Gf2p8_11d> {
    fn fft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        unsafe {
            if beta == Gf2p8_11d::zero() {
                match k {
                    0 => {}
                    1 => unrolled_11d::fft_sharded_gfni_2(shards, shard_len),
                    2 => unrolled_11d::fft_sharded_gfni_4(shards, shard_len),
                    3 => unrolled_11d::fft_sharded_gfni_8(shards, shard_len),
                    4 => unrolled_11d::fft_sharded_gfni_16(shards, shard_len),
                    5 => unrolled_11d::fft_sharded_gfni_32(shards, shard_len),
                    6 => unrolled_11d::fft_sharded_gfni_64(shards, shard_len),
                    7 => unrolled_11d::fft_sharded_gfni_128(shards, shard_len),
                    8 => unrolled_11d::fft_sharded_gfni_256(shards, shard_len),
                    _ => unreachable!("k={k} must be in 0..=8"),
                }
            } else {
                fft_sharded_gfni(basis, shards, shard_len, k, beta);
            }
        }
    }

    fn ifft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        unsafe {
            if beta == Gf2p8_11d::zero() {
                match k {
                    0 => {}
                    1 => unrolled_11d::ifft_sharded_gfni_2(shards, shard_len),
                    2 => unrolled_11d::ifft_sharded_gfni_4(shards, shard_len),
                    3 => unrolled_11d::ifft_sharded_gfni_8(shards, shard_len),
                    4 => unrolled_11d::ifft_sharded_gfni_16(shards, shard_len),
                    5 => unrolled_11d::ifft_sharded_gfni_32(shards, shard_len),
                    6 => unrolled_11d::ifft_sharded_gfni_64(shards, shard_len),
                    7 => unrolled_11d::ifft_sharded_gfni_128(shards, shard_len),
                    8 => unrolled_11d::ifft_sharded_gfni_256(shards, shard_len),
                    _ => unreachable!("k={k} must be in 0..=8"),
                }
            } else {
                match (k, u8::from(beta)) {
                    (0, _) => {}
                    (1, b) if b == CANTOR_SUBSPACE[1] => {
                        unrolled_11d::ifft_sharded_gfni_2_01(shards, shard_len)
                    }

                    (2, b) if b == CANTOR_SUBSPACE[2] => {
                        unrolled_11d::ifft_sharded_gfni_4_d6(shards, shard_len)
                    }
                    (3, b) if b == CANTOR_SUBSPACE[4] => {
                        unrolled_11d::ifft_sharded_gfni_8_98(shards, shard_len)
                    }
                    (4, b) if b == CANTOR_SUBSPACE[8] => {
                        unrolled_11d::ifft_sharded_gfni_16_92(shards, shard_len)
                    }
                    (5, b) if b == CANTOR_SUBSPACE[16] => {
                        unrolled_11d::ifft_sharded_gfni_32_56(shards, shard_len)
                    }
                    (6, b) if b == CANTOR_SUBSPACE[32] => {
                        unrolled_11d::ifft_sharded_gfni_64_c8(shards, shard_len)
                    }
                    (7, b) if b == CANTOR_SUBSPACE[64] => {
                        unrolled_11d::ifft_sharded_gfni_128_58(shards, shard_len)
                    }
                    (8, b) if b == CANTOR_SUBSPACE[128] => {
                        unrolled_11d::ifft_sharded_gfni_256_e7(shards, shard_len)
                    }
                    _ => ifft_sharded_gfni(basis, shards, shard_len, k, beta),
                }
            }
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix() as i64) };
        unsafe { scale_gfni(src, dst, dst.len(), mat) }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix() as i64) };
        unsafe { scale_in_place(dst, dst.len(), mat) }
    }
}

impl<G: Gf2p8Lut> GfniKernel<G> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
#[cfg(native_gfni)]
mod tests {
    use super::*;
    use crate::{kernel::lut_kernel, poly_11d_lut::CantorBasisLut11d};
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_gfni_cfg() {
        let target_arch_x86_64 = cfg!(target_arch = "x86_64");
        let avx512f = is_x86_feature_detected!("avx512f");
        let avx512bw = is_x86_feature_detected!("avx512bw");
        let gfni = is_x86_feature_detected!("gfni");

        assert!(target_arch_x86_64);
        assert!(avx512f);
        assert!(avx512bw);
        assert!(gfni);
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

    /// GFNI FFT produces the same evaluations as the LUT butterfly.
    /// shard_len covers: pure tail (63), exact ZMM (64), ZMM + tail (65), two ZMMs (128).
    #[test]
    fn fft_gfni_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65, 128] {
            for k in 1u8..=4 {
                let n = 1 << k;
                // Non-zero beta so twiddles are not trivially zero.
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                lut_kernel::fft_sharded(&basis, &mut lut, shard_len, k, beta);
                unsafe {
                    fft_sharded_gfni(&basis, &mut gfni, shard_len, k, beta);
                }

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// GFNI IFFT produces the same coefficients as the LUT butterfly.
    #[test]
    fn ifft_gfni_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65, 128] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                lut_kernel::ifft_sharded(&basis, &mut lut, shard_len, k, beta);
                unsafe {
                    ifft_sharded_gfni(&basis, &mut gfni, shard_len, k, beta);
                }

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// IFFT;FFT ~= Id.
    #[test]
    fn ifft_then_fft_gfni_is_identity() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let original = make_shards(n, shard_len);
                let mut data = original.clone();

                unsafe {
                    ifft_sharded_gfni(&basis, &mut data, shard_len, k, beta);
                    fft_sharded_gfni(&basis, &mut data, shard_len, k, beta);
                }

                assert_eq!(data, original, "k={k} shard_len={shard_len}");
            }
        }
    }
}

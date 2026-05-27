use super::Kernel;
use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};
use core::arch::x86_64::*;
#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
use std::marker::PhantomData;

/// Forward butterfly transforming (a, b) into (a + T·b, b + a + T·b).
#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
unsafe fn butterfly_fwd_gfni(a: *mut u8, b: *mut u8, len: usize, mat: __m512i) {
    unsafe {
        let mut i = 0;
        while i + 64 <= len {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·b
            let va = _mm512_xor_si512(va, t); // a + T·b  = g0
            let vb = _mm512_xor_si512(vb, va); // b + g0   = g1
            _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
            i += 64;
        }
        if i < len {
            let k = (1u64 << (len - i)) - 1;
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
#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
unsafe fn butterfly_inv_gfni(a: *mut u8, b: *mut u8, len: usize, mat: __m512i) {
    unsafe {
        let mut i = 0;
        while i + 64 <= len {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let vb = _mm512_xor_si512(vb, va); // d' = g0 + g1
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·d'
            let va = _mm512_xor_si512(va, t); // d  = g0 + T·d'
            _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
            _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
            i += 64;
        }
        if i < len {
            let k = (1u64 << (len - i)) - 1;
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

#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
fn fft_sharded_gfni<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [&mut [G]],
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1usize << (k - 1);
    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let mat = unsafe { _mm512_set1_epi64(twiddle.gfni_mul_matrix() as i64) };

    for i in 0..half {
        let (left, right) = shards.split_at_mut(i + half);
        unsafe {
            butterfly_fwd_gfni(
                left[i].as_mut_ptr() as *mut u8,
                right[0].as_mut_ptr() as *mut u8,
                left[i].len(),
                mat,
            );
        }
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    fft_sharded_gfni(basis, &mut shards[..half], k - 1, beta);
    fft_sharded_gfni(basis, &mut shards[half..], k - 1, next_beta);
}

#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
fn ifft_sharded_gfni<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [&mut [G]],
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1usize << (k - 1);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    ifft_sharded_gfni(basis, &mut shards[..half], k - 1, beta);
    ifft_sharded_gfni(basis, &mut shards[half..], k - 1, next_beta);

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let mat = unsafe { _mm512_set1_epi64(twiddle.gfni_mul_matrix() as i64) };

    for i in 0..half {
        let (left, right) = shards.split_at_mut(i + half);
        unsafe {
            butterfly_inv_gfni(
                left[i].as_mut_ptr() as *mut u8,
                right[0].as_mut_ptr() as *mut u8,
                left[i].len(),
                mat,
            )
        };
    }
}

#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
unsafe fn scale_gfni(dst: *mut u8, src: *const u8, len: usize, mat: __m512i) {
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let v = _mm512_loadu_si512(src.add(i) as *const __m512i);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, r);
            i += 64;
        }
    }
    if i < len {
        unsafe {
            let k = (1u64 << (len - i)) - 1;
            let v = _mm512_maskz_loadu_epi8(k, src.add(i) as *const i8);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_mask_storeu_epi8(dst.add(i) as *mut i8, k, r);
        }
    }
}

unsafe fn scale_in_place(dst: *mut u8, len: usize, mat: __m512i) {
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
        unsafe {
            let k = (1u64 << (len - i)) - 1;
            let v = _mm512_maskz_loadu_epi8(k, dst.add(i) as *const i8);
            let v = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            _mm512_mask_storeu_epi8(dst.add(i) as *mut i8, k, v);
        }
    }
}

#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
pub struct GfniKernel<G: Gf2p8Lut>(PhantomData<G>);

#[cfg(any(avx512_gfni, feature = "avx512_gfni"))]
impl<G: Gf2p8Lut> Kernel<G> for GfniKernel<G> {
    fn fft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G) {
        fft_sharded_gfni(basis, shards, k, beta)
    }

    fn ifft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G) {
        ifft_sharded_gfni(basis, shards, k, beta)
    }

    fn scale(dst: &mut [G], src: &[G], scalar: G) {
        unsafe {
            let mat = _mm512_set1_epi64(scalar.gfni_mul_matrix() as i64);
            scale_gfni(dst.as_mut_ptr() as _, src.as_ptr() as _, dst.len(), mat)
        }
    }

    fn scale_in_place(dst: &mut [G], scalar: G) {
        unsafe {
            let mat = _mm512_set1_epi64(scalar.gfni_mul_matrix() as i64);
            scale_in_place(dst.as_mut_ptr() as _, dst.len(), mat)
        }
    }
}

impl<G: Gf2p8Lut> GfniKernel<G> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_11d_lut::CantorBasisLut11d;
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_gfni_cfg() {
        println!("target_arch x86_64: {}", cfg!(target_arch = "x86_64"));
        println!("avx512f: {}", cfg!(target_feature = "avx512f"));
        println!("avx512bw: {}", cfg!(target_feature = "avx512bw"));
        println!("gfni: {}", cfg!(target_feature = "gfni"));
    }

    fn make_shards(n: usize, shard_len: usize) -> Vec<Vec<Gf2p8_11d>> {
        (0..n)
            .map(|i| {
                (0..shard_len)
                    .map(|j| Gf2p8_11d::from((i * 37 + j * 13 + 1) as u8))
                    .collect()
            })
            .collect()
    }

    /// GFNI FFT produces the same evaluations as the LUT butterfly.
    /// shard_len covers: pure tail (63), exact ZMM (64), ZMM + tail (65), two ZMMs (128).
    #[test]
    #[cfg(avx512_gfni)]
    fn fft_gfni_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65, 128] {
            for k in 1u8..=4 {
                let n = 1 << k;
                // Non-zero beta so twiddles are not trivially zero.
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                let mut lut_slices: Vec<&mut [Gf2p8_11d]> =
                    lut.iter_mut().map(|s| s.as_mut_slice()).collect();
                basis.fft_sharded(&mut lut_slices, k, beta);

                let mut gfni_slices: Vec<&mut [Gf2p8_11d]> =
                    gfni.iter_mut().map(|s| s.as_mut_slice()).collect();
                unsafe {
                    fft_sharded_gfni(&basis, &mut gfni_slices, k, beta);
                }

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// GFNI IFFT produces the same coefficients as the LUT butterfly.
    #[test]
    #[cfg(avx512_gfni)]
    fn ifft_gfni_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65, 128] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut lut = make_shards(n, shard_len);
                let mut gfni = lut.clone();

                let mut lut_slices: Vec<&mut [Gf2p8_11d]> =
                    lut.iter_mut().map(|s| s.as_mut_slice()).collect();
                basis.ifft_sharded(&mut lut_slices, k, beta);

                let mut gfni_slices: Vec<&mut [Gf2p8_11d]> =
                    gfni.iter_mut().map(|s| s.as_mut_slice()).collect();
                unsafe {
                    ifft_sharded_gfni(&basis, &mut gfni_slices, k, beta);
                }

                assert_eq!(lut, gfni, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// IFFT;FFT ~= Id.
    #[test]
    #[cfg(avx512_gfni)]
    fn ifft_then_fft_gfni_is_identity() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 63, 64, 65] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let original = make_shards(n, shard_len);
                let mut data = original.clone();

                let mut slices: Vec<&mut [Gf2p8_11d]> =
                    data.iter_mut().map(|s| s.as_mut_slice()).collect();
                unsafe {
                    ifft_sharded_gfni(&basis, &mut slices, k, beta);
                    fft_sharded_gfni(&basis, &mut slices, k, beta);
                }

                assert_eq!(data, original, "k={k} shard_len={shard_len}");
            }
        }
    }
}

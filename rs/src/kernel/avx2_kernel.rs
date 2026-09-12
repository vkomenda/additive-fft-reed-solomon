use super::Kernel;
use crate::{
    gf2p8lut::{CantorBasisLut, Gf2p8Lut},
    poly_11d_lut::generated::{CANTOR_SUBSPACE, NIBBLE_MUL_TABLE},
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, NibbleMulTable};
use core::arch::x86_64::*;
use std::marker::PhantomData;

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_avx2_kernel_11d.rs"));
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_vec(v: __m256i, m: &NibbleMulTable) -> __m256i {
    unsafe {
        let lo_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(m.0.as_ptr() as *const __m128i));
        let hi_v = _mm256_broadcastsi128_si256(_mm_loadu_si128(m.1.as_ptr() as *const __m128i));
        let mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_shuffle_epi8(lo_v, _mm256_and_si256(v, mask));
        let hi = _mm256_shuffle_epi8(hi_v, _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask));
        _mm256_xor_si256(lo, hi)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
                let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
                let t = mul_vec(vb, m); // T·b
                let va = _mm256_xor_si256(va, t); // a + T·b  = g0
                let vb = _mm256_xor_si256(vb, va); // b + g0   = g1
                _mm256_storeu_si256(a.add(i) as *mut __m256i, va);
                _mm256_storeu_si256(b.add(i) as *mut __m256i, vb);
            }
            i += 32;
        }
    }
    // Handle the tail scalar. Masked AVX2 load/store ops work on 4-byte dwords, hence apply the
    // multiplication table elementwise instead.
    while i < len {
        let x = a[i];
        let y = b[i];
        let g0 = x.add(y.nibble_mul(m));
        a[i] = g0;
        b[i] = y.add(g0);
        i += 1;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
                let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
                let vb = _mm256_xor_si256(vb, va); // d' = g0 + g1
                let t = mul_vec(vb, m); // T·d'
                let va = _mm256_xor_si256(va, t); // d  = g0 + T·d'
                _mm256_storeu_si256(a.add(i) as *mut __m256i, va);
                _mm256_storeu_si256(b.add(i) as *mut __m256i, vb);
            }
            i += 32;
        }
    }
    while i < len {
        let x = a[i];
        let y = b[i];
        let y = x.add(y);
        let x = x.add(y.nibble_mul(m));
        a[i] = x;
        b[i] = y;
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
fn fft_sharded<G: Gf2p8Lut>(
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
    let m = &NIBBLE_MUL_TABLE[twiddle.into_usize()];

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_fwd(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            m,
        );
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    fft_sharded(basis, &mut shards[..h], shard_len, k - 1, beta);
    fft_sharded(basis, &mut shards[h..], shard_len, k - 1, next_beta);
}

#[target_feature(enable = "avx2")]
fn ifft_sharded<G: Gf2p8Lut>(
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
    ifft_sharded(
        basis,
        &mut shards[..half * shard_len],
        shard_len,
        k - 1,
        beta,
    );
    ifft_sharded(
        basis,
        &mut shards[half * shard_len..],
        shard_len,
        k - 1,
        next_beta,
    );

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let m = &NIBBLE_MUL_TABLE[twiddle.into_usize()];

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_inv(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            m,
        )
    }
}

#[target_feature(enable = "avx2")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let src = src.as_ptr() as *const u8;
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
                let r = mul_vec(v, m);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, r);
            }
            i += 32;
        }
    }
    while i < len {
        let x = src[i];
        let r = x.nibble_mul(m);
        dst[i] = r;
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let v = _mm256_loadu_si256(dst.add(i) as *const __m256i);
                let r = mul_vec(v, m);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, r);
            }
            i += 32;
        }
    }
    while i < len {
        let x = dst[i];
        let r = x.nibble_mul(m);
        dst[i] = r;
        i += 1;
    }
}

#[derive(Default)]
pub struct Avx2Kernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for Avx2Kernel<Gf2p8_11d> {
    const ALIGN: usize = 32;

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
                    1 => unrolled_11d::fft_sharded_avx2_2(shards, shard_len),
                    2 => unrolled_11d::fft_sharded_avx2_4(shards, shard_len),
                    3 => unrolled_11d::fft_sharded_avx2_8(shards, shard_len),
                    4 => unrolled_11d::fft_sharded_avx2_16(shards, shard_len),
                    5 => unrolled_11d::fft_sharded_avx2_32(shards, shard_len),
                    6 => unrolled_11d::fft_sharded_avx2_64(shards, shard_len),
                    7 => unrolled_11d::fft_sharded_avx2_128(shards, shard_len),
                    8 => unrolled_11d::fft_sharded_avx2_256(shards, shard_len),
                    _ => unreachable!("k={k} must be in 0..=8"),
                }
            } else {
                fft_sharded(basis, shards, shard_len, k, beta);
            }
        }
    }

    fn fft_sharded_zero_padded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, log_support: u8) {
        unsafe {
            match (k, log_support) {
                (1, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_2_1(shards, shard_len),
                (2, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_4_1(shards, shard_len),
                (2, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_4_2(shards, shard_len),
                (3, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_8_1(shards, shard_len),
                (3, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_8_2(shards, shard_len),
                (3, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_8_4(shards, shard_len),
                (4, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_16_1(shards, shard_len),
                (4, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_16_2(shards, shard_len),
                (4, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_16_4(shards, shard_len),
                (4, 3) => unrolled_11d::fft_sharded_zero_padded_avx2_16_8(shards, shard_len),
                (5, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_32_1(shards, shard_len),
                (5, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_32_2(shards, shard_len),
                (5, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_32_4(shards, shard_len),
                (5, 3) => unrolled_11d::fft_sharded_zero_padded_avx2_32_8(shards, shard_len),
                (5, 4) => unrolled_11d::fft_sharded_zero_padded_avx2_32_16(shards, shard_len),
                (6, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_64_1(shards, shard_len),
                (6, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_64_2(shards, shard_len),
                (6, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_64_4(shards, shard_len),
                (6, 3) => unrolled_11d::fft_sharded_zero_padded_avx2_64_8(shards, shard_len),
                (6, 4) => unrolled_11d::fft_sharded_zero_padded_avx2_64_16(shards, shard_len),
                (6, 5) => unrolled_11d::fft_sharded_zero_padded_avx2_64_32(shards, shard_len),
                (7, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_128_1(shards, shard_len),
                (7, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_128_2(shards, shard_len),
                (7, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_128_4(shards, shard_len),
                (7, 3) => unrolled_11d::fft_sharded_zero_padded_avx2_128_8(shards, shard_len),
                (7, 4) => unrolled_11d::fft_sharded_zero_padded_avx2_128_16(shards, shard_len),
                (7, 5) => unrolled_11d::fft_sharded_zero_padded_avx2_128_32(shards, shard_len),
                (7, 6) => unrolled_11d::fft_sharded_zero_padded_avx2_128_64(shards, shard_len),
                (8, 0) => unrolled_11d::fft_sharded_zero_padded_avx2_256_1(shards, shard_len),
                (8, 1) => unrolled_11d::fft_sharded_zero_padded_avx2_256_2(shards, shard_len),
                (8, 2) => unrolled_11d::fft_sharded_zero_padded_avx2_256_4(shards, shard_len),
                (8, 3) => unrolled_11d::fft_sharded_zero_padded_avx2_256_8(shards, shard_len),
                (8, 4) => unrolled_11d::fft_sharded_zero_padded_avx2_256_16(shards, shard_len),
                (8, 5) => unrolled_11d::fft_sharded_zero_padded_avx2_256_32(shards, shard_len),
                (8, 6) => unrolled_11d::fft_sharded_zero_padded_avx2_256_64(shards, shard_len),
                (8, 7) => unrolled_11d::fft_sharded_zero_padded_avx2_256_128(shards, shard_len),
                _ => unreachable!("k={k} must be in 1..=8 and log_support must be < k"),
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
                    1 => unrolled_11d::ifft_sharded_avx2_2(shards, shard_len),
                    2 => unrolled_11d::ifft_sharded_avx2_4(shards, shard_len),
                    3 => unrolled_11d::ifft_sharded_avx2_8(shards, shard_len),
                    4 => unrolled_11d::ifft_sharded_avx2_16(shards, shard_len),
                    5 => unrolled_11d::ifft_sharded_avx2_32(shards, shard_len),
                    6 => unrolled_11d::ifft_sharded_avx2_64(shards, shard_len),
                    7 => unrolled_11d::ifft_sharded_avx2_128(shards, shard_len),
                    8 => unrolled_11d::ifft_sharded_avx2_256(shards, shard_len),
                    _ => unreachable!("k={k} must be in 0..=8"),
                }
            } else {
                match (k, u8::from(beta)) {
                    (0, _) => {}
                    (1, b) if b == CANTOR_SUBSPACE[1] => {
                        unrolled_11d::ifft_sharded_avx2_2_01(shards, shard_len)
                    }

                    (2, b) if b == CANTOR_SUBSPACE[2] => {
                        unrolled_11d::ifft_sharded_avx2_4_d6(shards, shard_len)
                    }
                    (3, b) if b == CANTOR_SUBSPACE[4] => {
                        unrolled_11d::ifft_sharded_avx2_8_98(shards, shard_len)
                    }
                    (4, b) if b == CANTOR_SUBSPACE[8] => {
                        unrolled_11d::ifft_sharded_avx2_16_92(shards, shard_len)
                    }
                    (5, b) if b == CANTOR_SUBSPACE[16] => {
                        unrolled_11d::ifft_sharded_avx2_32_56(shards, shard_len)
                    }
                    (6, b) if b == CANTOR_SUBSPACE[32] => {
                        unrolled_11d::ifft_sharded_avx2_64_c8(shards, shard_len)
                    }
                    (7, b) if b == CANTOR_SUBSPACE[64] => {
                        unrolled_11d::ifft_sharded_avx2_128_58(shards, shard_len)
                    }
                    (8, b) if b == CANTOR_SUBSPACE[128] => {
                        unrolled_11d::ifft_sharded_avx2_256_e7(shards, shard_len)
                    }
                    _ => ifft_sharded(basis, shards, shard_len, k, beta),
                }
            }
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale(src, dst, dst.len(), m);
        }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale_in_place(dst, dst.len(), m);
        }
    }
}

#[cfg(test)]
#[cfg(native_avx2)]
mod tests {
    use super::*;
    use crate::{kernel::lut_kernel, poly_11d_lut::CantorBasisLut11d};
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_avx2_cfg() {
        let target_avx2 = cfg!(target_arch = "x86_64");
        let avx2 = is_x86_feature_detected!("avx2");

        assert!(target_avx2);
        assert!(avx2);
    }

    #[test]
    fn mul_vec_matches_mul_lut() {
        for p in 0..=255u8 {
            let m = &NIBBLE_MUL_TABLE[p as usize];
            for x in 0..=255u8 {
                let expected = Gf2p8_11d(x).mul_lut(Gf2p8_11d(p));

                let actual = unsafe {
                    let v = _mm256_set1_epi8(x as i8); // copy x cross all 32 lanes
                    let r = mul_vec(v, m);
                    let byte0 = _mm256_extract_epi8::<0>(r) as u8;
                    let byte16 = _mm256_extract_epi8::<16>(r) as u8;
                    assert_eq!(byte0, byte16, "mul table not broadcast");
                    Gf2p8_11d(byte0)
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

    /// AVX2 FFT produces the same evaluations as the LUT butterfly.
    /// shard_len covers: pure tail (15), aligned (16), aligned + tail (17), two aligned (32).
    #[test]
    fn fft_avx2_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                // Non-zero beta so twiddles are not trivially zero.
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = expected.clone();

                lut_kernel::fft_sharded(&basis, &mut expected, shard_len, k, beta);
                unsafe {
                    fft_sharded(&basis, &mut actual, shard_len, k, beta);
                }

                assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// AVX2 IFFT produces the same coefficients as the LUT butterfly.
    #[test]
    fn ifft_avx2_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = expected.clone();

                lut_kernel::ifft_sharded(&basis, &mut expected, shard_len, k, beta);
                unsafe {
                    ifft_sharded(&basis, &mut actual, shard_len, k, beta);
                }

                assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// IFFT;FFT ~= Id.
    #[test]
    fn ifft_then_fft_avx2_is_identity() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let original = make_shards(n, shard_len);
                let mut data = original.clone();

                unsafe {
                    ifft_sharded(&basis, &mut data, shard_len, k, beta);
                    fft_sharded(&basis, &mut data, shard_len, k, beta);
                }

                assert_eq!(data, original, "k={k} shard_len={shard_len}");
            }
        }
    }
}

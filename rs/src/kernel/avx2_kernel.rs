use super::{Kernel, shard_groups};
use crate::poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d};
use crate::{
    gf2p8lut::{Gf2p8Lut, SubspaceBasisLut},
    poly_11d_lut::generated::{NIBBLE_MUL_BY_LOG, NIBBLE_MUL_TABLE, cantor, standard},
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, NibbleMulTable, Z255};
use core::arch::x86_64::*;
use std::marker::PhantomData;

type Multiplier = &'static NibbleMulTable;

#[inline]
#[target_feature(enable = "avx2")]
fn mul_vec_inner(v: __m256i, m_lo: __m256i, m_hi: __m256i) -> __m256i {
    let mask = _mm256_set1_epi8(0x0f);
    let lo = _mm256_shuffle_epi8(m_lo, _mm256_and_si256(v, mask));
    let hi = _mm256_shuffle_epi8(m_hi, _mm256_and_si256(_mm256_srli_epi16::<4>(v), mask));
    _mm256_xor_si256(lo, hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_mul_table(t: Multiplier) -> (__m256i, __m256i) {
    (
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(t[0].as_ptr() as *const __m128i) }),
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(t[1].as_ptr() as *const __m128i) }),
    )
}

#[target_feature(enable = "avx2")]
fn butterfly_fwd_dit2<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: Multiplier,
) {
    let (m_lo, m_hi) = load_mul_table(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 32 <= shard_len {
                unsafe {
                    let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
                    let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
                    let t = mul_vec_inner(vb, m_lo, m_hi); // T·b
                    let va = _mm256_xor_si256(va, t); // a + T·b = g0
                    let vb = _mm256_xor_si256(vb, va); // b + g0 = g1
                    _mm256_storeu_si256(a.add(i) as *mut __m256i, va);
                    _mm256_storeu_si256(b.add(i) as *mut __m256i, vb);
                }
                i += 32;
            }
        }
        // Handle the tail scalar. Masked AVX2 load/store ops work on 4-byte dwords, hence apply the
        // multiplication table elementwise instead.
        while i < shard_len {
            let x = a[i];
            let y = b[i];
            let g0 = x.add(y.nibble_mul(m));
            a[i] = g0;
            b[i] = y.add(g0);
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_fwd_dit2_zero<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 32 <= shard_len {
                unsafe {
                    let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
                    let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
                    let vb = _mm256_xor_si256(vb, va); // b + g0 = g1
                    _mm256_storeu_si256(b.add(i) as *mut __m256i, vb);
                }
                i += 32;
            }
        }
        while i < shard_len {
            let x = a[i];
            let y = b[i];
            b[i] = y.add(x);
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_inv_dit2<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: Multiplier,
) {
    let (m_lo, m_hi) = load_mul_table(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 32 <= shard_len {
                unsafe {
                    let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
                    let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
                    let vb = _mm256_xor_si256(vb, va); // d' = g0 + g1
                    let va = _mm256_xor_si256(va, mul_vec_inner(vb, m_lo, m_hi)); // d = g0 + T·d'
                    _mm256_storeu_si256(a.add(i) as *mut __m256i, va);
                    _mm256_storeu_si256(b.add(i) as *mut __m256i, vb);
                }
                i += 32;
            }
        }
        // Handle the tail scalar. Masked AVX2 load/store ops work on 4-byte dwords, hence apply the
        // multiplication table elementwise instead.
        while i < shard_len {
            let x = a[i];
            let y = b[i];
            let y = x.add(y);
            let x = x.add(y.nibble_mul(m));
            a[i] = x;
            b[i] = y;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_fwd_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let (m02_lo, m02_hi) = load_mul_table(m02);
    let (m01_lo, m01_hi) = load_mul_table(m01);
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 32 <= shard_len {
            unsafe {
                let mut w0 = _mm256_loadu_si256(p0.add(i) as *const __m256i);
                let mut w1 = _mm256_loadu_si256(p1.add(i) as *const __m256i);
                let mut w2 = _mm256_loadu_si256(p2.add(i) as *const __m256i);
                let mut w3 = _mm256_loadu_si256(p3.add(i) as *const __m256i);

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w0 = _mm256_xor_si256(w0, mul_vec_inner(w2, m02_lo, m02_hi));
                w1 = _mm256_xor_si256(w1, mul_vec_inner(w3, m02_lo, m02_hi));
                w2 = _mm256_xor_si256(w2, w0);
                w3 = _mm256_xor_si256(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w0 = _mm256_xor_si256(w0, mul_vec_inner(w1, m01_lo, m01_hi));
                w2 = _mm256_xor_si256(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w1 = _mm256_xor_si256(w1, w0);
                w3 = _mm256_xor_si256(w3, w2);

                _mm256_storeu_si256(p0.add(i) as *mut __m256i, w0);
                _mm256_storeu_si256(p1.add(i) as *mut __m256i, w1);
                _mm256_storeu_si256(p2.add(i) as *mut __m256i, w2);
                _mm256_storeu_si256(p3.add(i) as *mut __m256i, w3);
            }
            i += 32;
        }

        while i < shard_len {
            let (mut w0, mut w1, mut w2, mut w3) = (s0[i], s1[i], s2[i], s3[i]);
            w0 = w0.add(w2.nibble_mul(m02));
            w1 = w1.add(w3.nibble_mul(m02));
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w0 = w0.add(w1.nibble_mul(m01));
            w2 = w2.add(w3.nibble_mul(m23));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            s0[i] = w0;
            s1[i] = w1;
            s2[i] = w2;
            s3[i] = w3;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_fwd_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: Multiplier,
) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 32 <= shard_len {
            unsafe {
                let w0 = _mm256_loadu_si256(p0.add(i) as *const __m256i);
                let mut w1 = _mm256_loadu_si256(p1.add(i) as *const __m256i);
                let mut w2 = _mm256_loadu_si256(p2.add(i) as *const __m256i);
                let mut w3 = _mm256_loadu_si256(p3.add(i) as *const __m256i);

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w2 = _mm256_xor_si256(w2, w0);
                w3 = _mm256_xor_si256(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w2 = _mm256_xor_si256(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w1 = _mm256_xor_si256(w1, w0);
                w3 = _mm256_xor_si256(w3, w2);

                _mm256_storeu_si256(p1.add(i) as *mut __m256i, w1);
                _mm256_storeu_si256(p2.add(i) as *mut __m256i, w2);
                _mm256_storeu_si256(p3.add(i) as *mut __m256i, w3);
            }
            i += 32;
        }

        while i < shard_len {
            let (w0, mut w1, mut w2, mut w3) = (s0[i], s1[i], s2[i], s3[i]);
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w2 = w2.add(w3.nibble_mul(m23));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            s1[i] = w1;
            s2[i] = w2;
            s3[i] = w3;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_inv_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let (m02_lo, m02_hi) = load_mul_table(m02);
    let (m01_lo, m01_hi) = load_mul_table(m01);
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 32 <= shard_len {
            unsafe {
                let mut w0 = _mm256_loadu_si256(p0.add(i) as *const __m256i);
                let mut w1 = _mm256_loadu_si256(p1.add(i) as *const __m256i);
                let mut w2 = _mm256_loadu_si256(p2.add(i) as *const __m256i);
                let mut w3 = _mm256_loadu_si256(p3.add(i) as *const __m256i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm256_xor_si256(w3, w2);
                w1 = _mm256_xor_si256(w1, w0);
                w2 = _mm256_xor_si256(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w0 = _mm256_xor_si256(w0, mul_vec_inner(w1, m01_lo, m01_hi));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = _mm256_xor_si256(w3, w1);
                w2 = _mm256_xor_si256(w2, w0);
                w1 = _mm256_xor_si256(w1, mul_vec_inner(w3, m02_lo, m02_hi));
                w0 = _mm256_xor_si256(w0, mul_vec_inner(w2, m02_lo, m02_hi));

                _mm256_storeu_si256(p0.add(i) as *mut __m256i, w0);
                _mm256_storeu_si256(p1.add(i) as *mut __m256i, w1);
                _mm256_storeu_si256(p2.add(i) as *mut __m256i, w2);
                _mm256_storeu_si256(p3.add(i) as *mut __m256i, w3);
            }
            i += 32;
        }

        while i < shard_len {
            let (mut w0, mut w1, mut w2, mut w3) = (s0[i], s1[i], s2[i], s3[i]);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.nibble_mul(m23));
            w0 = w0.add(w1.nibble_mul(m01));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            w1 = w1.add(w3.nibble_mul(m02));
            w0 = w0.add(w2.nibble_mul(m02));
            s0[i] = w0;
            s1[i] = w1;
            s2[i] = w2;
            s3[i] = w3;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn butterfly_inv_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: Multiplier,
) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 32 <= shard_len {
            unsafe {
                let w0 = _mm256_loadu_si256(p0.add(i) as *const __m256i);
                let mut w1 = _mm256_loadu_si256(p1.add(i) as *const __m256i);
                let mut w2 = _mm256_loadu_si256(p2.add(i) as *const __m256i);
                let mut w3 = _mm256_loadu_si256(p3.add(i) as *const __m256i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm256_xor_si256(w3, w2);
                w1 = _mm256_xor_si256(w1, w0);
                w2 = _mm256_xor_si256(w2, mul_vec_inner(w3, m23_lo, m23_hi));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = _mm256_xor_si256(w3, w1);
                w2 = _mm256_xor_si256(w2, w0);

                _mm256_storeu_si256(p1.add(i) as *mut __m256i, w1);
                _mm256_storeu_si256(p2.add(i) as *mut __m256i, w2);
                _mm256_storeu_si256(p3.add(i) as *mut __m256i, w3);
            }
            i += 32;
        }

        while i < shard_len {
            let (w0, mut w1, mut w2, mut w3) = (s0[i], s1[i], s2[i], s3[i]);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.nibble_mul(m23));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            s1[i] = w1;
            s2[i] = w2;
            s3[i] = w3;
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    {
        let src = src.as_ptr() as *const u8;
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
                let r = mul_vec_inner(v, m_lo, m_hi);
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
fn scale_add<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    {
        let src = src.as_ptr() as *const u8;
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let v = _mm256_loadu_si256(src.add(i) as *const __m256i);
                let mut w = _mm256_loadu_si256(dst.add(i) as *const __m256i);
                let r = mul_vec_inner(v, m_lo, m_hi);
                w = _mm256_xor_si256(w, r);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, w);
            }
            i += 32;
        }
    }
    while i < len {
        let x = src[i];
        let r = x.nibble_mul(m);
        dst[i] = dst[i].add(r);
        i += 1;
    }
}

#[target_feature(enable = "avx2")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    {
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 32 <= len {
            unsafe {
                let v = _mm256_loadu_si256(dst.add(i) as *const __m256i);
                let r = mul_vec_inner(v, m_lo, m_hi);
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

#[target_feature(enable = "avx2")]
fn fft_sharded_cantor(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K1_O0,
                &cantor::FFT_DIT4_NIBBLE_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K2_O0,
                &cantor::FFT_DIT4_NIBBLE_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K3_O0,
                &cantor::FFT_DIT4_NIBBLE_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K4_O0,
                &cantor::FFT_DIT4_NIBBLE_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K5_O0,
                &cantor::FFT_DIT4_NIBBLE_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K6_O0,
                &cantor::FFT_DIT4_NIBBLE_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K7_O0,
                &cantor::FFT_DIT4_NIBBLE_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K8_O0,
                &cantor::FFT_DIT4_NIBBLE_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "avx2")]
fn fft_sharded_radix2_first_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    beta: Gf2p8_11d,
) {
    type K = Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O0,
                cantor::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O0,
                cantor::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O0,
                cantor::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O0,
                cantor::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O0,
                cantor::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O0,
                cantor::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O0,
                cantor::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K8_O0,
                cantor::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "avx2")]
fn ifft_sharded_cantor(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O0,
                cantor::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O0,
                cantor::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O0,
                cantor::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O0,
                cantor::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O0,
                cantor::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O0,
                cantor::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O0,
                cantor::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K8_O0,
                cantor::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        match (k, u8::from(beta)) {
            (0, _) => {}
            (1, b) if b == cantor::SUBSPACE[2] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O2,
                cantor::IFFT_DIT4_NIBBLE_K1_O2_TAIL.as_ref(),
            ),
            (2, b) if b == cantor::SUBSPACE[4] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O4,
                cantor::IFFT_DIT4_NIBBLE_K2_O4_TAIL.as_ref(),
            ),
            (3, b) if b == cantor::SUBSPACE[8] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O8,
                cantor::IFFT_DIT4_NIBBLE_K3_O8_TAIL.as_ref(),
            ),
            (4, b) if b == cantor::SUBSPACE[16] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O16,
                cantor::IFFT_DIT4_NIBBLE_K4_O16_TAIL.as_ref(),
            ),
            (5, b) if b == cantor::SUBSPACE[32] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O32,
                cantor::IFFT_DIT4_NIBBLE_K5_O32_TAIL.as_ref(),
            ),
            (6, b) if b == cantor::SUBSPACE[64] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O64,
                cantor::IFFT_DIT4_NIBBLE_K6_O64_TAIL.as_ref(),
            ),
            (7, b) if b == cantor::SUBSPACE[128] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O128,
                cantor::IFFT_DIT4_NIBBLE_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, t);
            }
        }
    }
}

#[target_feature(enable = "avx2")]
fn fft_sharded_standard(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K1_O0,
                &standard::FFT_DIT4_NIBBLE_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K2_O0,
                &standard::FFT_DIT4_NIBBLE_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K3_O0,
                &standard::FFT_DIT4_NIBBLE_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K4_O0,
                &standard::FFT_DIT4_NIBBLE_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K5_O0,
                &standard::FFT_DIT4_NIBBLE_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K6_O0,
                &standard::FFT_DIT4_NIBBLE_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K7_O0,
                &standard::FFT_DIT4_NIBBLE_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K8_O0,
                &standard::FFT_DIT4_NIBBLE_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "avx2")]
fn fft_sharded_radix2_first_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    beta: Gf2p8_11d,
) {
    type K = Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O0,
                standard::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O0,
                standard::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O0,
                standard::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O0,
                standard::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O0,
                standard::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O0,
                standard::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O0,
                standard::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K8_O0,
                standard::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "avx2")]
fn ifft_sharded_standard(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O0,
                standard::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O0,
                standard::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O0,
                standard::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O0,
                standard::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O0,
                standard::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O0,
                standard::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O0,
                standard::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K8_O0,
                standard::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        match (k, u8::from(beta)) {
            (0, _) => {}
            (1, b) if b == standard::SUBSPACE[2] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O2,
                standard::IFFT_DIT4_NIBBLE_K1_O2_TAIL.as_ref(),
            ),
            (2, b) if b == standard::SUBSPACE[4] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O4,
                standard::IFFT_DIT4_NIBBLE_K2_O4_TAIL.as_ref(),
            ),
            (3, b) if b == standard::SUBSPACE[8] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O8,
                standard::IFFT_DIT4_NIBBLE_K3_O8_TAIL.as_ref(),
            ),
            (4, b) if b == standard::SUBSPACE[16] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O16,
                standard::IFFT_DIT4_NIBBLE_K4_O16_TAIL.as_ref(),
            ),
            (5, b) if b == standard::SUBSPACE[32] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O32,
                standard::IFFT_DIT4_NIBBLE_K5_O32_TAIL.as_ref(),
            ),
            (6, b) if b == standard::SUBSPACE[64] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O64,
                standard::IFFT_DIT4_NIBBLE_K6_O64_TAIL.as_ref(),
            ),
            (7, b) if b == standard::SUBSPACE[128] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O128,
                standard::IFFT_DIT4_NIBBLE_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, t);
            }
        }
    }
}

#[derive(Default)]
pub struct Avx2Kernel<G: Gf2p8Lut, B: SubspaceBasisLut<G>>(PhantomData<(G, B)>);

impl Kernel<Gf2p8_11d, CantorBasisLut11d> for Avx2Kernel<Gf2p8_11d, CantorBasisLut11d> {
    const ALIGN: usize = 32;

    type Multiplier = Multiplier;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &cantor::NODE_MUL_NIBBLE;

    fn mul_table(t: Gf2p8_11d) -> Self::Multiplier {
        &NIBBLE_MUL_TABLE[t.into_usize()]
    }

    fn is_zero_mul(m: &Self::Multiplier) -> bool {
        m[0][1] == 0
    }

    fn butterfly_fwd_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit2(shards, shard_len, base, d, m);
        }
    }

    fn butterfly_fwd_dit2_zero(shards: &mut [Gf2p8_11d], shard_len: usize, base: usize, d: usize) {
        unsafe {
            butterfly_fwd_dit2_zero(shards, shard_len, base, d);
        }
    }

    fn butterfly_fwd_dit4(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit4(shards, shard_len, base, d, m01, m23, m02);
        }
    }

    fn butterfly_fwd_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit4_zero(shards, shard_len, base, d, m23);
        }
    }

    fn butterfly_inv_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit2(shards, shard_len, base, d, m);
        }
    }

    fn butterfly_inv_dit4(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit4(shards, shard_len, base, d, m01, m23, m02);
        }
    }

    fn butterfly_inv_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit4_zero(shards, shard_len, base, d, m23);
        }
    }

    fn fft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            fft_sharded_cantor(shards, shard_len, k, beta);
        }
    }

    fn fft_sharded_radix2_first(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        unsafe {
            fft_sharded_radix2_first_cantor(shards, shard_len, k, beta);
        }
    }

    fn ifft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            ifft_sharded_cantor(shards, shard_len, k, beta);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale(src, dst, dst.len(), m);
        }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale_add(src, dst, dst.len(), m);
        }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale_in_place(dst, dst.len(), m);
        }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

impl Kernel<Gf2p8_11d, StandardBasisLut11d> for Avx2Kernel<Gf2p8_11d, StandardBasisLut11d> {
    const ALIGN: usize = 32;

    type Multiplier = Multiplier;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &standard::NODE_MUL_NIBBLE;

    fn mul_table(t: Gf2p8_11d) -> Self::Multiplier {
        &NIBBLE_MUL_TABLE[t.into_usize()]
    }

    fn is_zero_mul(m: &Self::Multiplier) -> bool {
        m[0][1] == 0
    }

    fn butterfly_fwd_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit2(shards, shard_len, base, d, m);
        }
    }

    fn butterfly_fwd_dit2_zero(shards: &mut [Gf2p8_11d], shard_len: usize, base: usize, d: usize) {
        unsafe {
            butterfly_fwd_dit2_zero(shards, shard_len, base, d);
        }
    }

    fn butterfly_fwd_dit4(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit4(shards, shard_len, base, d, m01, m23, m02);
        }
    }

    fn butterfly_fwd_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::Multiplier,
    ) {
        unsafe {
            butterfly_fwd_dit4_zero(shards, shard_len, base, d, m23);
        }
    }

    fn butterfly_inv_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit2(shards, shard_len, base, d, m);
        }
    }

    fn butterfly_inv_dit4(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit4(shards, shard_len, base, d, m01, m23, m02);
        }
    }

    fn butterfly_inv_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::Multiplier,
    ) {
        unsafe {
            butterfly_inv_dit4_zero(shards, shard_len, base, d, m23);
        }
    }

    fn fft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            fft_sharded_standard(shards, shard_len, k, beta);
        }
    }

    fn fft_sharded_radix2_first(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        unsafe {
            fft_sharded_radix2_first_standard(shards, shard_len, k, beta);
        }
    }

    fn ifft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            ifft_sharded_standard(shards, shard_len, k, beta);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale(src, dst, dst.len(), m);
        }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale_add(src, dst, dst.len(), m);
        }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        unsafe {
            scale_in_place(dst, dst.len(), m);
        }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

#[cfg(test)]
#[cfg(native_avx2)]
mod test {
    use super::*;
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_cfg() {
        let target = cfg!(target_arch = "x86_64");
        let feature = is_x86_feature_detected!("avx2");

        assert!(target);
        assert!(feature);
    }

    #[test]
    fn mul_vec_matches_mul_lut() {
        for p in 0..=255u8 {
            let m = &NIBBLE_MUL_TABLE[p as usize];
            for x in 0..=255u8 {
                let expected = Gf2p8_11d(x).mul_lut(Gf2p8_11d(p));

                let actual = unsafe {
                    let v = _mm256_set1_epi8(x as i8); // copy x cross all 32 lanes
                    let (m_lo, m_hi) = load_mul_table(m);
                    let r = mul_vec_inner(v, m_lo, m_hi);
                    let byte0 = _mm256_extract_epi8::<0>(r) as u8;
                    let byte16 = _mm256_extract_epi8::<16>(r) as u8;
                    assert_eq!(byte0, byte16, "mul table not broadcast");
                    Gf2p8_11d(byte0)
                };
                assert_eq!(expected, actual, "p={p:02x} x={x:02x}");
            }
        }
    }
}

#[cfg(test)]
#[cfg(native_avx2)]
mod test_cantor {
    use crate::poly_11d_lut::CantorBasisLut11d;

    type B = CantorBasisLut11d;
    include!("avx2_kernel_tests.rs");
}

#[cfg(test)]
#[cfg(native_avx2)]
mod test_standard {
    use crate::poly_11d_lut::StandardBasisLut11d;

    type B = StandardBasisLut11d;
    include!("avx2_kernel_tests.rs");
}

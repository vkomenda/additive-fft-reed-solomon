use super::{Kernel, shard_groups};
use crate::{
    gf2p8lut::{Gf2p8Lut, SubspaceBasisLut},
    poly_11d_lut::{
        CantorBasisLut11d, StandardBasisLut11d,
        generated::{GFNI_MUL_BY_LOG, cantor, standard},
    },
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, Z255};
use core::arch::x86_64::*;
use std::marker::PhantomData;

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd_dit2<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize, m: u64) {
    let m = _mm512_set1_epi64(m as i64);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
                let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
                let t = _mm512_gf2p8affine_epi64_epi8(vb, m, 0); // T·b
                let va = _mm512_xor_si512(va, t); // a + T·b = g0
                let vb = _mm512_xor_si512(vb, va); // b + g0 = g1
                _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
                _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
            }
            i += 64;
        }
        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let va = _mm512_maskz_loadu_epi8(k, a.add(i) as *const i8);
                let vb = _mm512_maskz_loadu_epi8(k, b.add(i) as *const i8);
                let t = _mm512_gf2p8affine_epi64_epi8(vb, m, 0);
                let va = _mm512_xor_si512(va, t);
                let vb = _mm512_xor_si512(vb, va);
                _mm512_mask_storeu_epi8(a.add(i) as *mut i8, k, va);
                _mm512_mask_storeu_epi8(b.add(i) as *mut i8, k, vb);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd_dit2_zero<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
                let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
                let vb = _mm512_xor_si512(vb, va); // b + g0 = g1
                _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
            }
            i += 64;
        }
        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let va = _mm512_maskz_loadu_epi8(k, a.add(i) as *const i8);
                let vb = _mm512_maskz_loadu_epi8(k, b.add(i) as *const i8);
                let vb = _mm512_xor_si512(vb, va);
                _mm512_mask_storeu_epi8(b.add(i) as *mut i8, k, vb);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_inv_dit2<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize, m: u64) {
    let m = _mm512_set1_epi64(m as i64);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
                let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
                let vb = _mm512_xor_si512(vb, va); // d' = g0 + g1
                let t = _mm512_gf2p8affine_epi64_epi8(vb, m, 0); // T·d'
                let va = _mm512_xor_si512(va, t); // d = g0 + T·d'
                _mm512_storeu_si512(a.add(i) as *mut __m512i, va);
                _mm512_storeu_si512(b.add(i) as *mut __m512i, vb);
            }
            i += 64;
        }
        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let va = _mm512_maskz_loadu_epi8(k, a.add(i) as *const i8);
                let vb = _mm512_maskz_loadu_epi8(k, b.add(i) as *const i8);
                let vb = _mm512_xor_si512(vb, va);
                let t = _mm512_gf2p8affine_epi64_epi8(vb, m, 0);
                let va = _mm512_xor_si512(va, t);
                _mm512_mask_storeu_epi8(a.add(i) as *mut i8, k, va);
                _mm512_mask_storeu_epi8(b.add(i) as *mut i8, k, vb);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: u64,
    m23: u64,
    m02: u64,
) {
    let m01 = _mm512_set1_epi64(m01 as i64);
    let m23 = _mm512_set1_epi64(m23 as i64);
    let m02 = _mm512_set1_epi64(m02 as i64);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let mut w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w2, m02, 0));
                w1 = _mm512_xor_si512(w1, _mm512_gf2p8affine_epi64_epi8(w3, m02, 0));
                w2 = _mm512_xor_si512(w2, w0);
                w3 = _mm512_xor_si512(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w1, m01, 0));
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w1 = _mm512_xor_si512(w1, w0);
                w3 = _mm512_xor_si512(w3, w2);

                _mm512_storeu_si512(p0.add(i) as *mut __m512i, w0);
                _mm512_storeu_si512(p1.add(i) as *mut __m512i, w1);
                _mm512_storeu_si512(p2.add(i) as *mut __m512i, w2);
                _mm512_storeu_si512(p3.add(i) as *mut __m512i, w3);
            }
            i += 64;
        }

        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let mut w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w2, m02, 0));
                w1 = _mm512_xor_si512(w1, _mm512_gf2p8affine_epi64_epi8(w3, m02, 0));
                w2 = _mm512_xor_si512(w2, w0);
                w3 = _mm512_xor_si512(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w1, m01, 0));
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w1 = _mm512_xor_si512(w1, w0);
                w3 = _mm512_xor_si512(w3, w2);

                _mm512_mask_storeu_epi8(p0.add(i) as *mut i8, k, w0);
                _mm512_mask_storeu_epi8(p1.add(i) as *mut i8, k, w1);
                _mm512_mask_storeu_epi8(p2.add(i) as *mut i8, k, w2);
                _mm512_mask_storeu_epi8(p3.add(i) as *mut i8, k, w3);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: u64,
) {
    let m23 = _mm512_set1_epi64(m23 as i64);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w2 = _mm512_xor_si512(w2, w0);
                w3 = _mm512_xor_si512(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w1 = _mm512_xor_si512(w1, w0);
                w3 = _mm512_xor_si512(w3, w2);

                _mm512_storeu_si512(p1.add(i) as *mut __m512i, w1);
                _mm512_storeu_si512(p2.add(i) as *mut __m512i, w2);
                _mm512_storeu_si512(p3.add(i) as *mut __m512i, w3);
            }
            i += 64;
        }

        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w2 = _mm512_xor_si512(w2, w0);
                w3 = _mm512_xor_si512(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w1 = _mm512_xor_si512(w1, w0);
                w3 = _mm512_xor_si512(w3, w2);

                _mm512_mask_storeu_epi8(p1.add(i) as *mut i8, k, w1);
                _mm512_mask_storeu_epi8(p2.add(i) as *mut i8, k, w2);
                _mm512_mask_storeu_epi8(p3.add(i) as *mut i8, k, w3);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_inv_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: u64,
    m23: u64,
    m02: u64,
) {
    let m01 = _mm512_set1_epi64(m01 as i64);
    let m23 = _mm512_set1_epi64(m23 as i64);
    let m02 = _mm512_set1_epi64(m02 as i64);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let mut w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm512_xor_si512(w3, w2);
                w1 = _mm512_xor_si512(w1, w0);
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w1, m01, 0));

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w3 = _mm512_xor_si512(w3, w1);
                w2 = _mm512_xor_si512(w2, w0);
                w1 = _mm512_xor_si512(w1, _mm512_gf2p8affine_epi64_epi8(w3, m02, 0));
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w2, m02, 0));

                _mm512_storeu_si512(p0.add(i) as *mut __m512i, w0);
                _mm512_storeu_si512(p1.add(i) as *mut __m512i, w1);
                _mm512_storeu_si512(p2.add(i) as *mut __m512i, w2);
                _mm512_storeu_si512(p3.add(i) as *mut __m512i, w3);
            }
            i += 64;
        }

        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let mut w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm512_xor_si512(w3, w2);
                w1 = _mm512_xor_si512(w1, w0);
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w1, m01, 0));

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w3 = _mm512_xor_si512(w3, w1);
                w2 = _mm512_xor_si512(w2, w0);
                w1 = _mm512_xor_si512(w1, _mm512_gf2p8affine_epi64_epi8(w3, m02, 0));
                w0 = _mm512_xor_si512(w0, _mm512_gf2p8affine_epi64_epi8(w2, m02, 0));

                _mm512_mask_storeu_epi8(p0.add(i) as *mut i8, k, w0);
                _mm512_mask_storeu_epi8(p1.add(i) as *mut i8, k, w1);
                _mm512_mask_storeu_epi8(p2.add(i) as *mut i8, k, w2);
                _mm512_mask_storeu_epi8(p3.add(i) as *mut i8, k, w3);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_inv_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: u64,
) {
    let m23 = _mm512_set1_epi64(m23 as i64);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 64 <= shard_len {
            unsafe {
                let w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm512_xor_si512(w3, w2);
                w1 = _mm512_xor_si512(w1, w0);
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w3 = _mm512_xor_si512(w3, w1);
                w2 = _mm512_xor_si512(w2, w0);

                _mm512_storeu_si512(p1.add(i) as *mut __m512i, w1);
                _mm512_storeu_si512(p2.add(i) as *mut __m512i, w2);
                _mm512_storeu_si512(p3.add(i) as *mut __m512i, w3);
            }
            i += 64;
        }

        if i < shard_len {
            let k = (1u64 << (shard_len - i)) - 1;
            unsafe {
                let w0 = _mm512_loadu_si512(p0.add(i) as *const __m512i);
                let mut w1 = _mm512_loadu_si512(p1.add(i) as *const __m512i);
                let mut w2 = _mm512_loadu_si512(p2.add(i) as *const __m512i);
                let mut w3 = _mm512_loadu_si512(p3.add(i) as *const __m512i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm512_xor_si512(w3, w2);
                w1 = _mm512_xor_si512(w1, w0);
                w2 = _mm512_xor_si512(w2, _mm512_gf2p8affine_epi64_epi8(w3, m23, 0));

                // Wide level: (w0, w2) and (w1, w3), both with m02. The two chains
                // are independent, so the multiplies are issued together.
                w3 = _mm512_xor_si512(w3, w1);
                w2 = _mm512_xor_si512(w2, w0);

                _mm512_mask_storeu_epi8(p1.add(i) as *mut i8, k, w1);
                _mm512_mask_storeu_epi8(p2.add(i) as *mut i8, k, w2);
                _mm512_mask_storeu_epi8(p3.add(i) as *mut i8, k, w3);
            }
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, mat: __m512i) {
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
fn scale_add<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, mat: __m512i) {
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let v = _mm512_loadu_si512(src.add(i) as *const __m512i);
            let mut w = _mm512_loadu_si512(dst.add(i) as *const __m512i);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            w = _mm512_xor_si512(w, r);
            _mm512_storeu_si512(dst.add(i) as *mut __m512i, w);
        }
        i += 64;
    }
    if i < len {
        let k = (1u64 << (len - i)) - 1;
        unsafe {
            let v = _mm512_maskz_loadu_epi8(k, src.add(i) as *const i8);
            let mut w = _mm512_maskz_loadu_epi8(k, dst.add(i) as *const i8);
            let r = _mm512_gf2p8affine_epi64_epi8(v, mat, 0);
            w = _mm512_xor_si512(w, r);
            _mm512_mask_storeu_epi8(dst.add(i) as *mut i8, k, w);
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

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_dit4_inner_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K1_O0,
                &cantor::FFT_DIT4_GFNI_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K2_O0,
                &cantor::FFT_DIT4_GFNI_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K3_O0,
                &cantor::FFT_DIT4_GFNI_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K4_O0,
                &cantor::FFT_DIT4_GFNI_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K5_O0,
                &cantor::FFT_DIT4_GFNI_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K6_O0,
                &cantor::FFT_DIT4_GFNI_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K7_O0,
                &cantor::FFT_DIT4_GFNI_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_GFNI_K8_O0,
                &cantor::FFT_DIT4_GFNI_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 1..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_dit4_outer_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K1_O0,
                cantor::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K2_O0,
                cantor::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K3_O0,
                cantor::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K4_O0,
                cantor::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K5_O0,
                cantor::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K6_O0,
                cantor::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K7_O0,
                cantor::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K8_O0,
                cantor::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn ifft_sharded_dit4_outer_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K1_O0,
                cantor::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K2_O0,
                cantor::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K3_O0,
                cantor::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K4_O0,
                cantor::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K5_O0,
                cantor::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K6_O0,
                cantor::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K7_O0,
                cantor::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_GFNI_K8_O0,
                cantor::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::ifft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_dit4_inner_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K1_O0,
                &standard::FFT_DIT4_GFNI_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K2_O0,
                &standard::FFT_DIT4_GFNI_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K3_O0,
                &standard::FFT_DIT4_GFNI_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K4_O0,
                &standard::FFT_DIT4_GFNI_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K5_O0,
                &standard::FFT_DIT4_GFNI_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K6_O0,
                &standard::FFT_DIT4_GFNI_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K7_O0,
                &standard::FFT_DIT4_GFNI_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_GFNI_K8_O0,
                &standard::FFT_DIT4_GFNI_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 1..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_dit4_outer_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K1_O0,
                standard::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K2_O0,
                standard::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K3_O0,
                standard::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K4_O0,
                standard::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K5_O0,
                standard::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K6_O0,
                standard::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K7_O0,
                standard::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K8_O0,
                standard::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn ifft_sharded_dit4_outer_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = GfniKernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K1_O0,
                standard::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K2_O0,
                standard::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K3_O0,
                standard::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K4_O0,
                standard::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K5_O0,
                standard::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K6_O0,
                standard::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K7_O0,
                standard::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_GFNI_K8_O0,
                standard::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::ifft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[derive(Default)]
pub struct GfniKernel<G: Gf2p8Lut, B: SubspaceBasisLut<G>>(PhantomData<(G, B)>);

impl Kernel<Gf2p8_11d, CantorBasisLut11d> for GfniKernel<Gf2p8_11d, CantorBasisLut11d> {
    const ALIGN: usize = 64;

    type Multiplier = u64;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &cantor::NODE_MUL_GFNI;

    fn mul_table(twiddle: Gf2p8_11d) -> Self::Multiplier {
        twiddle.gfni_mul_matrix_lut()
    }

    fn is_zero_mul(m: &Self::Multiplier) -> bool {
        *m == 0
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

    fn fft_sharded_dit4_inner_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            fft_sharded_dit4_inner_tail_cantor(shards, shard_len, k, coords);
        }
    }

    fn fft_sharded_dit4_outer_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            fft_sharded_dit4_outer_tail_cantor(shards, shard_len, k, coords);
        }
    }

    fn ifft_sharded_dit4_outer_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            ifft_sharded_dit4_outer_tail_cantor(shards, shard_len, k, coords);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale(src, dst, dst.len(), mat) }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale_add(src, dst, dst.len(), mat) }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale_in_place(dst, dst.len(), mat) }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = unsafe { _mm512_set1_epi64(GFNI_MUL_BY_LOG[log_m.0 as usize] as i64) };
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = unsafe { _mm512_set1_epi64(GFNI_MUL_BY_LOG[log_m.0 as usize] as i64) };
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

impl Kernel<Gf2p8_11d, StandardBasisLut11d> for GfniKernel<Gf2p8_11d, StandardBasisLut11d> {
    const ALIGN: usize = 64;

    type Multiplier = u64;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &standard::NODE_MUL_GFNI;

    fn mul_table(twiddle: Gf2p8_11d) -> Self::Multiplier {
        twiddle.gfni_mul_matrix_lut()
    }

    fn is_zero_mul(m: &Self::Multiplier) -> bool {
        *m == 0
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

    fn fft_sharded_dit4_inner_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            fft_sharded_dit4_inner_tail_standard(shards, shard_len, k, coords);
        }
    }

    fn fft_sharded_dit4_outer_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            fft_sharded_dit4_outer_tail_standard(shards, shard_len, k, coords);
        }
    }

    fn ifft_sharded_dit4_outer_tail(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        coords: usize,
    ) {
        unsafe {
            ifft_sharded_dit4_outer_tail_standard(shards, shard_len, k, coords);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale(src, dst, dst.len(), mat) }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale_add(src, dst, dst.len(), mat) }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale_in_place(dst, dst.len(), mat) }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = unsafe { _mm512_set1_epi64(GFNI_MUL_BY_LOG[log_m.0 as usize] as i64) };
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = unsafe { _mm512_set1_epi64(GFNI_MUL_BY_LOG[log_m.0 as usize] as i64) };
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

#[cfg(test)]
#[cfg(native_gfni)]
mod test {
    #[test]
    fn debug_cfg() {
        let target_arch_x86_64 = cfg!(target_arch = "x86_64");
        let avx512f = is_x86_feature_detected!("avx512f");
        let avx512bw = is_x86_feature_detected!("avx512bw");
        let gfni = is_x86_feature_detected!("gfni");

        assert!(target_arch_x86_64);
        assert!(avx512f);
        assert!(avx512bw);
        assert!(gfni);
    }
}

#[cfg(test)]
#[cfg(native_gfni)]
mod test_cantor {
    use crate::poly_11d_lut::CantorBasisLut11d;

    type B = CantorBasisLut11d;
    include!("gfni_kernel_tests.rs");
}

#[cfg(test)]
#[cfg(native_gfni)]
mod test_standard {
    use crate::poly_11d_lut::StandardBasisLut11d;

    type B = StandardBasisLut11d;
    include!("gfni_kernel_tests.rs");
}

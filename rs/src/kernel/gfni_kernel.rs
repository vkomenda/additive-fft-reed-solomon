use super::{Kernel, shard_groups};
use crate::{
    gf2p8lut::{CantorBasisLut, Gf2p8Lut},
    poly_11d_lut::generated::{self as tables, CANTOR_SUBSPACE, GFNI_MUL_BY_LOG, GFNI_MUL_TABLE},
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, Z255};
use core::arch::x86_64::*;
use std::marker::PhantomData;

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_gfni_kernel_11d.rs"));
}

/// Forward butterfly transforming (a, b) into (a + T·b, b + a + T·b).
#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, mat: __m512i) {
    let a = a.as_mut_ptr();
    let b = b.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·b
            let va = _mm512_xor_si512(va, t); // a + T·b = g0
            let vb = _mm512_xor_si512(vb, va); // b + g0 = g1
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
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, mat: __m512i) {
    let a = a.as_mut_ptr();
    let b = b.as_mut_ptr();
    let mut i = 0;
    while i + 64 <= len {
        unsafe {
            let va = _mm512_loadu_si512(a.add(i) as *const __m512i);
            let vb = _mm512_loadu_si512(b.add(i) as *const __m512i);
            let vb = _mm512_xor_si512(vb, va); // d' = g0 + g1
            let t = _mm512_gf2p8affine_epi64_epi8(vb, mat, 0); // T·d'
            let va = _mm512_xor_si512(va, t); // d = g0 + T·d'
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
    let mat = _mm512_set1_epi64(twiddle.gfni_mul_matrix_lut() as i64);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_fwd(
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
    let mat = _mm512_set1_epi64(twiddle.gfni_mul_matrix_lut() as i64);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_inv(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            mat,
        )
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
fn fft_sharded_iterative(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    if beta == Gf2p8_11d::zero() {
        match k {
            1 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K1_O0,
                &tables::FFT_DIT4_GFNI_K1_O0_TAIL,
            ),
            2 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K2_O0,
                &tables::FFT_DIT4_GFNI_K2_O0_TAIL,
            ),
            3 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K3_O0,
                &tables::FFT_DIT4_GFNI_K3_O0_TAIL,
            ),
            4 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K4_O0,
                &tables::FFT_DIT4_GFNI_K4_O0_TAIL,
            ),
            5 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K5_O0,
                &tables::FFT_DIT4_GFNI_K5_O0_TAIL,
            ),
            6 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K6_O0,
                &tables::FFT_DIT4_GFNI_K6_O0_TAIL,
            ),
            7 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K7_O0,
                &tables::FFT_DIT4_GFNI_K7_O0_TAIL,
            ),
            8 => Self::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_GFNI_K8_O0,
                &tables::FFT_DIT4_GFNI_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 1..=8"),
        }
    } else {
        todo!();
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn fft_sharded_iterative_mirrored(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    beta: Gf2p8_11d,
) {
    if beta == Gf2p8_11d::zero() {
        match k {
            1 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K1_O0,
                tables::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K2_O0,
                tables::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K3_O0,
                tables::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K4_O0,
                tables::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K5_O0,
                tables::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K6_O0,
                tables::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K7_O0,
                tables::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => Self::fft_sharded_dit4_mirrored_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K8_O0,
                tables::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 1..=8"),
        }
    } else {
        todo!();
    }
}

#[target_feature(enable = "avx512f,avx512bw,gfni")]
fn ifft_sharded_iterative(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    if beta == Gf2p8_11d::zero() {
        match k {
            1 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K1_O0,
                tables::IFFT_DIT4_GFNI_K1_O0_TAIL.as_ref(),
            ),
            2 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K2_O0,
                tables::IFFT_DIT4_GFNI_K2_O0_TAIL.as_ref(),
            ),
            3 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K3_O0,
                tables::IFFT_DIT4_GFNI_K3_O0_TAIL.as_ref(),
            ),
            4 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K4_O0,
                tables::IFFT_DIT4_GFNI_K4_O0_TAIL.as_ref(),
            ),
            5 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K5_O0,
                tables::IFFT_DIT4_GFNI_K5_O0_TAIL.as_ref(),
            ),
            6 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K6_O0,
                tables::IFFT_DIT4_GFNI_K6_O0_TAIL.as_ref(),
            ),
            7 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K7_O0,
                tables::IFFT_DIT4_GFNI_K7_O0_TAIL.as_ref(),
            ),
            8 => Self::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_GFNI_K8_O0,
                tables::IFFT_DIT4_GFNI_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 1..=8"),
        }
    } else {
        todo!();
    }
}

#[derive(Default)]
pub struct GfniKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for GfniKernel<Gf2p8_11d> {
    const ALIGN: usize = 64;

    type MulTable = u64;

    fn mul_table(twiddle: Gf2p8_11d) -> Self::MulTable {
        twiddle.gfni_mul_matrix_lut()
    }

    fn is_zero_mul(m: &Self::MulTable) -> bool {
        *m == 0
    }

    fn butterfly_fwd_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::MulTable,
    ) {
        unsafe {
            butterfly_fwd_dit2(shards, shard_len, base, d, m);
        }
    }

    fn butterfly_fwd_dit4(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::MulTable,
        m23: Self::MulTable,
        m02: Self::MulTable,
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
        m23: Self::MulTable,
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
        m: Self::MulTable,
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
        m01: Self::MulTable,
        m23: Self::MulTable,
        m02: Self::MulTable,
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
        m23: Self::MulTable,
    ) {
        unsafe {
            butterfly_inv_dit4_zero(shards, shard_len, base, d, m23);
        }
    }

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

    fn fft_sharded_zero_padded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, log_support: u8) {
        unsafe {
            match (k, log_support) {
                (1, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_2_1(shards, shard_len),
                (2, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_4_1(shards, shard_len),
                (2, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_4_2(shards, shard_len),
                (3, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_8_1(shards, shard_len),
                (3, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_8_2(shards, shard_len),
                (3, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_8_4(shards, shard_len),
                (4, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_16_1(shards, shard_len),
                (4, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_16_2(shards, shard_len),
                (4, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_16_4(shards, shard_len),
                (4, 3) => unrolled_11d::fft_sharded_zero_padded_gfni_16_8(shards, shard_len),
                (5, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_32_1(shards, shard_len),
                (5, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_32_2(shards, shard_len),
                (5, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_32_4(shards, shard_len),
                (5, 3) => unrolled_11d::fft_sharded_zero_padded_gfni_32_8(shards, shard_len),
                (5, 4) => unrolled_11d::fft_sharded_zero_padded_gfni_32_16(shards, shard_len),
                (6, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_64_1(shards, shard_len),
                (6, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_64_2(shards, shard_len),
                (6, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_64_4(shards, shard_len),
                (6, 3) => unrolled_11d::fft_sharded_zero_padded_gfni_64_8(shards, shard_len),
                (6, 4) => unrolled_11d::fft_sharded_zero_padded_gfni_64_16(shards, shard_len),
                (6, 5) => unrolled_11d::fft_sharded_zero_padded_gfni_64_32(shards, shard_len),
                (7, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_128_1(shards, shard_len),
                (7, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_128_2(shards, shard_len),
                (7, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_128_4(shards, shard_len),
                (7, 3) => unrolled_11d::fft_sharded_zero_padded_gfni_128_8(shards, shard_len),
                (7, 4) => unrolled_11d::fft_sharded_zero_padded_gfni_128_16(shards, shard_len),
                (7, 5) => unrolled_11d::fft_sharded_zero_padded_gfni_128_32(shards, shard_len),
                (7, 6) => unrolled_11d::fft_sharded_zero_padded_gfni_128_64(shards, shard_len),
                (8, 0) => unrolled_11d::fft_sharded_zero_padded_gfni_256_1(shards, shard_len),
                (8, 1) => unrolled_11d::fft_sharded_zero_padded_gfni_256_2(shards, shard_len),
                (8, 2) => unrolled_11d::fft_sharded_zero_padded_gfni_256_4(shards, shard_len),
                (8, 3) => unrolled_11d::fft_sharded_zero_padded_gfni_256_8(shards, shard_len),
                (8, 4) => unrolled_11d::fft_sharded_zero_padded_gfni_256_16(shards, shard_len),
                (8, 5) => unrolled_11d::fft_sharded_zero_padded_gfni_256_32(shards, shard_len),
                (8, 6) => unrolled_11d::fft_sharded_zero_padded_gfni_256_64(shards, shard_len),
                (8, 7) => unrolled_11d::fft_sharded_zero_padded_gfni_256_128(shards, shard_len),
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

    fn fft_sharded_iterative(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            fft_sharded_iterative(shards, shard_len, k, beta);
        }
    }

    fn fft_sharded_iterative_mirrored(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        unsafe {
            fft_sharded_iterative_mirrored(shards, shard_len, k, beta);
        }
    }

    fn ifft_sharded_iterative(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        unsafe {
            ifft_sharded_iterative(shards, shard_len, k, beta);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let mat = unsafe { _mm512_set1_epi64(scalar.gfni_mul_matrix_lut() as i64) };
        unsafe { scale(src, dst, dst.len(), mat) }
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

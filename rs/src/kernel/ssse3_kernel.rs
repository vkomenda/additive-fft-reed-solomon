use super::{Dit2Node, Dit4Node, Kernel, ShardTuple};
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
#[target_feature(enable = "ssse3")]
fn load<T: Copy>(chunk: &[T; 16]) -> __m128i {
    // Panic at compile time if the element type is not a byte.
    const { assert!(size_of::<T>() == 1) };
    // SAFETY: `chunk` is a valid reference to 16 bytes. loadu has no alignment requirement.
    unsafe { _mm_loadu_si128(chunk.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn store<G: Gf2p8>(chunk: &mut [G; 16], v: __m128i) {
    // SAFETY: `chunk` is a valid reference to 16 bytes. storeu has no alignment requirement.
    unsafe { _mm_storeu_si128(chunk.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "ssse3")]
fn mul_vec_inner(v: __m128i, m_lo: __m128i, m_hi: __m128i) -> __m128i {
    let mask = _mm_set1_epi8(0x0f);
    let lo = _mm_shuffle_epi8(m_lo, _mm_and_si128(v, mask));
    let hi = _mm_shuffle_epi8(m_hi, _mm_and_si128(_mm_srli_epi16::<4>(v), mask));
    _mm_xor_si128(lo, hi)
}

#[inline]
#[target_feature(enable = "ssse3")]
fn load_mul_table(m: Multiplier) -> (__m128i, __m128i) {
    (load(&m[0]), load(&m[1]))
}

#[target_feature(enable = "ssse3")]
fn butterfly_fwd_dit2<G: Gf2p8>(node: Dit2Node<'_, G>, m: Multiplier) {
    let (m_lo, m_hi) = load_mul_table(m);

    for tup in node.split() {
        let (chunks, [tail0, tail1]) = tup.into_chunks::<16>();
        for [p0, p1] in chunks {
            let mut w1 = load(p1);
            let w0 = _mm_xor_si128(load(p0), mul_vec_inner(w1, m_lo, m_hi));
            w1 = _mm_xor_si128(w1, w0);
            store(p0, w0);
            store(p1, w1);
        }
        for (x, y) in tail0.iter_mut().zip(tail1) {
            let g0 = x.add(y.nibble_mul(m));
            *x = g0;
            *y = y.add(g0);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_fwd_dit2_zero<G: Gf2p8>(node: Dit2Node<'_, G>) {
    for tup in node.split() {
        let (chunks, [tail0, tail1]) = tup.into_chunks::<16>();
        for [p0, p1] in chunks {
            let w1 = _mm_xor_si128(load(p1), load(p0));
            store(p1, w1);
        }
        for (x, y) in tail0.iter_mut().zip(tail1) {
            *y = y.add(*x);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_inv_dit2<G: Gf2p8>(node: Dit2Node<'_, G>, m: Multiplier) {
    let (m_lo, m_hi) = load_mul_table(m);

    for tup in node.split() {
        let (chunks, [tail0, tail1]) = tup.into_chunks::<16>();
        for [p0, p1] in chunks {
            let mut w0 = load(p0);
            let w1 = _mm_xor_si128(load(p1), w0);
            w0 = _mm_xor_si128(w0, mul_vec_inner(w1, m_lo, m_hi));
            store(p0, w0);
            store(p1, w1);
        }
        for (x, y) in tail0.iter_mut().zip(tail1) {
            *y = x.add(*y);
            *x = x.add(y.nibble_mul(m));
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_fwd_dit4<G: Gf2p8>(
    node: Dit4Node<'_, G>,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let (m02_lo, m02_hi) = load_mul_table(m02);
    let (m01_lo, m01_hi) = load_mul_table(m01);
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for tup in node.split() {
        let (chunks, [tail0, tail1, tail2, tail3]) = tup.into_chunks::<16>();
        for [p0, p1, p2, p3] in chunks {
            let mut w0 = load(p0);
            let mut w1 = load(p1);
            let mut w2 = load(p2);
            let mut w3 = load(p3);

            // Wide level: (w0, w2) and (w1, w3), both with m02.
            w0 = _mm_xor_si128(w0, mul_vec_inner(w2, m02_lo, m02_hi));
            w1 = _mm_xor_si128(w1, mul_vec_inner(w3, m02_lo, m02_hi));
            w2 = _mm_xor_si128(w2, w0);
            w3 = _mm_xor_si128(w3, w1);

            // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
            w0 = _mm_xor_si128(w0, mul_vec_inner(w1, m01_lo, m01_hi));
            w2 = _mm_xor_si128(w2, mul_vec_inner(w3, m23_lo, m23_hi));
            w1 = _mm_xor_si128(w1, w0);
            w3 = _mm_xor_si128(w3, w2);

            store(p0, w0);
            store(p1, w1);
            store(p2, w2);
            store(p3, w3);
        }

        for (((x0, x1), x2), x3) in tail0.iter_mut().zip(tail1).zip(tail2).zip(tail3) {
            let (mut w0, mut w1, mut w2, mut w3) = (*x0, *x1, *x2, *x3);
            w0 = w0.add(w2.nibble_mul(m02));
            w1 = w1.add(w3.nibble_mul(m02));
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w0 = w0.add(w1.nibble_mul(m01));
            w2 = w2.add(w3.nibble_mul(m23));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            (*x0, *x1, *x2, *x3) = (w0, w1, w2, w3);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_fwd_dit4_zero<G: Gf2p8>(node: Dit4Node<'_, G>, m23: Multiplier) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for tup in node.split() {
        let (chunks, [tail0, tail1, tail2, tail3]) = tup.into_chunks::<16>();
        for [p0, p1, p2, p3] in chunks {
            let w0 = load(p0);
            let mut w1 = load(p1);
            let mut w2 = load(p2);
            let mut w3 = load(p3);

            // Wide level: (w0, w2) and (w1, w3), both with m02.
            w2 = _mm_xor_si128(w2, w0);
            w3 = _mm_xor_si128(w3, w1);

            // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
            w2 = _mm_xor_si128(w2, mul_vec_inner(w3, m23_lo, m23_hi));
            w1 = _mm_xor_si128(w1, w0);
            w3 = _mm_xor_si128(w3, w2);

            store(p1, w1);
            store(p2, w2);
            store(p3, w3);
        }

        for (((x0, x1), x2), x3) in tail0.iter_mut().zip(tail1).zip(tail2).zip(tail3) {
            let (w0, mut w1, mut w2, mut w3) = (*x0, *x1, *x2, *x3);
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w2 = w2.add(w3.nibble_mul(m23));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            (*x1, *x2, *x3) = (w1, w2, w3);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_inv_dit4<G: Gf2p8>(
    node: Dit4Node<'_, G>,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let (m02_lo, m02_hi) = load_mul_table(m02);
    let (m01_lo, m01_hi) = load_mul_table(m01);
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for tup in node.split() {
        let (chunks, [tail0, tail1, tail2, tail3]) = tup.into_chunks::<16>();
        for [p0, p1, p2, p3] in chunks {
            let mut w0 = load(p0);
            let mut w1 = load(p1);
            let mut w2 = load(p2);
            let mut w3 = load(p3);

            // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
            w3 = _mm_xor_si128(w3, w2);
            w1 = _mm_xor_si128(w1, w0);
            w2 = _mm_xor_si128(w2, mul_vec_inner(w3, m23_lo, m23_hi));
            w0 = _mm_xor_si128(w0, mul_vec_inner(w1, m01_lo, m01_hi));

            // Wide level: (w0, w2) and (w1, w3), both with m02.
            w3 = _mm_xor_si128(w3, w1);
            w2 = _mm_xor_si128(w2, w0);
            w1 = _mm_xor_si128(w1, mul_vec_inner(w3, m02_lo, m02_hi));
            w0 = _mm_xor_si128(w0, mul_vec_inner(w2, m02_lo, m02_hi));

            store(p0, w0);
            store(p1, w1);
            store(p2, w2);
            store(p3, w3);
        }

        for (((x0, x1), x2), x3) in tail0.iter_mut().zip(tail1).zip(tail2).zip(tail3) {
            let (mut w0, mut w1, mut w2, mut w3) = (*x0, *x1, *x2, *x3);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.nibble_mul(m23));
            w0 = w0.add(w1.nibble_mul(m01));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            w1 = w1.add(w3.nibble_mul(m02));
            w0 = w0.add(w2.nibble_mul(m02));
            (*x0, *x1, *x2, *x3) = (w0, w1, w2, w3);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn butterfly_inv_dit4_zero<G: Gf2p8>(node: Dit4Node<'_, G>, m23: Multiplier) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for tup in node.split() {
        let (chunks, [tail0, tail1, tail2, tail3]) = tup.into_chunks::<16>();
        for [p0, p1, p2, p3] in chunks {
            let w0 = load(p0);
            let mut w1 = load(p1);
            let mut w2 = load(p2);
            let mut w3 = load(p3);

            // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
            w3 = _mm_xor_si128(w3, w2);
            w1 = _mm_xor_si128(w1, w0);
            w2 = _mm_xor_si128(w2, mul_vec_inner(w3, m23_lo, m23_hi));

            // Wide level: (w0, w2) and (w1, w3), both with m02.
            w3 = _mm_xor_si128(w3, w1);
            w2 = _mm_xor_si128(w2, w0);

            store(p1, w1);
            store(p2, w2);
            store(p3, w3);
        }

        for (((x0, x1), x2), x3) in tail0.iter_mut().zip(tail1).zip(tail2).zip(tail3) {
            let (w0, mut w1, mut w2, mut w3) = (*x0, *x1, *x2, *x3);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.nibble_mul(m23));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            (*x1, *x2, *x3) = (w1, w2, w3);
        }
    }
}

#[target_feature(enable = "ssse3")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    let psrc = src.as_ptr() as *const u8;
    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(psrc.add(i) as *const __m128i);
            let r = mul_vec_inner(v, m_lo, m_hi);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, r);
        }
        i += 16;
    }

    for j in i..len {
        dst[j] = src[j].nibble_mul(m);
    }
}

#[target_feature(enable = "ssse3")]
fn scale_add<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    let psrc = src.as_ptr() as *const u8;
    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(psrc.add(i) as *const __m128i);
            let mut w = _mm_loadu_si128(pdst.add(i) as *const __m128i);
            let r = mul_vec_inner(v, m_lo, m_hi);
            w = _mm_xor_si128(w, r);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, w);
        }
        i += 16;
    }

    for j in i..len {
        let r = src[j].nibble_mul(m);
        dst[j] = dst[j].add(r);
    }
}

#[target_feature(enable = "ssse3")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let (m_lo, m_hi) = load_mul_table(m);

    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(pdst.add(i) as *const __m128i);
            let r = mul_vec_inner(v, m_lo, m_hi);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, r);
        }
        i += 16;
    }

    for j in i..len {
        dst[j] = dst[j].nibble_mul(m);
    }
}

#[target_feature(enable = "ssse3")]
fn fft_sharded_dit4_inner_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K1_O0,
                &cantor::FFT_DIT4_NIBBLE_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K2_O0,
                &cantor::FFT_DIT4_NIBBLE_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K3_O0,
                &cantor::FFT_DIT4_NIBBLE_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K4_O0,
                &cantor::FFT_DIT4_NIBBLE_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K5_O0,
                &cantor::FFT_DIT4_NIBBLE_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K6_O0,
                &cantor::FFT_DIT4_NIBBLE_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K7_O0,
                &cantor::FFT_DIT4_NIBBLE_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_NIBBLE_K8_O0,
                &cantor::FFT_DIT4_NIBBLE_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "ssse3")]
fn fft_sharded_dit4_outer_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O0,
                cantor::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O0,
                cantor::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O0,
                cantor::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O0,
                cantor::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O0,
                cantor::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O0,
                cantor::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O0,
                cantor::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K8_O0,
                cantor::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "ssse3")]
fn ifft_sharded_dit4_outer_tail_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O0,
                cantor::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O0,
                cantor::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O0,
                cantor::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O0,
                cantor::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O0,
                cantor::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O0,
                cantor::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O0,
                cantor::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K8_O0,
                cantor::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        match (k, coords) {
            (0, _) => {}
            (1, 2) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K1_O2,
                cantor::IFFT_DIT4_NIBBLE_K1_O2_TAIL.as_ref(),
            ),
            (2, 4) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K2_O4,
                cantor::IFFT_DIT4_NIBBLE_K2_O4_TAIL.as_ref(),
            ),
            (3, 8) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K3_O8,
                cantor::IFFT_DIT4_NIBBLE_K3_O8_TAIL.as_ref(),
            ),
            (4, 16) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K4_O16,
                cantor::IFFT_DIT4_NIBBLE_K4_O16_TAIL.as_ref(),
            ),
            (5, 32) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K5_O32,
                cantor::IFFT_DIT4_NIBBLE_K5_O32_TAIL.as_ref(),
            ),
            (6, 64) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K6_O64,
                cantor::IFFT_DIT4_NIBBLE_K6_O64_TAIL.as_ref(),
            ),
            (7, 128) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_NIBBLE_K7_O128,
                cantor::IFFT_DIT4_NIBBLE_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, coords);
            }
        }
    }
}

#[target_feature(enable = "ssse3")]
fn fft_sharded_dit4_inner_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K1_O0,
                &standard::FFT_DIT4_NIBBLE_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K2_O0,
                &standard::FFT_DIT4_NIBBLE_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K3_O0,
                &standard::FFT_DIT4_NIBBLE_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K4_O0,
                &standard::FFT_DIT4_NIBBLE_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K5_O0,
                &standard::FFT_DIT4_NIBBLE_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K6_O0,
                &standard::FFT_DIT4_NIBBLE_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K7_O0,
                &standard::FFT_DIT4_NIBBLE_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_inner_tail_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_NIBBLE_K8_O0,
                &standard::FFT_DIT4_NIBBLE_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "ssse3")]
fn fft_sharded_dit4_outer_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O0,
                standard::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O0,
                standard::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O0,
                standard::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O0,
                standard::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O0,
                standard::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O0,
                standard::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O0,
                standard::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K8_O0,
                standard::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        K::fft_sharded_dit4_indexed(shards, shard_len, k, coords);
    }
}

#[target_feature(enable = "ssse3")]
fn ifft_sharded_dit4_outer_tail_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    coords: usize,
) {
    type K = Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if coords == 0 {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O0,
                standard::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O0,
                standard::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O0,
                standard::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O0,
                standard::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O0,
                standard::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O0,
                standard::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O0,
                standard::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K8_O0,
                standard::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        match (k, coords) {
            (0, _) => {}
            (1, 2) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K1_O2,
                standard::IFFT_DIT4_NIBBLE_K1_O2_TAIL.as_ref(),
            ),
            (2, 4) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K2_O4,
                standard::IFFT_DIT4_NIBBLE_K2_O4_TAIL.as_ref(),
            ),
            (3, 8) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K3_O8,
                standard::IFFT_DIT4_NIBBLE_K3_O8_TAIL.as_ref(),
            ),
            (4, 16) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K4_O16,
                standard::IFFT_DIT4_NIBBLE_K4_O16_TAIL.as_ref(),
            ),
            (5, 32) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K5_O32,
                standard::IFFT_DIT4_NIBBLE_K5_O32_TAIL.as_ref(),
            ),
            (6, 64) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K6_O64,
                standard::IFFT_DIT4_NIBBLE_K6_O64_TAIL.as_ref(),
            ),
            (7, 128) => K::ifft_sharded_dit4_outer_tail_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_NIBBLE_K7_O128,
                standard::IFFT_DIT4_NIBBLE_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, coords);
            }
        }
    }
}
#[derive(Default)]
pub struct Ssse3Kernel<G: Gf2p8Lut, B: SubspaceBasisLut<G>>(PhantomData<(G, B)>);

impl Kernel<Gf2p8_11d, CantorBasisLut11d> for Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d> {
    const ALIGN: usize = 16;

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

impl Kernel<Gf2p8_11d, StandardBasisLut11d> for Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d> {
    const ALIGN: usize = 16;

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
#[cfg(native_ssse3)]
mod test {
    use super::*;
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_cfg() {
        let target = cfg!(target_arch = "x86_64");
        let feature = is_x86_feature_detected!("ssse3");

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
                    let v = _mm_set1_epi8(x as i8); // copy x cross all 16 lanes
                    let (m_lo, m_hi) = load_mul_table(m);
                    let r = mul_vec_inner(v, m_lo, m_hi);
                    let byte0 = _mm_extract_epi8::<0>(r) as u8;
                    Gf2p8_11d(byte0)
                };
                assert_eq!(expected, actual, "p={p:02x} x={x:02x}");
            }
        }
    }
}

#[cfg(test)]
#[cfg(native_ssse3)]
mod test_cantor {
    use crate::poly_11d_lut::CantorBasisLut11d;

    type B = CantorBasisLut11d;
    include!("ssse3_kernel_tests.rs");
}

#[cfg(test)]
#[cfg(native_ssse3)]
mod test_standard {
    use crate::poly_11d_lut::StandardBasisLut11d;

    type B = StandardBasisLut11d;
    include!("ssse3_kernel_tests.rs");
}

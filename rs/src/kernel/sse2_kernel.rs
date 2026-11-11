use super::{Kernel, shard_groups};
use crate::gf2p8lut::{Gf2p8Lut, SubspaceBasisLut};
use crate::poly_11d_lut::generated::{EXP_TABLE, cantor, standard};
use crate::poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, Z255};
use core::arch::x86_64::*;
use std::marker::PhantomData;

type Multiplier = u8;

/// x · 2 in GF(2^8) mod G::POLY, bytewise.
#[inline]
#[target_feature(enable = "sse2")]
fn xtime<G: Gf2p8>(x: __m128i) -> __m128i {
    let carry = _mm_cmplt_epi8(x, _mm_setzero_si128());
    let shifted = _mm_add_epi8(x, x); // bytewise x << 1
    _mm_xor_si128(shifted, _mm_and_si128(carry, _mm_set1_epi8(G::POLY as i8)))
}

#[inline]
#[target_feature(enable = "sse2")]
fn mul_consts<G: Gf2p8>(c: Multiplier) -> [__m128i; 8] {
    let mut v = _mm_set1_epi8(c as i8);
    std::array::from_fn(|_| {
        let out = v;
        v = xtime::<G>(v);
        out
    })
}

/// x·c given the node's mul_consts(c): XOR of c·2^i over the set bits i of x.
#[inline]
#[target_feature(enable = "sse2")]
fn mul_vec_inner(mut x: __m128i, tv: &[__m128i; 8]) -> __m128i {
    let zero = _mm_setzero_si128();
    let mut acc = zero;
    for i in (0..8).rev() {
        let bit = _mm_cmplt_epi8(x, zero); // 0xff where bit i is set
        acc = _mm_xor_si128(acc, _mm_and_si128(bit, tv[i]));
        x = _mm_add_epi8(x, x); // next bit into the sign
    }
    acc
}

#[target_feature(enable = "sse2")]
fn butterfly_fwd_dit2<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    start: usize,
    d: usize,
    m: Multiplier,
) {
    let mv = mul_consts::<G>(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, start, d) {
        let (pa, pb) = (a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8);
        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let va = _mm_loadu_si128(pa.add(i) as *const __m128i);
                let vb = _mm_loadu_si128(pb.add(i) as *const __m128i);
                let va = _mm_xor_si128(va, mul_vec_inner(vb, &mv)); // g0 = a + T·b
                let vb = _mm_xor_si128(vb, va); // g1 = b + g0
                _mm_storeu_si128(pa.add(i) as *mut __m128i, va);
                _mm_storeu_si128(pb.add(i) as *mut __m128i, vb);
            }
            i += 16;
        }
        for (x, y) in a[i..].iter_mut().zip(&mut b[i..]) {
            let g0 = x.add(y.mul_lut(G::from(m)));
            *x = g0;
            *y = y.add(g0);
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_fwd_dit2_zero<G: Gf2p8>(shards: &mut [G], shard_len: usize, start: usize, d: usize) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, start, d) {
        let mut i = 0;
        let (pa, pb) = (a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8);
        while i + 16 <= shard_len {
            unsafe {
                let va = _mm_loadu_si128(pa.add(i) as *const __m128i);
                let vb = _mm_loadu_si128(pb.add(i) as *const __m128i);
                let vb = _mm_xor_si128(vb, va); // g1 = b + g0
                _mm_storeu_si128(pb.add(i) as *mut __m128i, vb);
            }
            i += 16;
        }
        for (x, y) in a[i..].iter().zip(&mut b[i..]) {
            *y = y.add(*x);
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_inv_dit2<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: Multiplier,
) {
    let mv = mul_consts::<G>(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        let pa = a.as_mut_ptr() as *mut u8;
        let pb = b.as_mut_ptr() as *mut u8;
        while i + 16 <= shard_len {
            unsafe {
                let va = _mm_loadu_si128(pa.add(i) as *const __m128i);
                let vb = _mm_loadu_si128(pb.add(i) as *const __m128i);
                let vb = _mm_xor_si128(vb, va); // d' = g0 + g1
                let va = _mm_xor_si128(va, mul_vec_inner(vb, &mv)); // d = g0 + T·d'
                _mm_storeu_si128(pa.add(i) as *mut __m128i, va);
                _mm_storeu_si128(pb.add(i) as *mut __m128i, vb);
            }
            i += 16;
        }
        for (x, y) in a[i..].iter_mut().zip(&mut b[i..]) {
            *y = x.add(*y);
            *x = x.add(y.mul_lut(G::from(m)));
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_fwd_dit4<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let m02v = mul_consts::<G>(m02);
    let m01v = mul_consts::<G>(m01);
    let m23v = mul_consts::<G>(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p = [&mut *s0, &mut *s1, &mut *s2, &mut *s3].map(|s| s.as_mut_ptr() as *mut u8);

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let mut w0 = _mm_loadu_si128(p[0].add(i) as *const __m128i);
                let mut w1 = _mm_loadu_si128(p[1].add(i) as *const __m128i);
                let mut w2 = _mm_loadu_si128(p[2].add(i) as *const __m128i);
                let mut w3 = _mm_loadu_si128(p[3].add(i) as *const __m128i);

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w0 = _mm_xor_si128(w0, mul_vec_inner(w2, &m02v));
                w1 = _mm_xor_si128(w1, mul_vec_inner(w3, &m02v));
                w2 = _mm_xor_si128(w2, w0);
                w3 = _mm_xor_si128(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w0 = _mm_xor_si128(w0, mul_vec_inner(w1, &m01v));
                w2 = _mm_xor_si128(w2, mul_vec_inner(w3, &m23v));
                w1 = _mm_xor_si128(w1, w0);
                w3 = _mm_xor_si128(w3, w2);

                _mm_storeu_si128(p[0].add(i) as *mut __m128i, w0);
                _mm_storeu_si128(p[1].add(i) as *mut __m128i, w1);
                _mm_storeu_si128(p[2].add(i) as *mut __m128i, w2);
                _mm_storeu_si128(p[3].add(i) as *mut __m128i, w3);
            }
            i += 16;
        }

        for j in i..shard_len {
            let (mut w0, mut w1, mut w2, mut w3) = (s0[j], s1[j], s2[j], s3[j]);
            w0 = w0.add(w2.mul_lut(G::from(m02)));
            w1 = w1.add(w3.mul_lut(G::from(m02)));
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w0 = w0.add(w1.mul_lut(G::from(m01)));
            w2 = w2.add(w3.mul_lut(G::from(m23)));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            (s0[j], s1[j], s2[j], s3[j]) = (w0, w1, w2, w3);
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_fwd_dit4_zero<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: Multiplier,
) {
    let m23v = mul_consts::<G>(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p = [&mut *s0, &mut *s1, &mut *s2, &mut *s3].map(|s| s.as_mut_ptr() as *mut u8);

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let w0 = _mm_loadu_si128(p[0].add(i) as *const __m128i);
                let mut w1 = _mm_loadu_si128(p[1].add(i) as *const __m128i);
                let mut w2 = _mm_loadu_si128(p[2].add(i) as *const __m128i);
                let mut w3 = _mm_loadu_si128(p[3].add(i) as *const __m128i);

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w2 = _mm_xor_si128(w2, w0);
                w3 = _mm_xor_si128(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w2 = _mm_xor_si128(w2, mul_vec_inner(w3, &m23v));
                w1 = _mm_xor_si128(w1, w0);
                w3 = _mm_xor_si128(w3, w2);

                _mm_storeu_si128(p[1].add(i) as *mut __m128i, w1);
                _mm_storeu_si128(p[2].add(i) as *mut __m128i, w2);
                _mm_storeu_si128(p[3].add(i) as *mut __m128i, w3);
            }
            i += 16;
        }

        for j in i..shard_len {
            let (w0, mut w1, mut w2, mut w3) = (s0[j], s1[j], s2[j], s3[j]);
            w2 = w2.add(w0);
            w3 = w3.add(w1);
            w2 = w2.add(w3.mul_lut(G::from(m23)));
            w1 = w1.add(w0);
            w3 = w3.add(w2);
            (s1[j], s2[j], s3[j]) = (w1, w2, w3);
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_inv_dit4<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: Multiplier,
    m23: Multiplier,
    m02: Multiplier,
) {
    let m02v = mul_consts::<G>(m02);
    let m01v = mul_consts::<G>(m01);
    let m23v = mul_consts::<G>(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p = [&mut *s0, &mut *s1, &mut *s2, &mut *s3].map(|s| s.as_mut_ptr() as *mut u8);

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let mut w0 = _mm_loadu_si128(p[0].add(i) as *const __m128i);
                let mut w1 = _mm_loadu_si128(p[1].add(i) as *const __m128i);
                let mut w2 = _mm_loadu_si128(p[2].add(i) as *const __m128i);
                let mut w3 = _mm_loadu_si128(p[3].add(i) as *const __m128i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm_xor_si128(w3, w2);
                w1 = _mm_xor_si128(w1, w0);
                w2 = _mm_xor_si128(w2, mul_vec_inner(w3, &m23v));
                w0 = _mm_xor_si128(w0, mul_vec_inner(w1, &m01v));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = _mm_xor_si128(w3, w1);
                w2 = _mm_xor_si128(w2, w0);
                w1 = _mm_xor_si128(w1, mul_vec_inner(w3, &m02v));
                w0 = _mm_xor_si128(w0, mul_vec_inner(w2, &m02v));

                _mm_storeu_si128(p[0].add(i) as *mut __m128i, w0);
                _mm_storeu_si128(p[1].add(i) as *mut __m128i, w1);
                _mm_storeu_si128(p[2].add(i) as *mut __m128i, w2);
                _mm_storeu_si128(p[3].add(i) as *mut __m128i, w3);
            }
            i += 16;
        }

        for j in i..shard_len {
            let (mut w0, mut w1, mut w2, mut w3) = (s0[j], s1[j], s2[j], s3[j]);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.mul_lut(G::from(m23)));
            w0 = w0.add(w1.mul_lut(G::from(m01)));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            w1 = w1.add(w3.mul_lut(G::from(m02)));
            w0 = w0.add(w2.mul_lut(G::from(m02)));
            (s0[j], s1[j], s2[j], s3[j]) = (w0, w1, w2, w3);
        }
    }
}

#[target_feature(enable = "sse2")]
fn butterfly_inv_dit4_zero<G: Gf2p8Lut>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: Multiplier,
) {
    let m23v = mul_consts::<G>(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p = [&mut *s0, &mut *s1, &mut *s2, &mut *s3].map(|s| s.as_mut_ptr() as *mut u8);

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let w0 = _mm_loadu_si128(p[0].add(i) as *const __m128i);
                let mut w1 = _mm_loadu_si128(p[1].add(i) as *const __m128i);
                let mut w2 = _mm_loadu_si128(p[2].add(i) as *const __m128i);
                let mut w3 = _mm_loadu_si128(p[3].add(i) as *const __m128i);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = _mm_xor_si128(w3, w2);
                w1 = _mm_xor_si128(w1, w0);
                w2 = _mm_xor_si128(w2, mul_vec_inner(w3, &m23v));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = _mm_xor_si128(w3, w1);
                w2 = _mm_xor_si128(w2, w0);

                _mm_storeu_si128(p[1].add(i) as *mut __m128i, w1);
                _mm_storeu_si128(p[2].add(i) as *mut __m128i, w2);
                _mm_storeu_si128(p[3].add(i) as *mut __m128i, w3);
            }
            i += 16;
        }

        for j in i..shard_len {
            let (w0, mut w1, mut w2, mut w3) = (s0[j], s1[j], s2[j], s3[j]);
            w3 = w3.add(w2);
            w1 = w1.add(w0);
            w2 = w2.add(w3.mul_lut(G::from(m23)));
            w3 = w3.add(w1);
            w2 = w2.add(w0);
            (s1[j], s2[j], s3[j]) = (w1, w2, w3);
        }
    }
}

#[target_feature(enable = "sse2")]
fn scale<G: Gf2p8Lut>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let mv = mul_consts::<G>(m);

    let psrc = src.as_ptr() as *const u8;
    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(psrc.add(i) as *const __m128i);
            let r = mul_vec_inner(v, &mv);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, r);
        }
        i += 16;
    }

    for j in i..len {
        dst[j] = src[j].mul_lut(G::from(m));
    }
}

#[target_feature(enable = "sse2")]
fn scale_add<G: Gf2p8Lut>(src: &[G], dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let mv = mul_consts::<G>(m);

    let psrc = src.as_ptr() as *const u8;
    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(psrc.add(i) as *const __m128i);
            let mut w = _mm_loadu_si128(pdst.add(i) as *const __m128i);
            let r = mul_vec_inner(v, &mv);
            w = _mm_xor_si128(w, r);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, w);
        }
        i += 16;
    }

    for j in i..len {
        let r = src[j].mul_lut(G::from(m));
        dst[j] = dst[j].add(r);
    }
}

#[target_feature(enable = "sse2")]
fn scale_in_place<G: Gf2p8Lut>(dst: &mut [G], len: usize, m: Multiplier) {
    let mut i = 0;
    let mv = mul_consts::<G>(m);

    let pdst = dst.as_mut_ptr() as *mut u8;
    while i + 16 <= len {
        unsafe {
            let v = _mm_loadu_si128(pdst.add(i) as *const __m128i);
            let r = mul_vec_inner(v, &mv);
            _mm_storeu_si128(pdst.add(i) as *mut __m128i, r);
        }
        i += 16;
    }

    for j in i..len {
        dst[j] = dst[j].mul_lut(G::from(m));
    }
}

#[target_feature(enable = "sse2")]
fn fft_sharded_cantor(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Sse2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K1_O0,
                &cantor::FFT_DIT4_PEASANT_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K2_O0,
                &cantor::FFT_DIT4_PEASANT_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K3_O0,
                &cantor::FFT_DIT4_PEASANT_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K4_O0,
                &cantor::FFT_DIT4_PEASANT_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K5_O0,
                &cantor::FFT_DIT4_PEASANT_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K6_O0,
                &cantor::FFT_DIT4_PEASANT_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K7_O0,
                &cantor::FFT_DIT4_PEASANT_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::FFT_DIT4_PEASANT_K8_O0,
                &cantor::FFT_DIT4_PEASANT_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "sse2")]
fn fft_sharded_radix2_first_cantor(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    beta: Gf2p8_11d,
) {
    type K = Sse2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K1_O0,
                cantor::IFFT_DIT4_PEASANT_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K2_O0,
                cantor::IFFT_DIT4_PEASANT_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K3_O0,
                cantor::IFFT_DIT4_PEASANT_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K4_O0,
                cantor::IFFT_DIT4_PEASANT_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K5_O0,
                cantor::IFFT_DIT4_PEASANT_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K6_O0,
                cantor::IFFT_DIT4_PEASANT_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K7_O0,
                cantor::IFFT_DIT4_PEASANT_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K8_O0,
                cantor::IFFT_DIT4_PEASANT_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "sse2")]
fn ifft_sharded_cantor(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Sse2Kernel<Gf2p8_11d, CantorBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K1_O0,
                cantor::IFFT_DIT4_PEASANT_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K2_O0,
                cantor::IFFT_DIT4_PEASANT_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K3_O0,
                cantor::IFFT_DIT4_PEASANT_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K4_O0,
                cantor::IFFT_DIT4_PEASANT_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K5_O0,
                cantor::IFFT_DIT4_PEASANT_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K6_O0,
                cantor::IFFT_DIT4_PEASANT_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K7_O0,
                cantor::IFFT_DIT4_PEASANT_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K8_O0,
                cantor::IFFT_DIT4_PEASANT_K8_O0_TAIL.as_ref(),
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
                &cantor::IFFT_DIT4_PEASANT_K1_O2,
                cantor::IFFT_DIT4_PEASANT_K1_O2_TAIL.as_ref(),
            ),
            (2, b) if b == cantor::SUBSPACE[4] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K2_O4,
                cantor::IFFT_DIT4_PEASANT_K2_O4_TAIL.as_ref(),
            ),
            (3, b) if b == cantor::SUBSPACE[8] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K3_O8,
                cantor::IFFT_DIT4_PEASANT_K3_O8_TAIL.as_ref(),
            ),
            (4, b) if b == cantor::SUBSPACE[16] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K4_O16,
                cantor::IFFT_DIT4_PEASANT_K4_O16_TAIL.as_ref(),
            ),
            (5, b) if b == cantor::SUBSPACE[32] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K5_O32,
                cantor::IFFT_DIT4_PEASANT_K5_O32_TAIL.as_ref(),
            ),
            (6, b) if b == cantor::SUBSPACE[64] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K6_O64,
                cantor::IFFT_DIT4_PEASANT_K6_O64_TAIL.as_ref(),
            ),
            (7, b) if b == cantor::SUBSPACE[128] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &cantor::IFFT_DIT4_PEASANT_K7_O128,
                cantor::IFFT_DIT4_PEASANT_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                let t = cantor::SUBSPACE_INDEX[beta.into_usize()] as usize;
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, t);
            }
        }
    }
}

#[target_feature(enable = "sse2")]
fn fft_sharded_standard(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Sse2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K1_O0,
                &standard::FFT_DIT4_PEASANT_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K2_O0,
                &standard::FFT_DIT4_PEASANT_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K3_O0,
                &standard::FFT_DIT4_PEASANT_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K4_O0,
                &standard::FFT_DIT4_PEASANT_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K5_O0,
                &standard::FFT_DIT4_PEASANT_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K6_O0,
                &standard::FFT_DIT4_PEASANT_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K7_O0,
                &standard::FFT_DIT4_PEASANT_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::FFT_DIT4_PEASANT_K8_O0,
                &standard::FFT_DIT4_PEASANT_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "sse2")]
fn fft_sharded_radix2_first_standard(
    shards: &mut [Gf2p8_11d],
    shard_len: usize,
    k: u8,
    beta: Gf2p8_11d,
) {
    type K = Sse2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K1_O0,
                standard::IFFT_DIT4_PEASANT_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K2_O0,
                standard::IFFT_DIT4_PEASANT_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K3_O0,
                standard::IFFT_DIT4_PEASANT_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K4_O0,
                standard::IFFT_DIT4_PEASANT_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K5_O0,
                standard::IFFT_DIT4_PEASANT_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K6_O0,
                standard::IFFT_DIT4_PEASANT_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K7_O0,
                standard::IFFT_DIT4_PEASANT_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_radix2_first_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K8_O0,
                standard::IFFT_DIT4_PEASANT_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

#[target_feature(enable = "sse2")]
fn ifft_sharded_standard(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = Sse2Kernel<Gf2p8_11d, StandardBasisLut11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K1_O0,
                standard::IFFT_DIT4_PEASANT_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K2_O0,
                standard::IFFT_DIT4_PEASANT_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K3_O0,
                standard::IFFT_DIT4_PEASANT_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K4_O0,
                standard::IFFT_DIT4_PEASANT_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K5_O0,
                standard::IFFT_DIT4_PEASANT_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K6_O0,
                standard::IFFT_DIT4_PEASANT_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K7_O0,
                standard::IFFT_DIT4_PEASANT_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K8_O0,
                standard::IFFT_DIT4_PEASANT_K8_O0_TAIL.as_ref(),
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
                &standard::IFFT_DIT4_PEASANT_K1_O2,
                standard::IFFT_DIT4_PEASANT_K1_O2_TAIL.as_ref(),
            ),
            (2, b) if b == standard::SUBSPACE[4] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K2_O4,
                standard::IFFT_DIT4_PEASANT_K2_O4_TAIL.as_ref(),
            ),
            (3, b) if b == standard::SUBSPACE[8] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K3_O8,
                standard::IFFT_DIT4_PEASANT_K3_O8_TAIL.as_ref(),
            ),
            (4, b) if b == standard::SUBSPACE[16] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K4_O16,
                standard::IFFT_DIT4_PEASANT_K4_O16_TAIL.as_ref(),
            ),
            (5, b) if b == standard::SUBSPACE[32] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K5_O32,
                standard::IFFT_DIT4_PEASANT_K5_O32_TAIL.as_ref(),
            ),
            (6, b) if b == standard::SUBSPACE[64] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K6_O64,
                standard::IFFT_DIT4_PEASANT_K6_O64_TAIL.as_ref(),
            ),
            (7, b) if b == standard::SUBSPACE[128] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &standard::IFFT_DIT4_PEASANT_K7_O128,
                standard::IFFT_DIT4_PEASANT_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                let t = standard::SUBSPACE_INDEX[beta.into_usize()] as usize;
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, t);
            }
        }
    }
}
#[derive(Default)]
pub struct Sse2Kernel<G: Gf2p8Lut, B: SubspaceBasisLut<G>>(PhantomData<(G, B)>);

impl Kernel<Gf2p8_11d, CantorBasisLut11d> for Sse2Kernel<Gf2p8_11d, CantorBasisLut11d> {
    const ALIGN: usize = 16;

    type Multiplier = u8;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &cantor::NODE_MUL_PEASANT;

    fn mul_table(t: Gf2p8_11d) -> Self::Multiplier {
        t.into()
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
        unsafe {
            scale(src, dst, dst.len(), scalar.into());
        }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        unsafe {
            scale_add(src, dst, dst.len(), scalar.into());
        }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        unsafe {
            scale_in_place(dst, dst.len(), scalar.into());
        }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = EXP_TABLE[log_m.0 as usize];
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = EXP_TABLE[log_m.0 as usize];
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

impl Kernel<Gf2p8_11d, StandardBasisLut11d> for Sse2Kernel<Gf2p8_11d, StandardBasisLut11d> {
    const ALIGN: usize = 16;

    type Multiplier = u8;

    const NODE_MUL: &'static [Self::Multiplier; 510] = &standard::NODE_MUL_PEASANT;

    fn mul_table(t: Gf2p8_11d) -> Self::Multiplier {
        t.into()
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
        unsafe {
            scale(src, dst, dst.len(), scalar.into());
        }
    }

    fn scale_add(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        unsafe {
            scale_add(src, dst, dst.len(), scalar.into());
        }
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        unsafe {
            scale_in_place(dst, dst.len(), scalar.into());
        }
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = EXP_TABLE[log_m.0 as usize];
        unsafe { scale(src, dst, dst.len(), m) };
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = EXP_TABLE[log_m.0 as usize];
        unsafe { scale_in_place(dst, dst.len(), m) };
    }
}

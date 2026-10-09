use super::{Kernel, shard_groups};
use crate::{
    gf2p8lut::Gf2p8Lut,
    poly_11d_lut::generated::{
        self as tables, CANTOR_SUBSPACE, NIBBLE_MUL_BY_LOG, NIBBLE_MUL_TABLE,
    },
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d, NibbleMulTable, Z255};
use core::arch::aarch64::*;
use std::marker::PhantomData;

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_neon_kernel_11d.rs"));
}

type MulTable = &'static NibbleMulTable;

#[inline]
fn mul_vec(v: uint8x16_t, m: &NibbleMulTable) -> uint8x16_t {
    unsafe {
        let m_lo = vld1q_u8(m.0.as_ptr());
        let m_hi = vld1q_u8(m.1.as_ptr());
        mul_vec_inner(v, m_lo, m_hi)
    }
}

#[inline]
fn mul_vec_inner(v: uint8x16_t, m_lo: uint8x16_t, m_hi: uint8x16_t) -> uint8x16_t {
    unsafe {
        let mask = vdupq_n_u8(0x0f);
        let lo = vqtbl1q_u8(m_lo, vandq_u8(v, mask));
        let hi = vqtbl1q_u8(m_hi, vshrq_n_u8(v, 4));
        veorq_u8(lo, hi)
    }
}

#[inline]
fn load_mul_table(m: MulTable) -> (uint8x16_t, uint8x16_t) {
    unsafe { (vld1q_u8(m.0.as_ptr()), vld1q_u8(m.1.as_ptr())) }
}

#[inline]
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;
        while i + 16 <= len {
            unsafe {
                let va = vld1q_u8(a.add(i));
                let vb = vld1q_u8(b.add(i));
                let va = veorq_u8(va, mul_vec(vb, m)); // g0 = a + T·b
                let vb = veorq_u8(vb, va); // g1 = b + g0
                vst1q_u8(a.add(i), va);
                vst1q_u8(b.add(i), vb);
            }
            i += 16;
        }
    }
    // Handle the tail scalar. There are no masked load/store ops in NEON, hence apply the
    // multiplication table elementwise.
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
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;
        while i + 16 <= len {
            unsafe {
                let va = vld1q_u8(a.add(i));
                let vb = vld1q_u8(b.add(i));
                let vb = veorq_u8(vb, va);
                let va = veorq_u8(va, mul_vec(vb, m));
                vst1q_u8(a.add(i), va);
                vst1q_u8(b.add(i), vb);
            }
            i += 16;
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

fn butterfly_fwd_dit2<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: MulTable,
) {
    let (m_lo, m_hi) = load_mul_table(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 16 <= shard_len {
                unsafe {
                    let va = vld1q_u8(a.add(i));
                    let vb = vld1q_u8(b.add(i));
                    let t = mul_vec_inner(vb, m_lo, m_hi); // T·b
                    let va = veorq_u8(va, t); // g0 = a + T·b
                    let vb = veorq_u8(vb, va); // g1 = b + g0
                    vst1q_u8(a.add(i), va);
                    vst1q_u8(b.add(i), vb);
                }
                i += 16;
            }
        }
        // Handle the tail scalar. There are no masked load/store ops in NEON, hence apply the
        // multiplication table elementwise.
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

fn butterfly_fwd_dit2_zero<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 16 <= shard_len {
                unsafe {
                    let va = vld1q_u8(a.add(i));
                    let vb = vld1q_u8(b.add(i));
                    let vb = veorq_u8(vb, va); // g1 = b + g0
                    vst1q_u8(b.add(i), vb);
                }
                i += 16;
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

fn butterfly_inv_dit2<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: MulTable,
) {
    let (m_lo, m_hi) = load_mul_table(m);

    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        let mut i = 0;
        {
            let a = a.as_mut_ptr() as *mut u8;
            let b = b.as_mut_ptr() as *mut u8;
            while i + 16 <= shard_len {
                unsafe {
                    let va = vld1q_u8(a.add(i));
                    let vb = vld1q_u8(b.add(i));
                    let vb = veorq_u8(vb, va);
                    let t = mul_vec_inner(vb, m_lo, m_hi);
                    let va = veorq_u8(va, t);
                    vst1q_u8(a.add(i), va);
                    vst1q_u8(b.add(i), vb);
                }
                i += 16;
            }
        }
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

fn butterfly_fwd_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: MulTable,
    m23: MulTable,
    m02: MulTable,
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
        while i + 16 <= shard_len {
            unsafe {
                let mut w0 = vld1q_u8(p0.add(i));
                let mut w1 = vld1q_u8(p1.add(i));
                let mut w2 = vld1q_u8(p2.add(i));
                let mut w3 = vld1q_u8(p3.add(i));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w0 = veorq_u8(w0, mul_vec_inner(w2, m02_lo, m02_hi));
                w1 = veorq_u8(w1, mul_vec_inner(w3, m02_lo, m02_hi));
                w2 = veorq_u8(w2, w0);
                w3 = veorq_u8(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w0 = veorq_u8(w0, mul_vec_inner(w1, m01_lo, m01_hi));
                w2 = veorq_u8(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w1 = veorq_u8(w1, w0);
                w3 = veorq_u8(w3, w2);

                vst1q_u8(p0.add(i), w0);
                vst1q_u8(p1.add(i), w1);
                vst1q_u8(p2.add(i), w2);
                vst1q_u8(p3.add(i), w3);
            }
            i += 16;
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

fn butterfly_fwd_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: MulTable,
) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let w0 = vld1q_u8(p0.add(i));
                let mut w1 = vld1q_u8(p1.add(i));
                let mut w2 = vld1q_u8(p2.add(i));
                let mut w3 = vld1q_u8(p3.add(i));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w2 = veorq_u8(w2, w0);
                w3 = veorq_u8(w3, w1);

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w2 = veorq_u8(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w1 = veorq_u8(w1, w0);
                w3 = veorq_u8(w3, w2);

                vst1q_u8(p1.add(i), w1);
                vst1q_u8(p2.add(i), w2);
                vst1q_u8(p3.add(i), w3);
            }
            i += 16;
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

fn butterfly_inv_dit4<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m01: MulTable,
    m23: MulTable,
    m02: MulTable,
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
        while i + 16 <= shard_len {
            unsafe {
                let mut w0 = vld1q_u8(p0.add(i));
                let mut w1 = vld1q_u8(p1.add(i));
                let mut w2 = vld1q_u8(p2.add(i));
                let mut w3 = vld1q_u8(p3.add(i));

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = veorq_u8(w3, w2);
                w1 = veorq_u8(w1, w0);
                w2 = veorq_u8(w2, mul_vec_inner(w3, m23_lo, m23_hi));
                w0 = veorq_u8(w0, mul_vec_inner(w1, m01_lo, m01_hi));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = veorq_u8(w3, w1);
                w2 = veorq_u8(w2, w0);
                w1 = veorq_u8(w1, mul_vec_inner(w3, m02_lo, m02_hi));
                w0 = veorq_u8(w0, mul_vec_inner(w2, m02_lo, m02_hi));

                vst1q_u8(p0.add(i), w0);
                vst1q_u8(p1.add(i), w1);
                vst1q_u8(p2.add(i), w2);
                vst1q_u8(p3.add(i), w3);
            }
            i += 16;
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

fn butterfly_inv_dit4_zero<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m23: MulTable,
) {
    let (m23_lo, m23_hi) = load_mul_table(m23);

    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        let p0 = s0.as_mut_ptr() as *mut u8;
        let p1 = s1.as_mut_ptr() as *mut u8;
        let p2 = s2.as_mut_ptr() as *mut u8;
        let p3 = s3.as_mut_ptr() as *mut u8;

        let mut i = 0;
        while i + 16 <= shard_len {
            unsafe {
                let w0 = vld1q_u8(p0.add(i));
                let mut w1 = vld1q_u8(p1.add(i));
                let mut w2 = vld1q_u8(p2.add(i));
                let mut w3 = vld1q_u8(p3.add(i));

                // Narrow level: (w0, w1) with m01, (w2, w3) with m23.
                w3 = veorq_u8(w3, w2);
                w1 = veorq_u8(w1, w0);
                w2 = veorq_u8(w2, mul_vec_inner(w3, m23_lo, m23_hi));

                // Wide level: (w0, w2) and (w1, w3), both with m02.
                w3 = veorq_u8(w3, w1);
                w2 = veorq_u8(w2, w0);

                vst1q_u8(p1.add(i), w1);
                vst1q_u8(p2.add(i), w2);
                vst1q_u8(p3.add(i), w3);
            }
            i += 16;
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

fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let src = src.as_ptr() as *const u8;
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 16 <= len {
            unsafe {
                let v = vld1q_u8(src.add(i));
                let r = mul_vec(v, m);
                vst1q_u8(dst.add(i), r);
            }
            i += 16;
        }
    }
    while i < len {
        let x = src[i];
        let r = x.nibble_mul(m);
        dst[i] = r;
        i += 1;
    }
}

fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, m: &NibbleMulTable) {
    let mut i = 0;
    {
        let dst = dst.as_mut_ptr() as *mut u8;
        while i + 16 <= len {
            unsafe {
                let v = vld1q_u8(dst.add(i));
                let r = mul_vec(v, m);
                vst1q_u8(dst.add(i), r);
            }
            i += 16;
        }
    }
    while i < len {
        let x = dst[i];
        let r = x.nibble_mul(m);
        dst[i] = r;
        i += 1;
    }
}

fn fft_sharded_radix2_last(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = NeonKernel<Gf2p8_11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K1_O0,
                &tables::FFT_DIT4_NIBBLE_K1_O0_TAIL,
            ),
            2 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K2_O0,
                &tables::FFT_DIT4_NIBBLE_K2_O0_TAIL,
            ),
            3 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K3_O0,
                &tables::FFT_DIT4_NIBBLE_K3_O0_TAIL,
            ),
            4 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K4_O0,
                &tables::FFT_DIT4_NIBBLE_K4_O0_TAIL,
            ),
            5 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K5_O0,
                &tables::FFT_DIT4_NIBBLE_K5_O0_TAIL,
            ),
            6 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K6_O0,
                &tables::FFT_DIT4_NIBBLE_K6_O0_TAIL,
            ),
            7 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K7_O0,
                &tables::FFT_DIT4_NIBBLE_K7_O0_TAIL,
            ),
            8 => K::fft_sharded_dit4_radix2_last_with(
                shards,
                shard_len,
                k,
                &tables::FFT_DIT4_NIBBLE_K8_O0,
                &tables::FFT_DIT4_NIBBLE_K8_O0_TAIL,
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = tables::CANTOR_SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

fn fft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = NeonKernel<Gf2p8_11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K1_O0,
                tables::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K2_O0,
                tables::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K3_O0,
                tables::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K4_O0,
                tables::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K5_O0,
                tables::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K6_O0,
                tables::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K7_O0,
                tables::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::fft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K8_O0,
                tables::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        let t = tables::CANTOR_SUBSPACE_INDEX[beta.into_usize()] as usize;
        K::fft_sharded_dit4_indexed(shards, shard_len, k, t);
    }
}

fn ifft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
    type K = NeonKernel<Gf2p8_11d>;

    if beta == Gf2p8_11d::zero() {
        match k {
            0 => {}
            1 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K1_O0,
                tables::IFFT_DIT4_NIBBLE_K1_O0_TAIL.as_ref(),
            ),
            2 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K2_O0,
                tables::IFFT_DIT4_NIBBLE_K2_O0_TAIL.as_ref(),
            ),
            3 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K3_O0,
                tables::IFFT_DIT4_NIBBLE_K3_O0_TAIL.as_ref(),
            ),
            4 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K4_O0,
                tables::IFFT_DIT4_NIBBLE_K4_O0_TAIL.as_ref(),
            ),
            5 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K5_O0,
                tables::IFFT_DIT4_NIBBLE_K5_O0_TAIL.as_ref(),
            ),
            6 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K6_O0,
                tables::IFFT_DIT4_NIBBLE_K6_O0_TAIL.as_ref(),
            ),
            7 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K7_O0,
                tables::IFFT_DIT4_NIBBLE_K7_O0_TAIL.as_ref(),
            ),
            8 => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K8_O0,
                tables::IFFT_DIT4_NIBBLE_K8_O0_TAIL.as_ref(),
            ),
            _ => unreachable!("k={k} must be in 0..=8"),
        }
    } else {
        match (k, u8::from(beta)) {
            (0, _) => {}
            (1, b) if b == CANTOR_SUBSPACE[2] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K1_O2,
                tables::IFFT_DIT4_NIBBLE_K1_O2_TAIL.as_ref(),
            ),
            (2, b) if b == CANTOR_SUBSPACE[4] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K2_O4,
                tables::IFFT_DIT4_NIBBLE_K2_O4_TAIL.as_ref(),
            ),
            (3, b) if b == CANTOR_SUBSPACE[8] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K3_O8,
                tables::IFFT_DIT4_NIBBLE_K3_O8_TAIL.as_ref(),
            ),
            (4, b) if b == CANTOR_SUBSPACE[16] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K4_O16,
                tables::IFFT_DIT4_NIBBLE_K4_O16_TAIL.as_ref(),
            ),
            (5, b) if b == CANTOR_SUBSPACE[32] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K5_O32,
                tables::IFFT_DIT4_NIBBLE_K5_O32_TAIL.as_ref(),
            ),
            (6, b) if b == CANTOR_SUBSPACE[64] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K6_O64,
                tables::IFFT_DIT4_NIBBLE_K6_O64_TAIL.as_ref(),
            ),
            (7, b) if b == CANTOR_SUBSPACE[128] => K::ifft_sharded_dit4_with(
                shards,
                shard_len,
                k,
                &tables::IFFT_DIT4_NIBBLE_K7_O128,
                tables::IFFT_DIT4_NIBBLE_K7_O128_TAIL.as_ref(),
            ),
            _ => {
                let t = tables::CANTOR_SUBSPACE_INDEX[beta.into_usize()] as usize;
                K::ifft_sharded_dit4_indexed(shards, shard_len, k, t);
            }
        }
    }
}

#[derive(Default)]
pub struct NeonKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for NeonKernel<Gf2p8_11d> {
    const ALIGN: usize = 16;

    type MulTable = MulTable;

    const NODE_MUL: &'static [Self::MulTable; 510] = &tables::NODE_MUL_NIBBLE;

    fn mul_table(t: Gf2p8_11d) -> Self::MulTable {
        &NIBBLE_MUL_TABLE[t.into_usize()]
    }

    fn is_zero_mul(m: &Self::MulTable) -> bool {
        m.0[1] == 0
    }

    fn butterfly_fwd_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::MulTable,
    ) {
        butterfly_fwd_dit2(shards, shard_len, base, d, m);
    }

    fn butterfly_fwd_dit2_zero(shards: &mut [Gf2p8_11d], shard_len: usize, base: usize, d: usize) {
        butterfly_fwd_dit2_zero(shards, shard_len, base, d);
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
        butterfly_fwd_dit4(shards, shard_len, base, d, m01, m23, m02);
    }

    fn butterfly_fwd_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::MulTable,
    ) {
        butterfly_fwd_dit4_zero(shards, shard_len, base, d, m23);
    }

    fn butterfly_inv_dit2(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::MulTable,
    ) {
        butterfly_inv_dit2(shards, shard_len, base, d, m);
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
        butterfly_inv_dit4(shards, shard_len, base, d, m01, m23, m02);
    }

    fn butterfly_inv_dit4_zero(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        base: usize,
        d: usize,
        m23: Self::MulTable,
    ) {
        butterfly_inv_dit4_zero(shards, shard_len, base, d, m23);
    }

    fn fft_sharded_zero_padded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, log_support: u8) {
        match (k, log_support) {
            (1, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k1_s1(shards, shard_len),
            (2, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k2_s1(shards, shard_len),
            (2, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k2_s2(shards, shard_len),
            (3, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k3_s1(shards, shard_len),
            (3, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k3_s2(shards, shard_len),
            (3, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k3_s4(shards, shard_len),
            (4, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k4_s1(shards, shard_len),
            (4, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k4_s2(shards, shard_len),
            (4, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k4_s4(shards, shard_len),
            (4, 3) => unrolled_11d::fft_sharded_zero_padded_neon_k4_s8(shards, shard_len),
            (5, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k5_s1(shards, shard_len),
            (5, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k5_s2(shards, shard_len),
            (5, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k5_s4(shards, shard_len),
            (5, 3) => unrolled_11d::fft_sharded_zero_padded_neon_k5_s8(shards, shard_len),
            (5, 4) => unrolled_11d::fft_sharded_zero_padded_neon_k5_s16(shards, shard_len),
            (6, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s1(shards, shard_len),
            (6, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s2(shards, shard_len),
            (6, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s4(shards, shard_len),
            (6, 3) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s8(shards, shard_len),
            (6, 4) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s16(shards, shard_len),
            (6, 5) => unrolled_11d::fft_sharded_zero_padded_neon_k6_s32(shards, shard_len),
            (7, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s1(shards, shard_len),
            (7, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s2(shards, shard_len),
            (7, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s4(shards, shard_len),
            (7, 3) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s8(shards, shard_len),
            (7, 4) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s16(shards, shard_len),
            (7, 5) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s32(shards, shard_len),
            (7, 6) => unrolled_11d::fft_sharded_zero_padded_neon_k7_s64(shards, shard_len),
            (8, 0) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s1(shards, shard_len),
            (8, 1) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s2(shards, shard_len),
            (8, 2) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s4(shards, shard_len),
            (8, 3) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s8(shards, shard_len),
            (8, 4) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s16(shards, shard_len),
            (8, 5) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s32(shards, shard_len),
            (8, 6) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s64(shards, shard_len),
            (8, 7) => unrolled_11d::fft_sharded_zero_padded_neon_k8_s128(shards, shard_len),
            _ => unreachable!("k={k} must be in 1..=8 and log_support must be < k"),
        }
    }

    fn fft_sharded_radix2_last(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        fft_sharded_radix2_last(shards, shard_len, k, beta);
    }

    fn fft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        fft_sharded(shards, shard_len, k, beta);
    }

    fn ifft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        ifft_sharded(shards, shard_len, k, beta);
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        scale(src, dst, dst.len(), m);
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = &NIBBLE_MUL_TABLE[scalar.into_usize()];
        scale_in_place(dst, dst.len(), m);
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        scale(src, dst, dst.len(), m);
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        let m = &NIBBLE_MUL_BY_LOG[log_m.0 as usize];
        scale_in_place(dst, dst.len(), m);
    }
}

#[cfg(test)]
#[cfg(native_neon)]
mod tests {
    use super::*;
    use crate::{gf2p8lut::CantorBasisLut, kernel::lut_kernel, poly_11d_lut::CantorBasisLut11d};
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_neon_cfg() {
        let target_aarch64 = cfg!(target_arch = "aarch64");

        assert!(target_aarch64);
    }

    #[test]
    fn mul_vec_matches_mul_lut() {
        for p in 0..=255u8 {
            let m = &NIBBLE_MUL_TABLE[p as usize];
            for x in 0..=255u8 {
                let expected = Gf2p8_11d(x).mul_lut(Gf2p8_11d(p));

                let actual = unsafe {
                    let v = vdupq_n_u8(x); // copy x cross all 16 lanes
                    Gf2p8_11d(vgetq_lane_u8::<0>(mul_vec(v, m)))
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

    /// NEON FFT produces the same evaluations as the LUT butterfly.
    /// shard_len covers: pure tail (15), aligned (16), aligned + tail (17), two aligned (32).
    #[test]
    fn fft_neon_matches_lut_rec() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                // Non-zero beta so twiddles are not trivially zero.
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = expected.clone();

                lut_kernel::fft_sharded_recursive(&basis, &mut expected, shard_len, k, beta);
                fft_sharded(&mut actual, shard_len, k, beta);

                assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// NEON IFFT produces the same coefficients as the LUT butterfly.
    #[test]
    fn ifft_neon_matches_lut_rec() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = expected.clone();

                lut_kernel::ifft_sharded_recursive(&basis, &mut expected, shard_len, k, beta);
                ifft_sharded(&mut actual, shard_len, k, beta);

                assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// IFFT;FFT ~= Id.
    #[test]
    fn ifft_then_fft_neon_is_identity() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let original = make_shards(n, shard_len);
                let mut data = original.clone();

                ifft_sharded(&mut data, shard_len, k, beta);
                fft_sharded(&mut data, shard_len, k, beta);

                assert_eq!(data, original, "k={k} shard_len={shard_len}");
            }
        }
    }
}

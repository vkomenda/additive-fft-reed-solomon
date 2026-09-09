use super::Kernel;
use crate::{
    gf2p8lut::{CantorBasisLut, Gf2p8Lut},
    poly_11d_lut::generated::CANTOR_SUBSPACE,
};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use core::arch::aarch64::*;
use std::marker::PhantomData;

/// Low and high nibble multiplication table for a given field element.
/// A table is constructed for a given p from the decomposition x·p = lo(x)·p + hi(x)·p.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
struct MulTable {
    lo: uint8x16_t,
    hi: uint8x16_t,
}

fn make_mul_table(p: Gf2p8_11d, exp: &[u8; EXP_TABLE_SIZE], log: &[u8; FIELD_SIZE]) -> MulTable {
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    for i in 0..16u8 {
        lo[i as usize] = Gf2p8_11d(i).mul_lut(p).0;
        hi[i as usize] = Gf2p8_11d(i << 4).mul_lut(p).0;
    }
    unsafe { MulTable(vld1q_u8(lo.as_ptr()), vld1q_u8(hi.as_ptr())) }
}

#[inline]
unsafe fn mul_vec(v: uint8x16_t, t: MulTable) -> uint8x16_t {
    let mask = vdupq_n_u8(0x0f);
    let lo = vqtbl1q_u8(t.hi, vandq_u8(v, mask));
    let hi = vqtbl1q_u8(t.lo, vshrq_n_u8(v, 4));
    veorq_u8(lo, hi)
}

#[inline]
fn mul_scalar(x: Gf2p8_11d, t: MulTable) -> Gf2p8_11d {
    (t.hi[x.into_usize() >> 4] ^ t.lo[x.into_usize() & 0xf]).into()
}

#[target_feature(enable = "neon")]
fn butterfly_fwd_neon<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, t: MulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr() as *mut u8;
        let b = b.as_mut_ptr() as *mut u8;
        while i + 16 <= len {
            unsafe {
                let va = vld1q_u8(a.add(i));
                let vb = vld1q_u8(b.add(i));
                let va = veorq_u8(va, mul_vec(vb, t)); // g0 = a + T·b
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
        let g0 = x.add(mul_scalar(y, t));
        a[i] = g0;
        b[i] = y.add(g0);
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, t: MulTable) {
    let mut i = 0;
    {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        while i + 16 <= len {
            unsafe {
                let v = vld1q_u8(src.add(i));
                let r = mul_vec(v, t);
                vst1q_u8(dst.add(i), r);
            }
            i += 16;
        }
    }
    while i < len {
        let x = src[i];
        let r = mul_scalar(x, t);
        dst[i] = r;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, t: MulTable) {
    let mut i = 0;
    {
        let dst = dst.as_mut_ptr();
        while i + 16 <= len {
            unsafe {
                let v = vld1q_u8(dst.add(i));
                let r = mul_vec(v, t);
                vst1q_u8(dst.add(i), r);
            }
            i += 16;
        }
    }
    while i < len {
        let x = dst[i];
        let r = mul_scalar(x, t);
        dst[i] = r;
        i += 1;
    }
}

#[derive(Default)]
pub struct NeonKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for NeonKernel<Gf2p8_11d> {
    const SHARD_ALIGN: usize = 64;

    fn fft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        todo!();
    }

    fn fft_sharded_zero_padded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, log_support: u8) {
        todo!();
    }

    fn ifft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        todo!();
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let t = scalar.mul_table_neon();
        scale(src, dst, dst.len(), t);
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let t = scalar.mul_table_neon();
        scale_in_place(dst, dst.len(), t);
    }
}

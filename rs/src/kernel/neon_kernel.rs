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
    lo: uint8x16t,
    hi: uint8x16_t,
    p: Gf2p8_11d,
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

#[target_feature(enable = "neon")]
fn butterfly_fwd_neon<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, t: MulTable) {
    let a = a.as_mut_ptr() as *mut u8;
    let b = b.as_mut_ptr() as *mut u8;
    let mut i = 0;
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
    // Handle the tail scalar. There are no masked load/store ops in NEON, hence use LUTs.
    while i < len {
        let x = a[i];
        let y = b[i];
        let g0 = x.add(y.mul_lut(t.p));
        a[i] = g0;
        b[i] = y.add(g0);
        i += 1;
    }
}

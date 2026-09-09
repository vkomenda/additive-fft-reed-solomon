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
unsafe fn mul_vec(v: uint8x16_t, m: MulTable) -> uint8x16_t {
    let mask = vdupq_n_u8(0x0f);
    let lo = vqtbl1q_u8(m.hi, vandq_u8(v, mask));
    let hi = vqtbl1q_u8(m.lo, vshrq_n_u8(v, 4));
    veorq_u8(lo, hi)
}

#[inline]
fn mul_scalar(x: Gf2p8_11d, m: MulTable) -> Gf2p8_11d {
    (m.hi[x.into_usize() >> 4] ^ m.lo[x.into_usize() & 0xf]).into()
}

#[target_feature(enable = "neon")]
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: MulTable) {
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
        let g0 = x.add(mul_scalar(y, m));
        a[i] = g0;
        b[i] = y.add(g0);
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], len: usize, m: MulTable) {
    let mut i = 0;
    {
        let a = a.as_mut_ptr();
        let b = b.as_mut_ptr();
        while i + 16 <= len {
            unsafe {
                let va = vld1q_u8(a.add(i));
                let vb = vld1q_u8(b.add(i));
                let vb = veorq_u8(vb, va);
                let t = veorq_u8(va, mul_vec(vb, m));
                let va = veorq_u8(va, m);
                vst1q_u8(a.add(i), va);
                vst1q_u8(b.add(i), vb);
            }
            i += 16;
        }
    }
    if i < len {
        let x = a[i];
        let y = b[i];
        let y = x.add(y);
        let x = x.add(mul_scalar(y, m));
        a[i] = x;
        b[i] = y;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
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
    let t = make_mul_table(twiddle);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_fwd(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            t,
        );
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    fft_sharded(basis, &mut shards[..h], shard_len, k - 1, beta);
    fft_sharded(basis, &mut shards[h..], shard_len, k - 1, next_beta);
}

#[target_feature(enable = "neon")]
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
    let t = make_mul_table(twiddle);

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        butterfly_inv(
            &mut left[i * shard_len..],
            &mut right[..shard_len],
            shard_len,
            t,
        )
    }
}

#[target_feature(enable = "neon")]
fn scale<G: Gf2p8>(src: &[G], dst: &mut [G], len: usize, m: MulTable) {
    let mut i = 0;
    {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
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
        let r = mul_scalar(x, m);
        dst[i] = r;
        i += 1;
    }
}

#[target_feature(enable = "neon")]
fn scale_in_place<G: Gf2p8>(dst: &mut [G], len: usize, m: MulTable) {
    let mut i = 0;
    {
        let dst = dst.as_mut_ptr();
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
        let r = mul_scalar(x, m);
        dst[i] = r;
        i += 1;
    }
}

#[derive(Default)]
pub struct NeonKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for NeonKernel<Gf2p8_11d> {
    const SHARD_ALIGN: usize = 16;

    fn fft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        // FIXME: temporary recursive instantiation
        fft_sharded(basis, shards, shard_len, k, beta);
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
        // FIXME: temporary recursive instantiation
        ifft_sharded(basis, shards, shard_len, k, beta);
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = make_mul_table(scalar);
        scale(src, dst, dst.len(), m);
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        let m = make_mul_table(scalar);
        scale_in_place(dst, dst.len(), m);
    }
}

#[cfg(test)]
#[cfg(native_neon)]
mod tests {
    use super::*;
    use crate::{kernel::lut_kernel, poly_11d_lut::CantorBasisLut11d};
    use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

    #[test]
    fn debug_neon_cfg() {
        let target_aarch64 = cfg!(target_arch = "aarch64");

        assert!(target_aarch64);
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
    fn fft_neon_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                // Non-zero beta so twiddles are not trivially zero.
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = lut.clone();

                lut_kernel::fft_sharded(&basis, &mut expected, shard_len, k, beta);
                unsafe {
                    fft_sharded(&basis, &mut actual, shard_len, k, beta);
                }

                assert_eq!(expected, actual, "k={k} shard_len={shard_len}");
            }
        }
    }

    /// NEON IFFT produces the same coefficients as the LUT butterfly.
    #[test]
    fn ifft_neon_matches_lut() {
        let basis = CantorBasisLut11d;
        for shard_len in [1, 15, 16, 17, 32] {
            for k in 1u8..=4 {
                let n = 1 << k;
                let beta = basis.get_subspace_point_lut(n as u8);
                let mut expected = make_shards(n, shard_len);
                let mut actual = lut.clone();

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
    fn ifft_then_fft_neon_is_identity() {
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

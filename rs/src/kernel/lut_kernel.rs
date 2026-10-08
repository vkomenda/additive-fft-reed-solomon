use super::{Kernel, shard_groups};
#[cfg(test)]
use crate::gf2p8lut::CantorBasisLut;
use crate::gf2p8lut::Gf2p8Lut;
use crate::poly_11d_lut::CantorBasisLut11d;
use crate::poly_11d_lut::generated::{self as tables, CANTOR_SUBSPACE, EXP_TABLE, MUL_TABLE};
use additive_fft_reed_solomon_gf2p8::{FIELD_SIZE, Gf2p8, Gf2p8_11d, Z255};
use std::marker::PhantomData;

type MulTable = &'static [u8; FIELD_SIZE];

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_lut_kernel_11d.rs"));
}

/// Forward butterfly on one shard pair.
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], _len: usize, lut: &[u8; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        let t = G::from(lut[bi.into_usize()]); // T * b
        *ai = ai.add(t); // g0 = a + T*b
        *bi = bi.add(*ai); // g1 = g0 + b
    }
}

/// Inverse butterfly on one shard pair.
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], _len: usize, lut: &[u8; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        *bi = bi.add(*ai); //  d' = g0 + g1
        *ai = ai.add(G::from(lut[bi.into_usize()])); //  d = g0 + T*d'
    }
}

fn butterfly_fwd_dit2<G: Gf2p8>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
    m: MulTable,
) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
            *ai = ai.add(G::from(m[bi.into_usize()])); // g0 = a + T*b
            *bi = bi.add(*ai); // g1 = g0 + b
        }
    }
}

fn butterfly_fwd_dit2_zero<G: Gf2p8>(shards: &mut [G], shard_len: usize, base: usize, d: usize) {
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
            *bi = bi.add(*ai); // g1 = g0 + b
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
    for [a, b] in shard_groups::<_, 2>(shards, shard_len, base, d) {
        for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
            *bi = bi.add(*ai); //  d' = g0 + g1
            *ai = ai.add(G::from(m[bi.into_usize()])); //  d = g0 + T*d'
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
    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        for (((w0, w1), w2), w3) in s0
            .iter_mut()
            .zip(s1.iter_mut())
            .zip(s2.iter_mut())
            .zip(s3.iter_mut())
        {
            *w0 = w0.add(G::from(m02[w2.into_usize()]));
            *w1 = w1.add(G::from(m02[w3.into_usize()]));
            *w2 = w2.add(*w0);
            *w3 = w3.add(*w1);
            *w0 = w0.add(G::from(m01[w1.into_usize()]));
            *w2 = w2.add(G::from(m23[w3.into_usize()]));
            *w1 = w1.add(*w0);
            *w3 = w3.add(*w2);
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
    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        for (((w0, w1), w2), w3) in s0
            .iter()
            .zip(s1.iter_mut())
            .zip(s2.iter_mut())
            .zip(s3.iter_mut())
        {
            *w2 = w2.add(*w0);
            *w3 = w3.add(*w1);
            *w2 = w2.add(G::from(m23[w3.into_usize()]));
            *w1 = w1.add(*w0);
            *w3 = w3.add(*w2);
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
    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        for (((w0, w1), w2), w3) in s0
            .iter_mut()
            .zip(s1.iter_mut())
            .zip(s2.iter_mut())
            .zip(s3.iter_mut())
        {
            *w3 = w3.add(*w2);
            *w1 = w1.add(*w0);
            *w2 = w2.add(G::from(m23[w3.into_usize()]));
            *w0 = w0.add(G::from(m01[w1.into_usize()]));
            *w3 = w3.add(*w1);
            *w2 = w2.add(*w0);
            *w1 = w1.add(G::from(m02[w3.into_usize()]));
            *w0 = w0.add(G::from(m02[w2.into_usize()]));
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
    for [s0, s1, s2, s3] in shard_groups::<_, 4>(shards, shard_len, base, d) {
        for (((w0, w1), w2), w3) in s0
            .iter()
            .zip(s1.iter_mut())
            .zip(s2.iter_mut())
            .zip(s3.iter_mut())
        {
            *w3 = w3.add(*w2);
            *w1 = w1.add(*w0);
            *w2 = w2.add(G::from(m23[w3.into_usize()]));
            *w3 = w3.add(*w1);
            *w2 = w2.add(*w0);
        }
    }
}

#[cfg(test)]
pub(crate) fn fft_sharded_recursive<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G],
    shard_len: usize,
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1 << (k - 1);
    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let lut = &MUL_TABLE[twiddle.into_usize()];

    // Butterfly with one lut computed for the whole pass
    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        let a = &mut left[i * shard_len..(i + 1) * shard_len];
        let b = &mut right[..shard_len];
        butterfly_fwd(a, b, 0, lut);
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    fft_sharded_recursive(basis, &mut shards[..h], shard_len, k - 1, beta);
    fft_sharded_recursive(basis, &mut shards[h..], shard_len, k - 1, next_beta);
}

#[cfg(test)]
pub(crate) fn fft_sharded_zero_padded_recursive<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G],
    shard_len: usize,
    k: u8,
    beta: G,
    log_support: u8,
) {
    if k == 0 {
        return;
    }
    if log_support >= k {
        return fft_sharded_recursive(basis, shards, shard_len, k, beta);
    }
    let h = (1usize << (k - 1)) * shard_len;
    let (lo, hi) = shards.split_at_mut(h);
    hi.copy_from_slice(lo);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    fft_sharded_zero_padded_recursive(basis, lo, shard_len, k - 1, beta, log_support);
    fft_sharded_zero_padded_recursive(basis, hi, shard_len, k - 1, next_beta, log_support);
}

#[cfg(test)]
pub(crate) fn ifft_sharded_recursive<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G],
    shard_len: usize,
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1 << (k - 1);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    ifft_sharded_recursive(basis, &mut shards[..h], shard_len, k - 1, beta);
    ifft_sharded_recursive(basis, &mut shards[h..], shard_len, k - 1, next_beta);

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let lut = &MUL_TABLE[twiddle.into_usize()];

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        let a = &mut left[i * shard_len..(i + 1) * shard_len];
        let b = &mut right[..shard_len];
        butterfly_inv(a, b, 0, lut);
    }
}

pub(crate) fn scale<G: Gf2p8Lut>(src: &[G], dst: &mut [G], scalar: G) {
    let lut = &MUL_TABLE[scalar.into_usize()];
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = G::from(lut[s.into_usize()]);
    }
}

pub(crate) fn scale_in_place<G: Gf2p8Lut>(dst: &mut [G], scalar: G) {
    let lut = &MUL_TABLE[scalar.into_usize()];
    for b in dst.iter_mut() {
        *b = G::from(lut[b.into_usize()]);
    }
}

#[derive(Default)]
pub struct LutKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for LutKernel<Gf2p8_11d> {
    const ALIGN: usize = 1;

    type MulTable = MulTable;

    fn mul_table(t: Gf2p8_11d) -> Self::MulTable {
        &MUL_TABLE[t.into_usize()]
    }

    fn is_zero_mul(m: &Self::MulTable) -> bool {
        m[1] == 0
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

    fn fft_sharded_zero_padded_unrolled(
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        log_support: u8,
    ) {
        match (k, log_support) {
            (1, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k1_s1(shards, shard_len),
            (2, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k2_s1(shards, shard_len),
            (2, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k2_s2(shards, shard_len),
            (3, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k3_s1(shards, shard_len),
            (3, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k3_s2(shards, shard_len),
            (3, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k3_s4(shards, shard_len),
            (4, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k4_s1(shards, shard_len),
            (4, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k4_s2(shards, shard_len),
            (4, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k4_s4(shards, shard_len),
            (4, 3) => unrolled_11d::fft_sharded_zero_padded_lut_k4_s8(shards, shard_len),
            (5, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k5_s1(shards, shard_len),
            (5, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k5_s2(shards, shard_len),
            (5, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k5_s4(shards, shard_len),
            (5, 3) => unrolled_11d::fft_sharded_zero_padded_lut_k5_s8(shards, shard_len),
            (5, 4) => unrolled_11d::fft_sharded_zero_padded_lut_k5_s16(shards, shard_len),
            (6, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s1(shards, shard_len),
            (6, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s2(shards, shard_len),
            (6, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s4(shards, shard_len),
            (6, 3) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s8(shards, shard_len),
            (6, 4) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s16(shards, shard_len),
            (6, 5) => unrolled_11d::fft_sharded_zero_padded_lut_k6_s32(shards, shard_len),
            (7, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s1(shards, shard_len),
            (7, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s2(shards, shard_len),
            (7, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s4(shards, shard_len),
            (7, 3) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s8(shards, shard_len),
            (7, 4) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s16(shards, shard_len),
            (7, 5) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s32(shards, shard_len),
            (7, 6) => unrolled_11d::fft_sharded_zero_padded_lut_k7_s64(shards, shard_len),
            (8, 0) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s1(shards, shard_len),
            (8, 1) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s2(shards, shard_len),
            (8, 2) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s4(shards, shard_len),
            (8, 3) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s8(shards, shard_len),
            (8, 4) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s16(shards, shard_len),
            (8, 5) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s32(shards, shard_len),
            (8, 6) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s64(shards, shard_len),
            (8, 7) => unrolled_11d::fft_sharded_zero_padded_lut_k8_s128(shards, shard_len),
            _ => unreachable!("k={k} must be in 1..=8 and log_support must be < k"),
        }
    }

    fn fft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        if beta == Gf2p8_11d::zero() {
            match k {
                0 => {}
                1 => unrolled_11d::fft_sharded_lut_k1(shards, shard_len),
                2 => unrolled_11d::fft_sharded_lut_k2(shards, shard_len),
                3 => unrolled_11d::fft_sharded_lut_k3(shards, shard_len),
                4 => unrolled_11d::fft_sharded_lut_k4(shards, shard_len),
                5 => unrolled_11d::fft_sharded_lut_k5(shards, shard_len),
                6 => unrolled_11d::fft_sharded_lut_k6(shards, shard_len),
                7 => unrolled_11d::fft_sharded_lut_k7(shards, shard_len),
                8 => unrolled_11d::fft_sharded_lut_k8(shards, shard_len),
                _ => unreachable!("k={k} must be in 0..=8"),
            }
        } else {
            Self::fft_sharded_dit4(&CantorBasisLut11d, shards, shard_len, k, beta);
        }
    }

    fn ifft_sharded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        if beta == Gf2p8::zero() {
            match k {
                0 => {}
                1 => unrolled_11d::ifft_sharded_lut_k1(shards, shard_len),
                2 => unrolled_11d::ifft_sharded_lut_k2(shards, shard_len),
                3 => unrolled_11d::ifft_sharded_lut_k3(shards, shard_len),
                4 => unrolled_11d::ifft_sharded_lut_k4(shards, shard_len),
                5 => unrolled_11d::ifft_sharded_lut_k5(shards, shard_len),
                6 => unrolled_11d::ifft_sharded_lut_k6(shards, shard_len),
                7 => unrolled_11d::ifft_sharded_lut_k7(shards, shard_len),
                8 => unrolled_11d::ifft_sharded_lut_k8(shards, shard_len),
                _ => unreachable!("k={k} must be in 0..=8"),
            }
        } else {
            match (k, u8::from(beta)) {
                (0, _) => {}
                (1, b) if b == CANTOR_SUBSPACE[2] => {
                    unrolled_11d::ifft_sharded_lut_k1_od6(shards, shard_len)
                }
                (2, b) if b == CANTOR_SUBSPACE[4] => {
                    unrolled_11d::ifft_sharded_lut_k2_o98(shards, shard_len)
                }
                (3, b) if b == CANTOR_SUBSPACE[8] => {
                    unrolled_11d::ifft_sharded_lut_k3_o92(shards, shard_len)
                }
                (4, b) if b == CANTOR_SUBSPACE[16] => {
                    unrolled_11d::ifft_sharded_lut_k4_o56(shards, shard_len)
                }
                (5, b) if b == CANTOR_SUBSPACE[32] => {
                    unrolled_11d::ifft_sharded_lut_k5_oc8(shards, shard_len)
                }
                (6, b) if b == CANTOR_SUBSPACE[64] => {
                    unrolled_11d::ifft_sharded_lut_k6_o58(shards, shard_len)
                }
                (7, b) if b == CANTOR_SUBSPACE[128] => {
                    unrolled_11d::ifft_sharded_lut_k7_oe7(shards, shard_len)
                }
                _ => Self::ifft_sharded_dit4(&CantorBasisLut11d, shards, shard_len, k, beta),
            }
        }
    }

    fn fft_sharded_radix2_last(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, beta: Gf2p8_11d) {
        if beta == Gf2p8_11d::zero() {
            match k {
                1 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K1_O0,
                    &tables::FFT_DIT4_LUT_K1_O0_TAIL,
                ),
                2 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K2_O0,
                    &tables::FFT_DIT4_LUT_K2_O0_TAIL,
                ),
                3 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K3_O0,
                    &tables::FFT_DIT4_LUT_K3_O0_TAIL,
                ),
                4 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K4_O0,
                    &tables::FFT_DIT4_LUT_K4_O0_TAIL,
                ),
                5 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K5_O0,
                    &tables::FFT_DIT4_LUT_K5_O0_TAIL,
                ),
                6 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K6_O0,
                    &tables::FFT_DIT4_LUT_K6_O0_TAIL,
                ),
                7 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K7_O0,
                    &tables::FFT_DIT4_LUT_K7_O0_TAIL,
                ),
                8 => Self::fft_sharded_dit4_radix2_last_with(
                    shards,
                    shard_len,
                    k,
                    &tables::FFT_DIT4_LUT_K8_O0,
                    &tables::FFT_DIT4_LUT_K8_O0_TAIL,
                ),
                _ => unreachable!("k={k} must be in 1..=8"),
            }
        } else {
            Self::fft_sharded_dit4(&CantorBasisLut11d, shards, shard_len, k, beta);
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        scale(src, dst, scalar)
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        scale_in_place(dst, scalar)
    }

    fn scale_by_log(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], log_m: Z255) {
        // TODO: use a log-based multiplication table
        let m = EXP_TABLE[log_m.0 as usize];
        scale(src, dst, m.into());
    }

    fn scale_in_place_by_log(dst: &mut [Gf2p8_11d], log_m: Z255) {
        // TODO: use a log-based multiplication table
        let m = EXP_TABLE[log_m.0 as usize];
        scale_in_place(dst, m.into());
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly_11d_lut::CantorBasisLut11d;
    use rand::rngs::SmallRng;
    use rand::{Rng, RngExt, SeedableRng};

    #[test]
    fn fft_sharded_zero_padded_matches_fft_sharded_rec() {
        const SHARD_LEN: usize = 37;

        let mut rng = SmallRng::seed_from_u64(42);

        for k in 1..=8u8 {
            let n = 1usize << k;
            for log_support in 0..k {
                let support = 1usize << log_support;
                let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
                let support_len = support * SHARD_LEN;
                rng.fill_bytes(unsafe {
                    std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, support_len)
                });
                a[support * SHARD_LEN..].fill(Gf2p8_11d::zero());
                let mut b = a.clone();
                b[support * SHARD_LEN..].fill(Gf2p8_11d::from(0xaa));

                let basis = CantorBasisLut11d;

                let beta = Gf2p8_11d::zero();
                fft_sharded_recursive(&basis, &mut a, SHARD_LEN, k, beta);
                fft_sharded_zero_padded_recursive(&basis, &mut b, SHARD_LEN, k, beta, log_support);

                assert_eq!(a, b, "k={k} log_support={log_support}");
            }
        }
    }

    #[test]
    fn unrolled_fft_sharded_zero_padded_matches_fft_sharded_rec() {
        const SHARD_LEN: usize = 37;

        let mut rng = SmallRng::seed_from_u64(42);

        let k = 5;
        let n = 1 << k;
        let log_support = 3;
        let support = 1 << log_support;

        let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
        let support_len = support * SHARD_LEN;
        rng.fill_bytes(unsafe {
            std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, support_len)
        });
        a[support * SHARD_LEN..].fill(Gf2p8_11d::zero());
        let mut b = a.clone();
        b[support * SHARD_LEN..].fill(Gf2p8_11d::from(0xaa));

        let basis = CantorBasisLut11d;
        fft_sharded_recursive(&basis, &mut a, SHARD_LEN, k, Gf2p8_11d::zero());
        unrolled_11d::fft_sharded_zero_padded_lut_k5_s8(&mut b, SHARD_LEN);

        assert_eq!(a, b);
    }

    #[test]
    fn dit4_matches_two_dit2_levels() {
        const SHARD_LEN: usize = 37;
        let mut rng = SmallRng::seed_from_u64(1);

        for d in [1, 2, 4, 8] {
            for base in [0, 1, 3] {
                let n = base + 4 * d;

                let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
                rng.fill_bytes(unsafe {
                    std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
                });
                let mut b = a.clone();

                let m02 = LutKernel::mul_table(Gf2p8_11d(0x53));
                let m01 = LutKernel::mul_table(Gf2p8_11d(0x2a));
                let m23 = LutKernel::mul_table(Gf2p8_11d(0xc7));

                // The wide level over the whole node, then the narrow level over each half.
                LutKernel::butterfly_fwd_dit2(&mut a, SHARD_LEN, base, 2 * d, m02);
                LutKernel::butterfly_fwd_dit2(&mut a, SHARD_LEN, base, d, m01);
                LutKernel::butterfly_fwd_dit2(&mut a, SHARD_LEN, base + 2 * d, d, m23);

                LutKernel::butterfly_fwd_dit4(&mut b, SHARD_LEN, base, d, m01, m23, m02);

                assert_eq!(a, b, "d={d} base={base}");
            }
        }
    }

    #[test]
    fn fft_sharded_dit4_matches_fft_sharded_rec() {
        const SHARD_LEN: usize = 37;

        let mut rng = SmallRng::seed_from_u64(42);
        let beta: u8 = rng.random();
        let beta = Gf2p8_11d::from(beta);

        for k in 1..=8 {
            let n = 1 << k;

            let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
            rng.fill_bytes(unsafe {
                std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
            });
            let mut b = a.clone();

            fft_sharded_recursive(&CantorBasisLut11d, &mut a, SHARD_LEN, k, beta);
            LutKernel::fft_sharded_dit4(&CantorBasisLut11d, &mut b, SHARD_LEN, k, beta);

            assert_eq!(a, b, "k={k}");
        }
    }

    #[test]
    fn ifft_sharded_dit4_matches_ifft_sharded_rec() {
        const SHARD_LEN: usize = 37;

        let mut rng = SmallRng::seed_from_u64(42);
        let beta: u8 = rng.random();
        let beta = Gf2p8_11d::from(beta);

        for k in 1..=8 {
            let n = 1 << k;

            let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
            rng.fill_bytes(unsafe {
                std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
            });
            let mut b = a.clone();

            ifft_sharded_recursive(&CantorBasisLut11d, &mut a, SHARD_LEN, k, beta);
            LutKernel::ifft_sharded_dit4(&CantorBasisLut11d, &mut b, SHARD_LEN, k, beta);

            assert_eq!(a, b, "k={k}");
        }
    }
}

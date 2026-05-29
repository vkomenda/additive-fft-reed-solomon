use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};
use additive_fft_reed_solomon_gf2p8::FIELD_SIZE;
use std::marker::PhantomData;

use super::Kernel;

/// Forward butterfly on one shard pair.
fn butterfly_fwd<G: Gf2p8Lut>(a: &mut [G], b: &mut [G], lut: &[G; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        let t = lut[bi.into_usize()]; // T * b
        *ai = ai.add(t); // g0 = a + T*b
        *bi = bi.add(*ai); // g1 = g0 + b
    }
}

/// Inverse butterfly on one shard pair.
fn butterfly_inv<G: Gf2p8Lut>(a: &mut [G], b: &mut [G], lut: &[G; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        *bi = bi.add(*ai); //  d' = g0 + g1
        *ai = ai.add(lut[bi.into_usize()]); //  d  = g0 + T*d'
    }
}

pub fn fft_sharded<G: Gf2p8Lut>(
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
    let lut = twiddle.make_mul_lut();

    // Butterfly with one lut computed for the whole pass
    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        let a = &mut left[i * shard_len..(i + 1) * shard_len];
        let b = &mut right[..shard_len];
        butterfly_fwd(a, b, &lut);
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    let h = half * shard_len;
    fft_sharded(basis, &mut shards[..h], shard_len, k - 1, beta);
    fft_sharded(basis, &mut shards[h..], shard_len, k - 1, next_beta);
}

pub fn ifft_sharded<G: Gf2p8Lut>(
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
    ifft_sharded(basis, &mut shards[..h], shard_len, k - 1, beta);
    ifft_sharded(basis, &mut shards[h..], shard_len, k - 1, next_beta);

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let lut = twiddle.make_mul_lut();

    for i in 0..half {
        let (left, right) = shards.split_at_mut((i + half) * shard_len);
        let a = &mut left[i * shard_len..(i + 1) * shard_len];
        let b = &mut right[..shard_len];
        butterfly_inv(a, b, &lut);
    }
}

pub fn scale<G: Gf2p8Lut>(dst: &mut [G], src: &[G], scalar: G) {
    let lut = scalar.make_mul_lut();
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = lut[s.into_usize()];
    }
}

pub fn scale_in_place<G: Gf2p8Lut>(dst: &mut [G], scalar: G) {
    let lut = scalar.make_mul_lut();
    for b in dst.iter_mut() {
        *b = lut[b.into_usize()];
    }
}

pub struct LutKernel<G: Gf2p8Lut>(PhantomData<G>);

impl<G: Gf2p8Lut> Kernel<G> for LutKernel<G> {
    fn fft_sharded(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        fft_sharded(basis, shards, shard_len, k, beta)
    }

    fn ifft_sharded(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        ifft_sharded(basis, shards, shard_len, k, beta)
    }

    fn scale(dst: &mut [G], src: &[G], scalar: G) {
        scale(dst, src, scalar)
    }

    fn scale_in_place(dst: &mut [G], scalar: G) {
        scale_in_place(dst, scalar)
    }
}

use std::marker::PhantomData;

use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};

use super::Kernel;

pub fn fft_sharded<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [&mut [G]],
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
        let (left, right) = shards.split_at_mut(i + half);
        // Forward butterfly
        for (ai, bi) in left[i].iter_mut().zip(right[0].iter_mut()) {
            let t = lut[bi.into_usize()]; //  T * b
            *ai = ai.add(t); //  g0 = a + T*b
            *bi = bi.add(*ai); //  g1 = g0 + b
        }
    }

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    fft_sharded(basis, &mut shards[..half], k - 1, beta);
    fft_sharded(basis, &mut shards[half..], k - 1, next_beta);
}

pub fn ifft_sharded<G: Gf2p8Lut>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [&mut [G]],
    k: u8,
    beta: G,
) {
    if k == 0 {
        return;
    }
    let half = 1 << (k - 1);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    ifft_sharded(basis, &mut shards[..half], k - 1, beta);
    ifft_sharded(basis, &mut shards[half..], k - 1, next_beta);

    let twiddle = basis.eval_subspace_poly_lut(k - 1, beta);
    let lut = twiddle.make_mul_lut();

    for i in 0..half {
        let (left, right) = shards.split_at_mut(i + half);
        for (ai, bi) in left[i].iter_mut().zip(right[0].iter_mut()) {
            *bi = bi.add(*ai); //  d' = g0 + g1
            let t = lut[bi.into_usize()];
            *ai = ai.add(t); //  d  = g0 + T*d'
        }
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
    fn fft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G) {
        fft_sharded(basis, shards, k, beta)
    }

    fn ifft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G) {
        ifft_sharded(basis, shards, k, beta)
    }

    fn scale(dst: &mut [G], src: &[G], scalar: G) {
        scale(dst, src, scalar)
    }

    fn scale_in_place(dst: &mut [G], scalar: G) {
        scale_in_place(dst, scalar)
    }
}

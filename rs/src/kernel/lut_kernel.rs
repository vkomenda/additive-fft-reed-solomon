use super::Kernel;
use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};
use crate::poly_11d_lut::generated::CANTOR_SUBSPACE;
use additive_fft_reed_solomon_gf2p8::{FIELD_SIZE, Gf2p8, Gf2p8_11d};
use std::marker::PhantomData;

pub mod unrolled_11d {
    include!(concat!(env!("OUT_DIR"), "/unrolled_lut_kernel_11d.rs"));
}

/// Forward butterfly on one shard pair.
fn butterfly_fwd<G: Gf2p8>(a: &mut [G], b: &mut [G], lut: &[u8; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        let t = G::from(lut[bi.into_usize()]); // T * b
        *ai = ai.add(t); // g0 = a + T*b
        *bi = bi.add(*ai); // g1 = g0 + b
    }
}

/// Inverse butterfly on one shard pair.
fn butterfly_inv<G: Gf2p8>(a: &mut [G], b: &mut [G], lut: &[u8; FIELD_SIZE]) {
    for (ai, bi) in a.iter_mut().zip(b.iter_mut()) {
        *bi = bi.add(*ai); //  d' = g0 + g1
        *ai = ai.add(G::from(lut[bi.into_usize()])); //  d  = g0 + T*d'
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

pub fn fft_sharded_zero_padded<G: Gf2p8Lut>(
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
        return fft_sharded(basis, shards, shard_len, k, beta);
    }
    let h = (1usize << (k - 1)) * shard_len;
    let (lo, hi) = shards.split_at_mut(h);
    hi.copy_from_slice(lo);

    let next_beta = beta.add(basis.get_basis_point_lut(k - 1));
    fft_sharded_zero_padded(basis, lo, shard_len, k - 1, beta, log_support);
    fft_sharded_zero_padded(basis, hi, shard_len, k - 1, next_beta, log_support);
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

pub fn scale<G: Gf2p8Lut>(src: &[G], dst: &mut [G], scalar: G) {
    let lut = scalar.make_mul_lut();
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = G::from(lut[s.into_usize()]);
    }
}

pub fn scale_in_place<G: Gf2p8Lut>(dst: &mut [G], scalar: G) {
    let lut = scalar.make_mul_lut();
    for b in dst.iter_mut() {
        *b = G::from(lut[b.into_usize()]);
    }
}

#[derive(Default)]
pub struct LutKernel<G: Gf2p8Lut>(PhantomData<G>);

impl Kernel<Gf2p8_11d> for LutKernel<Gf2p8_11d> {
    fn fft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        if beta == Gf2p8_11d::zero() {
            match k {
                0 => {}
                1 => unrolled_11d::fft_sharded_lut_2(shards, shard_len),
                2 => unrolled_11d::fft_sharded_lut_4(shards, shard_len),
                3 => unrolled_11d::fft_sharded_lut_8(shards, shard_len),
                4 => unrolled_11d::fft_sharded_lut_16(shards, shard_len),
                5 => unrolled_11d::fft_sharded_lut_32(shards, shard_len),
                6 => unrolled_11d::fft_sharded_lut_64(shards, shard_len),
                7 => unrolled_11d::fft_sharded_lut_128(shards, shard_len),
                8 => unrolled_11d::fft_sharded_lut_256(shards, shard_len),
                _ => unreachable!("k={k} must be in 0..=8"),
            }
        } else {
            fft_sharded(basis, shards, shard_len, k, beta);
        }
    }

    fn fft_sharded_zero_padded(shards: &mut [Gf2p8_11d], shard_len: usize, k: u8, log_support: u8) {
        match (k, log_support) {
            (1, 0) => unrolled_11d::fft_sharded_zero_padded_lut_2_1(shards, shard_len),
            (2, 0) => unrolled_11d::fft_sharded_zero_padded_lut_4_1(shards, shard_len),
            (2, 1) => unrolled_11d::fft_sharded_zero_padded_lut_4_2(shards, shard_len),
            (3, 0) => unrolled_11d::fft_sharded_zero_padded_lut_8_1(shards, shard_len),
            (3, 1) => unrolled_11d::fft_sharded_zero_padded_lut_8_2(shards, shard_len),
            (3, 2) => unrolled_11d::fft_sharded_zero_padded_lut_8_4(shards, shard_len),
            (4, 0) => unrolled_11d::fft_sharded_zero_padded_lut_16_1(shards, shard_len),
            (4, 1) => unrolled_11d::fft_sharded_zero_padded_lut_16_2(shards, shard_len),
            (4, 2) => unrolled_11d::fft_sharded_zero_padded_lut_16_4(shards, shard_len),
            (4, 3) => unrolled_11d::fft_sharded_zero_padded_lut_16_8(shards, shard_len),
            (5, 0) => unrolled_11d::fft_sharded_zero_padded_lut_32_1(shards, shard_len),
            (5, 1) => unrolled_11d::fft_sharded_zero_padded_lut_32_2(shards, shard_len),
            (5, 2) => unrolled_11d::fft_sharded_zero_padded_lut_32_4(shards, shard_len),
            (5, 3) => unrolled_11d::fft_sharded_zero_padded_lut_32_8(shards, shard_len),
            (5, 4) => unrolled_11d::fft_sharded_zero_padded_lut_32_16(shards, shard_len),
            (6, 0) => unrolled_11d::fft_sharded_zero_padded_lut_64_1(shards, shard_len),
            (6, 1) => unrolled_11d::fft_sharded_zero_padded_lut_64_2(shards, shard_len),
            (6, 2) => unrolled_11d::fft_sharded_zero_padded_lut_64_4(shards, shard_len),
            (6, 3) => unrolled_11d::fft_sharded_zero_padded_lut_64_8(shards, shard_len),
            (6, 4) => unrolled_11d::fft_sharded_zero_padded_lut_64_16(shards, shard_len),
            (6, 5) => unrolled_11d::fft_sharded_zero_padded_lut_64_32(shards, shard_len),
            (7, 0) => unrolled_11d::fft_sharded_zero_padded_lut_128_1(shards, shard_len),
            (7, 1) => unrolled_11d::fft_sharded_zero_padded_lut_128_2(shards, shard_len),
            (7, 2) => unrolled_11d::fft_sharded_zero_padded_lut_128_4(shards, shard_len),
            (7, 3) => unrolled_11d::fft_sharded_zero_padded_lut_128_8(shards, shard_len),
            (7, 4) => unrolled_11d::fft_sharded_zero_padded_lut_128_16(shards, shard_len),
            (7, 5) => unrolled_11d::fft_sharded_zero_padded_lut_128_32(shards, shard_len),
            (7, 6) => unrolled_11d::fft_sharded_zero_padded_lut_128_64(shards, shard_len),
            (8, 0) => unrolled_11d::fft_sharded_zero_padded_lut_256_1(shards, shard_len),
            (8, 1) => unrolled_11d::fft_sharded_zero_padded_lut_256_2(shards, shard_len),
            (8, 2) => unrolled_11d::fft_sharded_zero_padded_lut_256_4(shards, shard_len),
            (8, 3) => unrolled_11d::fft_sharded_zero_padded_lut_256_8(shards, shard_len),
            (8, 4) => unrolled_11d::fft_sharded_zero_padded_lut_256_16(shards, shard_len),
            (8, 5) => unrolled_11d::fft_sharded_zero_padded_lut_256_32(shards, shard_len),
            (8, 6) => unrolled_11d::fft_sharded_zero_padded_lut_256_64(shards, shard_len),
            (8, 7) => unrolled_11d::fft_sharded_zero_padded_lut_256_128(shards, shard_len),
            _ => unreachable!("k={k} must be in 1..=8 and log_support must be < k"),
        }
    }

    fn ifft_sharded(
        basis: &impl CantorBasisLut<Gf2p8_11d>,
        shards: &mut [Gf2p8_11d],
        shard_len: usize,
        k: u8,
        beta: Gf2p8_11d,
    ) {
        if beta == Gf2p8::zero() {
            match k {
                0 => {}
                1 => unrolled_11d::ifft_sharded_lut_2(shards, shard_len),
                2 => unrolled_11d::ifft_sharded_lut_4(shards, shard_len),
                3 => unrolled_11d::ifft_sharded_lut_8(shards, shard_len),
                4 => unrolled_11d::ifft_sharded_lut_16(shards, shard_len),
                5 => unrolled_11d::ifft_sharded_lut_32(shards, shard_len),
                6 => unrolled_11d::ifft_sharded_lut_64(shards, shard_len),
                7 => unrolled_11d::ifft_sharded_lut_128(shards, shard_len),
                8 => unrolled_11d::ifft_sharded_lut_256(shards, shard_len),
                _ => unreachable!("k={k} must be in 0..=8"),
            }
        } else {
            match (k, u8::from(beta)) {
                (0, _) => {}
                (1, b) if b == CANTOR_SUBSPACE[1] => {
                    unrolled_11d::ifft_sharded_lut_2_01(shards, shard_len)
                }
                (2, b) if b == CANTOR_SUBSPACE[2] => {
                    unrolled_11d::ifft_sharded_lut_4_d6(shards, shard_len)
                }
                (3, b) if b == CANTOR_SUBSPACE[4] => {
                    unrolled_11d::ifft_sharded_lut_8_98(shards, shard_len)
                }
                (4, b) if b == CANTOR_SUBSPACE[8] => {
                    unrolled_11d::ifft_sharded_lut_16_92(shards, shard_len)
                }
                (5, b) if b == CANTOR_SUBSPACE[16] => {
                    unrolled_11d::ifft_sharded_lut_32_56(shards, shard_len)
                }
                (6, b) if b == CANTOR_SUBSPACE[32] => {
                    unrolled_11d::ifft_sharded_lut_64_c8(shards, shard_len)
                }
                (7, b) if b == CANTOR_SUBSPACE[64] => {
                    unrolled_11d::ifft_sharded_lut_128_58(shards, shard_len)
                }
                (8, b) if b == CANTOR_SUBSPACE[128] => {
                    unrolled_11d::ifft_sharded_lut_256_e7(shards, shard_len)
                }
                _ => ifft_sharded(basis, shards, shard_len, k, beta),
            }
        }
    }

    fn scale(src: &[Gf2p8_11d], dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        scale(src, dst, scalar)
    }

    fn scale_in_place(dst: &mut [Gf2p8_11d], scalar: Gf2p8_11d) {
        scale_in_place(dst, scalar)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::poly_11d_lut::CantorBasisLut11d;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn fft_sharded_zero_padded_matches_fft_sharded() {
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
                fft_sharded(&basis, &mut a, SHARD_LEN, k, beta);
                fft_sharded_zero_padded(&basis, &mut b, SHARD_LEN, k, beta, log_support);

                assert_eq!(a, b, "k={k} log_support={log_support}");
            }
        }
    }

    #[test]
    fn unrolled_fft_sharded_zero_padded_matches_fft_sharded() {
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
        fft_sharded(&basis, &mut a, SHARD_LEN, k, Gf2p8_11d::zero());
        unrolled_11d::fft_sharded_zero_padded_lut_32_8(&mut b, SHARD_LEN);
    }
}

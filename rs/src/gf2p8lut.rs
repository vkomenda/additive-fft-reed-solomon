use additive_fft_reed_solomon_gf2p8::{FIELD_SIZE, Gf2p8};

/// Precomputed lookup table group operations.
pub trait Gf2p8Lut: Gf2p8 {
    /// Multiplication by table lookup.
    fn mul_lut(self, other: Self) -> Self;

    /// Multiplicative inverse by table lookup.
    fn inv_lut(self) -> Self;

    /// Helper to multiply a shard by a scalar
    fn scale_shard(self, shard: &mut [u8]) {
        if self == Self::one() {
            return;
        }
        if self == Self::zero() {
            shard.fill(0);
            return;
        }

        for byte in shard.iter_mut() {
            *byte = self.mul_lut(Self::from(*byte)).into();
        }
    }

    /// Precompute the 256-entry multiply table for a fixed scalar once per
    /// butterfly level.
    fn make_mul_lut(self) -> [u8; FIELD_SIZE] {
        std::array::from_fn(|i| self.mul_lut(Self::from(i as u8)).into())
    }

    fn gfni_mul_matrix(self) -> u64;
}

/// Precompted lookup table operations on the Cantor basis subspace.
pub trait CantorBasisLut<G: Gf2p8Lut> {
    fn get_basis_point_lut(&self, i: u8) -> G;

    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly_lut(&self, i: u8, erased_indices: &[u8]) -> G {
        let alpha_i = self.get_subspace_point_lut(i);
        let mut eval: G = G::one();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = self.get_subspace_point_lut(j);
            eval = eval.mul_lut(alpha_i.add(alpha_j));
        }
        eval
    }

    /// Returns the i-th point $s_i(v_i)$ in the basis subspace.
    fn get_subspace_point_lut(&self, i: u8) -> G;

    /// Returns the coefficient mask of the k-th subspace polynomial.
    fn get_subspace_poly_coeff_lut(&self, k: u8) -> u8;

    fn eval_subspace_poly_lut(&self, k: u8, x: G) -> G;

    /// A basis of the algebraic ring $F_{2^m}[x]/(x^{2^m}-x)$ which forms the evaluation space.
    fn compute_evaluation_space_basis_point(&self, i: u8, x: G) -> G {
        let m = 8;
        let mut result: G = G::one();
        let mut s_j_x = x;

        for j in 0..m {
            let b_j = self.eval_subspace_poly_lut(j, x);

            if (i >> j) & 1 == 1 {
                let term = s_j_x.mul_lut(b_j.inv_lut());
                result = result.mul_lut(term);
            }

            // Update s_j_x to s_{j+1}_x by recursion:
            // s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
            s_j_x = s_j_x.mul_lut(s_j_x.add(b_j));
        }

        result
    }
}

/// The Lin-Chung-Han basis
pub trait LchBasisLut<G: Gf2p8Lut>: CantorBasisLut<G> {
    /// Evaluate the i-th LCH basis polynomial at point x.  The default implementation assumes a
    /// Cantor basis in the evaluation domain, which doesn't require scaling terms by a
    /// normalization factor.
    fn eval_lch_basis_poly(&self, i: u8, x: G) -> G {
        let mut result: G = G::one();

        for j in 0u8..8 {
            if (i >> j) & 1 == 1 {
                let s_j_x = self.eval_subspace_poly_lut(j, x);
                result = result.mul(s_j_x);
            }
        }

        result
    }

    /// $\overline{D}_h (x)$
    fn eval_transform_domain_poly(&self, coeffs: &[G], x: G) -> G {
        let mut result: G = G::zero();

        for (i, d) in coeffs.iter().enumerate() {
            let term = d.mul(self.eval_lch_basis_poly(i as u8, x));
            result = result.add(term);
        }

        result
    }
}

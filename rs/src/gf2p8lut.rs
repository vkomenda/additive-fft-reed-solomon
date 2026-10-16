use additive_fft_reed_solomon_gf2p8::{FIELD_SIZE, Gf2p8, Z255};

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

    fn gfni_mul_matrix_lut(self) -> u64;

    /// Low and high nibble multiplication table for NEON `tbl` and AVX2 `pshufb` lookups:
    /// `lo[i] = i * self`, `hi[i] = (i << 4) * self`.
    fn nibble_mul_table_lut(self) -> ([u8; 16], [u8; 16]);
}

/// Precompted lookup table operations on the Cantor basis subspace.
pub trait SubspaceBasisLut<G: Gf2p8Lut> {
    /// Whether every normalised subspace polynomial derivative is 1, allowing to reduce the formal
    /// derivative to an XOR walk. `true` for all Cantor bases.
    const UNIT_DERIVS: bool;

    /// norm(s)_j^2 = norm(s)_j + norm(s)_{j+1} for the normalised subspace polynomials.
    const UNIT_CARRY: bool;

    fn get_basis_point_lut(i: u8) -> G;

    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly_lut(i: u8, erased_indices: &[u8]) -> G {
        let alpha_i = Self::get_subspace_point_lut(i);
        let mut eval: G = G::one();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = Self::get_subspace_point_lut(j);
            eval = eval.mul_lut(alpha_i.add(alpha_j));
        }
        eval
    }

    /// Returns the i-th point $s_i(v_i)$ in the basis subspace.
    fn get_subspace_point_lut(i: u8) -> G;

    fn eval_subspace_poly_lut(k: u8, x: G) -> G;

    /// A basis of the algebraic ring $F_{2^m}[x]/(x^{2^m}-x)$ which forms the evaluation space.
    fn compute_evaluation_space_basis_point(i: u8, x: G) -> G {
        let m = 8;
        let mut result: G = G::one();
        let mut s_j_x = x;

        for j in 0..m {
            let b_j = Self::eval_subspace_poly_lut(j, x);

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

    fn log_walsh_lut() -> &'static [Z255; FIELD_SIZE];

    fn subspace_poly_deriv_lut() -> &'static [G; 8];
}

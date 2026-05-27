use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};
use additive_fft_reed_solomon_gf2p8::Gf2p8;

/// Basis-independent arithmetic on slices as polynomials.
pub trait PolySliceArith<G: Gf2p8>: AsRef<[G]> {
    fn degree(&self) -> Option<usize> {
        self.as_ref()
            .iter()
            .enumerate()
            .rev()
            .find(|(_, c)| **c != G::zero())
            .map(|(i, _)| i)
    }

    fn leading_coeff(&self) -> G {
        let coeffs = self.as_ref();
        coeffs
            .get(self.degree().unwrap_or(0))
            .copied()
            .unwrap_or(G::zero())
    }

    // /// Polynomial multiplication in the monomial basis.
    // fn poly_mul_mon(&self, b: &[G]) -> [G; N] {
    //     let mut res = [G::zero(); N];

    //     for (i, &ai) in self.iter().enumerate() {
    //         if ai == G::zero() {
    //             continue;
    //         }
    //         for (j, &bj) in b.iter().enumerate() {
    //             res[i + j] = res[i + j].add(ai.mul(bj));
    //         }
    //     }
    //     res
    // }
}

impl<G: Gf2p8Lut, T: AsRef<[G]>> PolySliceArith<G> for T {}

/// Basis-independent arithmetic on mutable slices as polynomials.
pub trait PolyMutSliceArith<G: Gf2p8>: AsMut<[G]> {
    fn poly_add_in_place(&mut self, b: &[G]) {
        for (ai, bi) in self.as_mut().iter_mut().zip(b.iter()) {
            *ai = ai.add(*bi);
        }
    }
}

impl<G: Gf2p8Lut> PolyMutSliceArith<G> for [G] {}
impl<G: Gf2p8Lut, T: AsMut<[G]>> PolyMutSliceArith<G> for T {}

/// Basis-independent arithmetic on arrays as polynomials.
pub mod poly {
    use crate::gf2p8lut::Gf2p8Lut;
    use additive_fft_reed_solomon_gf2p8::Gf2p8;

    /// Addition of polynomials. Works the same for both the monomial and X bases.
    pub fn add<G: Gf2p8, const N: usize>(a: &[G; N], b: &[G; N]) -> [G; N] {
        let mut res = *a;

        for (ai, bi) in res.iter_mut().zip(b.iter()) {
            *ai = ai.add(*bi);
        }

        res
    }

    /// Split p at 2^k: lo = p[0..2^k), hi = p[2^k..) shifted down.
    pub fn split_at<G: Gf2p8Lut, const N: usize>(p: &[G; N], k: u8) -> ([G; N], [G; N]) {
        let pivot = 1usize << k;
        let mut lo = [G::zero(); N];
        let mut hi = [G::zero(); N];
        lo[..pivot].copy_from_slice(&p[..pivot]);
        hi[..N - pivot].copy_from_slice(&p[pivot..]);
        (lo, hi)
    }

    /// Multiply p by s_k = X_{2^k} by shifting coefficients up by 2^k.
    ///
    /// Valid only when every non-zero coefficient of p sits at an index
    /// where bit k is 0 — always satisfied by the HGCD invariants.
    pub fn shift_up<G: Gf2p8Lut, const N: usize>(p: &[G; N], k: u8) -> [G; N] {
        let shift = 1usize << k;
        let mut out = [G::zero(); N];
        out[shift..].copy_from_slice(&p[..N - shift]);
        out
    }

    /// Derivative in the LNH basis based on Eq 82
    pub fn deriv_lnh<G: Gf2p8Lut, const N: usize>(coeffs: &[G; N]) -> [G; N] {
        let mut res = [G::zero(); N];

        let n = coeffs.len();
        if n <= 1 {
            return res;
        }

        res[..n].copy_from_slice(coeffs);

        let m = n.trailing_zeros() as usize;

        for j in 1..=m {
            let half = 1 << (j - 1);
            let step = 1 << j;
            for start in (0..n).step_by(step) {
                for i in 0..half {
                    // Eq 82 simplified: The derivative of the upper half
                    // interacts with the subspace derivative.
                    res[start + i] = res[start + i].add(res[start + i + half]);
                }
            }
        }

        res
    }
}

/// Basis-dependent polynomial arithmetic.
pub trait CantorBasisPolySliceArith<G: Gf2p8Lut>: CantorBasisLut<G> {
    /// Algorithm 1 in LNH paper.
    ///
    /// The input coefficients `coeff` represent a polynomial in the basis X. That is,
    /// a polynomial $\Sum_{i=0}^{2^k - 1} d_i X_i(x)$, for $d_i$ in `coeff`.
    ///
    /// The function outputs, in `coeff`, the evaluations of the input polynomial at points
    /// $\omega_i + \beta$ where $\omega_i$ are the points of the subspace $V_k$,
    /// for $0 \le i < 2^k$.
    fn fft_scalar(&self, coeffs: &mut [G], k: u8, beta: G) {
        if k == 0 {
            return;
        }

        let half = 1 << (k - 1);

        // Fetch the twiddle factor T = s_{k-1}(beta)
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        // Butterfly stage (line 3-6)
        for i in 0..half {
            let d_i = coeffs[i];
            let d_i_half = coeffs[i + half];

            // g_i_0 = d_i + T * d_{i+half}
            let g_i_0 = d_i.add(twiddle.mul(d_i_half));

            // g_i_1 = g_i_0 + d_{i+half}
            // This is equivalent to d_i + (T + 1) * d_{i+half}
            let g_i_1 = g_i_0.add(d_i_half);

            coeffs[i] = g_i_0;
            coeffs[i + half] = g_i_1;
        }

        // Recursive calls (line 7-8)
        // Left branch: FFT(g_0, k-1, beta)
        self.fft_scalar(&mut coeffs[..half], k - 1, beta);

        // Right branch: FFT(g_1, k-1, v_{k-1} + beta)
        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.fft_scalar(&mut coeffs[half..], k - 1, next_beta);
    }

    /// Algorithm 2 in LNH paper.
    fn ifft_scalar(&self, evals: &mut [G], k: u8, beta: G) {
        if k == 0 {
            return;
        }

        let half = 1 << (k - 1);

        self.ifft_scalar(&mut evals[..half], k - 1, beta);

        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.ifft_scalar(&mut evals[half..], k - 1, next_beta);

        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let g_i_0 = evals[i];
            let g_i_1 = evals[i + half];

            let d_i_half = g_i_0.add(g_i_1);
            let d_i = g_i_0.add(twiddle.mul(d_i_half));

            evals[i] = d_i;
            evals[i + half] = d_i_half;
        }
    }
}

impl<G: Gf2p8Lut, T: CantorBasisLut<G>> CantorBasisPolySliceArith<G> for T {}

pub trait CantorBasisPolyArith<G: Gf2p8Lut, const N: usize>: CantorBasisPolySliceArith<G> {
    /// Polynomial multiplication in the basis X.
    ///
    /// As defined in Appendix A of the LNH paper.
    fn poly_mul_lnh(&self, a: &[G; N], b: &[G; N]) -> [G; N] {
        let deg_a = if let Some(deg) = a.degree() {
            deg
        } else {
            return [G::zero(); N];
        };

        let deg_b = if let Some(deg) = b.degree() {
            deg
        } else {
            return [G::zero(); N];
        };

        // Determine the smallest power-of-2 size n >= deg_a + deg_b + 1
        let n = (deg_a + deg_b + 1).next_power_of_two();
        let n_log = n.trailing_zeros() as u8;

        let mut va = *a;
        let mut vb = *b;

        // Transform to evaluation space
        self.fft_scalar(&mut va[..n], n_log, G::zero());
        self.fft_scalar(&mut vb[..n], n_log, G::zero());

        // Pointwise multiplication
        for i in 0..n {
            va[i] = va[i].mul(vb[i]);
        }

        // Transform back to coefficient space
        self.ifft_scalar(&mut va[..n], n_log, G::zero());

        va
    }

    fn poly_div_lnh(&self, a: &[G; N], b: &[G; N]) -> Option<([G; N], [G; N])> {
        let b_deg = b.degree()?;

        let mut r = *a;
        let mut q = [G::zero(); N];

        let b_lc_inv = b[b_deg].inv_lut();

        // Standard synthetic division, but with basis-aware multiplication
        while let Some(r_deg) = r.degree() {
            if r_deg < b_deg {
                break;
            }

            let deg_diff = r_deg - b_deg;

            // Calculate leading coefficient of the quotient
            let factor = r[r_deg].mul(b_lc_inv);
            q[deg_diff] = q[deg_diff].add(factor);

            // Compute what to subtract: factor * X_{deg_diff} * b(x)
            let mut term_to_sub = [G::zero(); N];
            term_to_sub[deg_diff] = factor;

            // Basis-aware multiplication
            let product = self.poly_mul_lnh(&term_to_sub, b);

            // Update the remainder
            r.poly_add_in_place(&product);
        }

        Some((q, r))
    }

    /// Division in the monomial basis.
    /// s_t(x) = q_t(x) * s(x) + r_t(x)
    fn poly_div_mon(&self, a: &[G], b: &[G]) -> Option<([G; N], [G; N])> {
        let a_deg = if let Some(deg) = a.degree() {
            deg
        } else {
            return Some(([G::zero(); N], [G::zero(); N]));
        };

        let mut r = [G::zero(); N];
        r.copy_from_slice(a);
        let mut q = [G::zero(); N];
        let b_deg = b.degree()?;
        let b_lc_inv = b[b_deg].inv_lut();

        for i in (b_deg..a_deg + 1).rev() {
            let factor = r[i].mul(b_lc_inv);
            q[i - b_deg] = factor;
            for j in 0..=b_deg {
                r[i - b_deg + j] = r[i - b_deg + j].add(factor.mul(b[j]));
            }
        }
        Some((q, r))
    }

    // 2×2 matrix helpers (row-major: [m00, m01, m10, m11])
    fn mat_vec_lnh(&self, m: &[[G; N]; 4], v0: &[G; N], v1: &[G; N]) -> ([G; N], [G; N]) {
        (
            poly::add(&self.poly_mul_lnh(&m[0], v0), &self.poly_mul_lnh(&m[1], v1)),
            poly::add(&self.poly_mul_lnh(&m[2], v0), &self.poly_mul_lnh(&m[3], v1)),
        )
    }

    fn mat_mul_lnh(&self, a: &[[G; N]; 4], b: &[[G; N]; 4]) -> [[G; N]; 4] {
        [
            poly::add(
                &self.poly_mul_lnh(&a[0], &b[0]),
                &self.poly_mul_lnh(&a[1], &b[2]),
            ),
            poly::add(
                &self.poly_mul_lnh(&a[0], &b[1]),
                &self.poly_mul_lnh(&a[1], &b[3]),
            ),
            poly::add(
                &self.poly_mul_lnh(&a[2], &b[0]),
                &self.poly_mul_lnh(&a[3], &b[2]),
            ),
            poly::add(
                &self.poly_mul_lnh(&a[2], &b[1]),
                &self.poly_mul_lnh(&a[3], &b[3]),
            ),
        ]
    }
}

impl<G: Gf2p8Lut, T: CantorBasisLut<G>, const N: usize> CantorBasisPolyArith<G, N> for T {}

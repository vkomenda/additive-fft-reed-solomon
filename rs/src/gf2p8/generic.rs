use core::mem::MaybeUninit;

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
    fn make_mul_lut(self) -> [Self; 256] {
        std::array::from_fn(|i| self.mul_lut(Self::from(i as u8)))
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

    fn fft_sharded(&self, shards: &mut [&mut [G]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);
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

        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.fft_sharded(&mut shards[..half], k - 1, beta);
        self.fft_sharded(&mut shards[half..], k - 1, next_beta);
    }

    fn ifft_sharded(&self, shards: &mut [&mut [G]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);

        let next_beta = beta.add(self.get_basis_point_lut(k - 1));
        self.ifft_sharded(&mut shards[..half], k - 1, beta);
        self.ifft_sharded(&mut shards[half..], k - 1, next_beta);

        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);
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

    /// Fused multiply-add scaled to a subspace of size 2^k.
    ///
    /// out ^= (coeff * X_deg) * rhs (mod s_k(x))
    fn poly_fused_mul_add<I, O>(&self, mut out: O, rhs: I, coeff: G, deg: usize, k: u8)
    where
        I: AsRef<[G]>,
        O: AsMut<[G]>,
    {
        let n = 1 << k;

        // deg must be within the subspace, and input lengths must not exceed n.
        debug_assert!(deg < n);

        let rhs_ref = rhs.as_ref();
        let rhs_len = rhs_ref.len().min(n);

        // Use 256-sized buffers for the FFT implementation but process only the first n elements.
        let mut tmp_q = [G::zero(); FIELD_SIZE];
        let mut tmp_rhs = [G::zero(); FIELD_SIZE];

        // Prepare LCH coefficients
        tmp_q[deg] = coeff;
        tmp_rhs[..rhs_len].copy_from_slice(&rhs_ref[..rhs_len]);

        // FFT to the evaluation space of size n
        // beta = 0 because we are multiplying in the standard subspace V_k
        self.fft_scalar(&mut tmp_q, k, G::zero());
        self.fft_scalar(&mut tmp_rhs, k, G::zero());

        // Pointwise multiplication in the evaluation space
        for i in 0..n {
            tmp_q[i] = tmp_q[i].mul(tmp_rhs[i]);
        }

        // IFFT back to the coefficient space in the basis X
        self.ifft_scalar(&mut tmp_q, k, G::zero());

        // Accumulate into the output
        // out might be smaller than n, so we zip carefully
        let out_ref = out.as_mut();
        let limit = out_ref.len().min(n);
        for (o, t) in out_ref[..limit].iter_mut().zip(tmp_q[..limit].iter()) {
            *o = o.add(*t);
        }
    }

    fn solve_key_equation_eea(
        &self,
        syndrome: &[G; FIELD_SIZE],
        t_log: u8,
    ) -> Option<([G; FIELD_SIZE], [G; FIELD_SIZE])> {
        let mut st = [G::zero(); FIELD_SIZE];
        self.init_subspace_poly_coeffs(&mut st, t_log);
        let (qt, rt) = self.poly_div_lnh(&st, syndrome)?;
        let (u1, v1, _z1) = self.eea(syndrome, &rt, t_log);
        let lambda = self.poly_add(&u1, &self.poly_mul_lnh(&v1, &qt));
        Some((v1, lambda))
    }

    /// Extended Euclidean Algorithm.
    fn eea(
        &self,
        a: &[G], // syndrome
        b: &[G], // remainder r_t(x) from the initial division
        t_log: u8,
    ) -> ([G; FIELD_SIZE], [G; FIELD_SIZE], [G; FIELD_SIZE]) {
        let t_parity = 1 << t_log;
        let target_deg = t_parity / 2; // Stop when deg(z) < T/2

        let mut z0 = [G::zero(); FIELD_SIZE];
        z0.copy_from_slice(a);
        let mut z1 = [G::zero(); FIELD_SIZE];
        z1.copy_from_slice(b);

        let mut u0 = [G::zero(); FIELD_SIZE];
        u0[0] = G::one();
        let mut u1 = [G::zero(); FIELD_SIZE];

        let mut v0 = [G::zero(); FIELD_SIZE];
        let mut v1 = [G::zero(); FIELD_SIZE];
        v1[0] = G::one();

        while z1.degree().is_some_and(|d| d >= target_deg) {
            let (q, remainder) = self
                .poly_div_lnh(&z0, &z1)
                .expect("z1 is non-zero thanks to while condition");

            // Update r: r = r0 - q * r1
            z0 = z1;
            z1 = remainder;

            // Update u: u = u0 - q * u1
            let q_u1 = self.poly_mul_lnh(&q, &u1);
            let next_u = self.poly_add(&u0, &q_u1);
            u0 = u1;
            u1 = next_u;

            // Update v: v = v0 - q * v1
            let q_v1 = self.poly_mul_lnh(&q, &v1);
            let next_v = self.poly_add(&v0, &q_v1);
            v0 = v1;
            v1 = next_v;
        }

        // LNH page 9:
        // u1 = u auxiliary, v1 = v auxiliary, z1 = z error evaluator
        (u1, v1, z1)
    }

    fn init_subspace_poly_coeffs(&self, st: &mut [G], t_log: u8) {
        st[1 << t_log] = G::one(); // Coefficient of x stays 1.
    }

    /// Division in the monomial basis.
    /// s_t(x) = q_t(x) * s(x) + r_t(x)
    fn poly_div_mon(&self, a: &[G], b: &[G]) -> Option<([G; FIELD_SIZE], [G; FIELD_SIZE])> {
        let a_deg = if let Some(deg) = a.degree() {
            deg
        } else {
            return Some(([G::zero(); FIELD_SIZE], [G::zero(); FIELD_SIZE]));
        };

        let mut r = [G::zero(); FIELD_SIZE];
        r.copy_from_slice(a);
        let mut q = [G::zero(); FIELD_SIZE];
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

    /// Polynomial multiplication in the monomial basis.
    fn poly_mul_mon(&self, a: &[G], b: &[G]) -> [G; FIELD_SIZE] {
        let mut res = [G::zero(); FIELD_SIZE];

        for (i, &ai) in a.iter().enumerate() {
            if ai == G::zero() {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                res[i + j] = res[i + j].add(ai.mul(bj));
            }
        }
        res
    }

    /// Addition of polynomials. Works the same for both the monomial and X bases.
    fn poly_add(&self, a: &[G], b: &[G]) -> [G; FIELD_SIZE] {
        let mut res = [G::zero(); FIELD_SIZE];
        res.copy_from_slice(a);

        for (ai, bi) in res.iter_mut().zip(b.iter()) {
            *ai = ai.add(*bi);
        }

        res
    }

    fn poly_add_inplace(&self, a: &mut [G], b: &[G]) {
        for (ai, bi) in a.iter_mut().zip(b.iter()) {
            *ai = ai.add(*bi);
        }
    }

    /// Polynomial multiplication in the basis X.
    ///
    /// As defined in Appendix A of the LNH paper.
    fn poly_mul_lnh(&self, a: &[G; FIELD_SIZE], b: &[G; FIELD_SIZE]) -> [G; FIELD_SIZE] {
        let deg_a = if let Some(deg) = a.degree() {
            deg
        } else {
            return [G::zero(); FIELD_SIZE];
        };

        let deg_b = if let Some(deg) = b.degree() {
            deg
        } else {
            return [G::zero(); FIELD_SIZE];
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

    fn poly_div_lnh(
        &self,
        a: &[G; FIELD_SIZE],
        b: &[G; FIELD_SIZE],
    ) -> Option<([G; FIELD_SIZE], [G; FIELD_SIZE])> {
        let b_deg = b.degree()?;

        let mut r = *a;
        let mut q = [G::zero(); FIELD_SIZE];

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
            let mut term_to_sub = [G::zero(); FIELD_SIZE];
            term_to_sub[deg_diff] = factor;

            // Basis-aware multiplication
            let product = self.poly_mul_lnh(&term_to_sub, b);

            // Update the remainder
            r = self.poly_add(&r, &product);
        }

        Some((q, r))
    }

    /// Split p at 2^k: lo = p[0..2^k), hi = p[2^k..) shifted down.
    fn poly_split_at(&self, p: &[G; FIELD_SIZE], k: u8) -> ([G; FIELD_SIZE], [G; FIELD_SIZE]) {
        let pivot = 1usize << k;
        let mut lo = [G::zero(); FIELD_SIZE];
        let mut hi = [G::zero(); FIELD_SIZE];
        lo[..pivot].copy_from_slice(&p[..pivot]);
        hi[..FIELD_SIZE - pivot].copy_from_slice(&p[pivot..]);
        (lo, hi)
    }

    /// Multiply p by s_k = X_{2^k} by shifting coefficients up by 2^k.
    ///
    /// Valid only when every non-zero coefficient of p sits at an index
    /// where bit k is 0 — always satisfied by the HGCD invariants.
    fn poly_shift_up(&self, p: &[G; FIELD_SIZE], k: u8) -> [G; FIELD_SIZE] {
        let shift = 1usize << k;
        let mut out = [G::zero(); FIELD_SIZE];
        out[shift..].copy_from_slice(&p[..FIELD_SIZE - shift]);
        out
    }

    /// Step-8 decomposition: given p and the current HGCD level g, return
    /// $(p_ll, p_m)$ such that $p = p_ll + s_{g-2}(x) · p_m$, where
    /// $$
    ///   p_ll = p[0 .. 2^{g-2})
    ///   p_m  = p_lh + (s_{g-2} + s_{g-2}(v_{g-2})) · p_h
    /// $$
    /// with $p_lh = p[2^{g-2} .. 2^{g-1})$ and $p_h = p[2^{g-1} .. 2^{g-1}+2^{g-2})$.
    ///
    /// Derivation: $s_{g-2}^2 = s_{g-1} + c·s_{g-2}$  (Cantor basis recursion),
    /// so $s_{g-2}·p_m$ expands back to $s_{g-2}·p_lh + s_{g-1}·p_h$, recovering p.
    fn poly_hgcd_middle(&self, p: &[G; FIELD_SIZE], g: u8) -> ([G; FIELD_SIZE], [G; FIELD_SIZE]) {
        debug_assert!(g >= 2);
        let q = 1usize << (g - 2); // 2^{g-2}
        let h = 1usize << (g - 1); // 2^{g-1}
        let c = self.eval_subspace_poly_lut(g - 2, self.get_subspace_point_lut(1u8 << (g - 2)));

        let mut p_ll = [G::zero(); FIELD_SIZE];
        let mut p_m = [G::zero(); FIELD_SIZE];
        p_ll[..q].copy_from_slice(&p[..q]);

        for i in 0..q {
            // Lower block of p_m: derived from p[q..h] and p[h..h+q]
            p_m[i] = p[i + q].add(c.mul_lut(p[i + h]));
            p_m[i + q] = p[i + h];
            // Upper block of p_m: derived from p[h+q..h+2q] and p[2h..2h+q]
            // (these indices are 0 for deg(p) < 2^g but non-zero when deg(p) = 2^{g-1}..2^g-1)
            p_m[i + h] = p[i + h + q].add(c.mul_lut(p[i + 2 * h]));
            p_m[i + h + q] = p[i + 2 * h];
        }

        (p_ll, p_m)
    }

    // 2×2 matrix helpers (row-major: [m00, m01, m10, m11])
    fn mat_vec_lnh(
        &self,
        m: &[[G; FIELD_SIZE]; 4],
        v0: &[G; FIELD_SIZE],
        v1: &[G; FIELD_SIZE],
    ) -> ([G; FIELD_SIZE], [G; FIELD_SIZE]) {
        (
            self.poly_add(&self.poly_mul_lnh(&m[0], v0), &self.poly_mul_lnh(&m[1], v1)),
            self.poly_add(&self.poly_mul_lnh(&m[2], v0), &self.poly_mul_lnh(&m[3], v1)),
        )
    }

    fn mat_mul_lnh(
        &self,
        a: &[[G; FIELD_SIZE]; 4],
        b: &[[G; FIELD_SIZE]; 4],
    ) -> [[G; FIELD_SIZE]; 4] {
        [
            self.poly_add(
                &self.poly_mul_lnh(&a[0], &b[0]),
                &self.poly_mul_lnh(&a[1], &b[2]),
            ),
            self.poly_add(
                &self.poly_mul_lnh(&a[0], &b[1]),
                &self.poly_mul_lnh(&a[1], &b[3]),
            ),
            self.poly_add(
                &self.poly_mul_lnh(&a[2], &b[0]),
                &self.poly_mul_lnh(&a[3], &b[2]),
            ),
            self.poly_add(
                &self.poly_mul_lnh(&a[2], &b[1]),
                &self.poly_mul_lnh(&a[3], &b[3]),
            ),
        ]
    }

    /// Half-GCD algorithm (Algorithm 5, LNH).
    ///
    /// Preconditions: $deg(b) \le deg(a),  2^{g-1} \le deg(a) < 2^g$.
    ///
    /// Returns (z0, z1, M) where M = [m00, m01, m10, m11] (row-major) satisfies
    ///   $[z_0, z_1]^T = M · [a, b]^T$,
    ///   $deg(z_0) \ge 2^{g-1}, deg(z_1) < 2^{g-1}$.
    fn hgcd(
        &self,
        a: &[G; FIELD_SIZE],
        b: &[G; FIELD_SIZE],
        g: u8,
    ) -> ([G; FIELD_SIZE], [G; FIELD_SIZE], [[G; FIELD_SIZE]; 4]) {
        let zero = [G::zero(); FIELD_SIZE];
        let one = {
            let mut p = zero;
            p[0] = G::one();
            p
        };
        let half = 1usize << (g - 1);

        // Base case (Algorithm 5 lines 1-2)
        // deg(b) < 2^{g-1}: Z = [a, b], M = I.
        if b.degree().is_none_or(|d| d < half) {
            return (*a, *b, [one, zero, zero, one]);
        }

        // Step 3: split at 2^{g-1}, recurse on high halves
        let (a_l, a_h) = self.poly_split_at(a, g - 1);
        let (b_l, b_h) = self.poly_split_at(b, g - 1);
        let (z_h0, z_h1, m_h) = self.hgcd(&a_h, &b_h, g - 1);

        // Step 4: z_M = Z_H · s_{g-1} + M_H · [a_L, b_L]^T
        // Equivalently M_H · (s_{g-1}·[a_H,b_H] + [a_L,b_L]) = M_H · [a, b].
        let (mv0, mv1) = self.mat_vec_lnh(&m_h, &a_l, &b_l);
        let z_m0 = self.poly_add(&self.poly_shift_up(&z_h0, g - 1), &mv0);
        let z_m1 = self.poly_add(&self.poly_shift_up(&z_h1, g - 1), &mv1);

        // Step 5: early return if deg(z_M1) < 2^{g-1} (lines 5-6)
        if z_m1.degree().is_none_or(|d| d < half) {
            return (z_m0, z_m1, m_h);
        }

        // Step 7: divide z_M0 by z_M1
        let (q_m, r_m) = self
            .poly_div_lnh(&z_m0, &z_m1)
            .expect("z_m1 non-zero: guaranteed by the HGCD degree invariant");

        // Step 8: decompose z_M1 and r_M into LL and M parts
        let (z_m1_ll, z_m1_m) = self.poly_hgcd_middle(&z_m1, g);
        let (r_m_ll, r_m_m) = self.poly_hgcd_middle(&r_m, g);

        // Step 9: second recursive call
        let (y_m0, y_m1, m_m) = self.hgcd(&z_m1_m, &r_m_m, g - 1);

        // Step 10
        let swap = [zero, one, one, q_m];
        let m_r = self.mat_mul_lnh(&m_m, &self.mat_mul_lnh(&swap, &m_h));

        // Y_M · s_{g-2}: y_m1 is safe to shift (degree < 2^{g-2}, bit g-2 always 0),
        // but y_m0 can have coefficients at index 2^{g-2} (bit g-2 set), so it
        // requires a proper polynomial multiplication rather than a plain index shift.
        let sg_minus_2 = {
            let mut p = [G::zero(); FIELD_SIZE];
            p[1 << (g - 2)] = G::one(); // s_{g-2} = X_{2^{g-2}} in basis X
            p
        };
        let (mv0, mv1) = self.mat_vec_lnh(&m_m, &z_m1_ll, &r_m_ll);
        let z_r0 = self.poly_add(&self.poly_mul_lnh(&y_m0, &sg_minus_2), &mv0);
        let z_r1 = self.poly_add(&self.poly_shift_up(&y_m1, g - 2), &mv1); // y_m1 safe

        (z_r0, z_r1, m_r)
    }

    /// This is functionally equivalent to `solve_key_equation_eea` and is what the LNH paper
    /// has.
    fn solve_key_equation_hgcd(
        &self,
        syndrome: &[G; FIELD_SIZE],
        t_log: u8,
    ) -> Option<([G; FIELD_SIZE], [G; FIELD_SIZE])> {
        let mut st = [G::zero(); FIELD_SIZE];
        self.init_subspace_poly_coeffs(&mut st, t_log);

        // s_t = q_t · s + r_t
        let (q_t, r_t) = self.poly_div_lnh(&st, syndrome)?;

        // HGCD(s, r_t, t_log) => M = [m00,m01,m10,m11] with
        //   z1 = m10·s + m11·r_t
        //      = m11·s_t + (m10 + m11·q_t)·s     [since r_t = s_t − q_t·s]
        // so the error locator is λ = m10 + m11·q_t  (eq. 79, GF(2): - is +)
        // and the error evaluator is v1 = m11
        let (_z0, _z1, m) = self.hgcd(syndrome, &r_t, t_log);

        let u1 = m[2]; // m10
        let v1 = m[3]; // m11
        let lambda = self.poly_add(&u1, &self.poly_mul_lnh(&v1, &q_t));

        Some((v1, lambda))
    }
}

/// Derivative in the LNH basis based on Eq 82
pub fn deriv_poly_lnh<G: Gf2p8>(coeffs: &[G; FIELD_SIZE]) -> [G; FIELD_SIZE] {
    let mut res = [G::zero(); FIELD_SIZE];

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

pub trait Codec<G: Gf2p8Lut>: CantorBasisLut<G> + LchBasisLut<G> {
    fn encode_systematic_scalar(&self, message: &[G], parity: &mut [G]) {
        let t_parity = parity.len();
        let t_log = t_parity.trailing_zeros() as u8;
        let k_msg = message.len();

        // Compute parity image (v0') using LNH Eq 68
        parity.fill(G::zero());
        let mut workspace = [G::zero(); FIELD_SIZE / 2];

        for i in 0..k_msg / t_parity {
            workspace[..t_parity].copy_from_slice(&message[i * t_parity..(i + 1) * t_parity]);
            let omega = self.get_subspace_point_lut(((i + 1) * t_parity) as u8);
            self.ifft_scalar(&mut workspace[..t_parity], t_log, omega);
            self.poly_add_inplace(parity, &workspace[..t_parity]);
        }

        // Compute parity (v0)
        self.fft_scalar(parity, t_log, G::zero());
    }

    fn encode_systematic_sharded(&self, message: &[&[G]], parity: &mut [&mut [G]]) {
        let t_parity = parity.len();
        let t_log = t_parity.trailing_zeros() as u8;
        let shard_len = parity[0].len();

        for shard in parity.iter_mut() {
            shard.fill(G::zero());
        }

        // TODO: accept the workspace or the backing store as a fn argument
        let mut backing = vec![G::zero(); t_parity * shard_len];

        // Fixed-size header array on the stack: FIELD_SIZE/2 × 16 bytes = 2 KB,
        // regardless of shard size or t_parity.
        let mut hdrs: [MaybeUninit<&mut [G]>; FIELD_SIZE / 2] =
            unsafe { MaybeUninit::uninit().assume_init() };

        for (i, chunk) in backing.chunks_mut(shard_len).enumerate() {
            hdrs[i].write(chunk);
        }

        let workspace: &mut [&mut [G]] =
            unsafe { std::slice::from_raw_parts_mut(hdrs.as_mut_ptr() as *mut &mut [G], t_parity) };
        for i in 0..message.len() / t_parity {
            for j in 0..t_parity {
                workspace[j].copy_from_slice(message[i * t_parity + j]);
            }
            let omega = self.get_subspace_point_lut(((i + 1) * t_parity) as u8);
            self.ifft_sharded(workspace, t_log, omega);
            for j in 0..t_parity {
                G::shard_add(parity[j], workspace[j]);
            }
        }

        self.fft_sharded(parity, t_log, G::zero());
    }

    /// Syndrome calculation (scalar).
    /// Computes s = sum_{i=0}^{n/T-1} IFFT(r_i, t, omega_{i*T})
    fn compute_syndrome_scalar(
        &self,
        received: &[G], // Size n (e.g., 256)
        t_log: u8,      // log_2(T)
    ) -> [G; FIELD_SIZE] {
        let t_parity = 1 << t_log;
        // Reserve the extra bit for the key equation solver (EEA requirement).
        let mut syndrome = [G::zero(); FIELD_SIZE];
        let mut workspace = [G::zero(); FIELD_SIZE];

        for (i, chunk) in received.chunks(t_parity).enumerate() {
            // beta corresponds to the starting point of the i-th chunk: omega_{i*T}
            let omega_idx = (i * t_parity) as u8;
            let beta = self.get_subspace_point_lut(omega_idx);

            // Copy received chunk into workspace
            // Pad with zeros if the last chunk is partial (Eq 63)
            workspace[..t_parity].fill(G::zero());
            for (w, &r) in workspace[..t_parity].iter_mut().zip(chunk.iter()) {
                *w = r;
            }

            // Perform the partial IFFT (Algorithm 2)
            // This moves the chunk from evaluation space to basis X coefficients
            self.ifft_scalar(&mut workspace[..t_parity], t_log, beta);

            // Accumulate into the syndrome buffer
            for (s, &w) in syndrome
                .iter_mut()
                .take(t_parity)
                .zip(workspace[..t_parity].iter())
            {
                *s = s.add(w);
            }
        }

        syndrome
    }

    /// Recomputes data from parity when $n = 2T$.
    fn recompute_data_from_parity(&self, received: &mut [G]) {
        let n = received.len();
        let n_log = n.trailing_zeros() as u8;
        let t_log = n_log - 1;
        let t_parity = 1 << t_log;

        let mut workspace = [G::zero(); FIELD_SIZE];

        workspace[..t_parity].copy_from_slice(&received[..t_parity]);
        self.ifft_scalar(&mut workspace, t_log, G::zero());
        self.fft_scalar(
            &mut workspace,
            t_log,
            self.get_subspace_point_lut(t_parity as u8),
        );

        received[t_parity..].copy_from_slice(&workspace[..t_parity]);
    }

    /// Recomputes data shards from parity shards when $n = 2T$.
    fn recompute_data_from_parity_sharded(&self, received: &mut [&mut [G]]) {
        let n = received.len();
        let n_log = n.trailing_zeros() as u8;
        let t_log = n_log - 1;
        let t_parity = 1 << t_log;
        let shard_len = received[0].len();

        let mut backing = vec![G::zero(); t_parity * shard_len];
        let mut hdrs: [MaybeUninit<&mut [G]>; FIELD_SIZE / 2] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (i, chunk) in backing.chunks_mut(shard_len).enumerate() {
            hdrs[i].write(chunk);
        }
        let workspace: &mut [&mut [G]] =
            unsafe { std::slice::from_raw_parts_mut(hdrs.as_mut_ptr() as *mut &mut [G], t_parity) };

        // Copy parity shards into workspace, then recover polynomial coefficients
        for i in 0..t_parity {
            workspace[i].copy_from_slice(received[i]);
        }
        self.ifft_sharded(workspace, t_log, G::zero());
        let omega = self.get_subspace_point_lut(t_parity as u8);
        self.fft_sharded(workspace, t_log, omega);

        for i in 0..t_parity {
            // G::shard_add(&mut received[i + t_parity], &workspace[i]);
            received[i + t_parity].copy_from_slice(workspace[i]);
        }
    }

    /// Systematic scalar RS decoder.
    ///
    /// # Arguments
    /// * `received` - Received bytes, including both parity and message. Contains the decoding upon return.
    /// * `k_msg`    - Message length such that $T = 256 - k$ is the number of parity shards, $T$ is a power of 2.
    ///
    /// # Returns
    /// * `true`     - if decoding succeeded
    /// * `false`    - if decoding failed
    fn decode_systematic_scalar(&self, received: &mut [G], k_msg: usize) -> bool {
        let n = received.len();
        let t_parity = n - k_msg;
        if t_parity == 0 {
            return true;
        }
        let t_log = t_parity.trailing_zeros() as u8;

        // Step 1: syndrome
        let syndrome = self.compute_syndrome_scalar(received, t_log);
        if syndrome.iter().take(t_parity).all(|&c| c == G::zero()) {
            // println!("Syndrome is zero.");
            return true;
        }
        // println!(
        //     "syndrome[..t_parity] = {:?}",
        //     &syndrome[..t_parity]
        //         .iter()
        //         .map(|&x| x.into())
        //         .collect::<Vec<_>>()
        // );

        // Step 2: key equation
        let (v1, lambda) = match self.solve_key_equation_hgcd(&syndrome, t_log) {
            Some(pair) => pair,
            None => {
                // TODO: error type enum
                // println!("Key equation has no solution.");
                return false;
            }
        };

        let deg_lambda = match lambda.degree() {
            Some(d) => d,
            None => {
                // println!("Zero locator - no errors.");
                return true;
            }
        };

        // Step 3: root-finding - one T-point FFT per chunk
        let mut error_indices: Vec<usize> = Vec::with_capacity(deg_lambda);

        'root: for chunk in 0..(n / t_parity) {
            let beta = self.get_subspace_point_lut((chunk * t_parity) as u8);
            let mut evals = lambda;
            self.fft_scalar(&mut evals[..t_parity], t_log, beta);

            for offset in 0..t_parity {
                if evals[offset] == G::zero() {
                    error_indices.push(chunk * t_parity + offset);
                    if error_indices.len() == deg_lambda {
                        break 'root; // found all roots, stop scanning
                    }
                }
            }
        }

        if error_indices.len() < deg_lambda {
            // println!(
            //     "Too few roots. deg_lambda={deg_lambda}, error_indices={error_indices:?}, syndrome={:?}, v1={:?}, lambda={:?}",
            //     syndrome.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
            //     v1.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
            //     lambda.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
            // );
            return false; // too few roots, uncorrectable
        }

        // Step 4: error values - same per-chunk FFT structure as step 3
        let lambdap = deriv_poly_lnh(&lambda);

        for chunk in 0..(n / t_parity) {
            let chunk_errors: Vec<(usize, usize)> = error_indices
                .iter()
                .filter(|&&g| g / t_parity == chunk)
                .map(|&g| (g, g % t_parity))
                .collect();

            if chunk_errors.is_empty() {
                continue;
            }

            let beta = self.get_subspace_point_lut((chunk * t_parity) as u8);

            let mut q_evals = v1;
            let mut lp_evals = lambdap;
            self.fft_scalar(&mut q_evals[..t_parity], t_log, beta);
            self.fft_scalar(&mut lp_evals[..t_parity], t_log, beta);

            for (global, offset) in chunk_errors {
                let lp = lp_evals[offset];
                if lp == G::zero() {
                    continue;
                }
                let error_val = q_evals[offset].mul_lut(lp.inv_lut());
                received[global] = received[global].add(error_val);
            }
        }

        true
    }

    fn erasure_locator_and_denominators(
        &self,
        erasure_positions: &[usize],
    ) -> ([G; FIELD_SIZE], [G; FIELD_SIZE]) {
        let erasure_count = erasure_positions.len();

        let mut pts = [G::zero(); FIELD_SIZE];
        for (k, &pos) in erasure_positions.iter().enumerate() {
            pts[k] = self.get_subspace_point_lut(pos as u8);
        }

        let mut lambda = [G::zero(); FIELD_SIZE];
        lambda[0] = G::one();
        for k in 0..erasure_count {
            let p = pts[k];
            for j in (1..=k + 1).rev() {
                lambda[j] = lambda[j - 1].add(p.mul_lut(lambda[j]));
            }
            lambda[0] = p.mul_lut(lambda[0]);
        }

        let mut denoms = [G::zero(); FIELD_SIZE];
        for j in 0..erasure_count {
            denoms[j] = (0..erasure_count)
                .filter(|&i| i != j)
                .fold(G::one(), |acc, i| acc.mul_lut(pts[j].add(pts[i])));
        }

        (lambda, denoms)
    }

    fn forney_sharded(
        q: &[&[G]],
        erasure_positions: &[usize],
        denoms: &[G],
        out: &mut [&mut [G]], // one shard per erased position, in order
    ) {
        for (k, (&pos, &d)) in erasure_positions.iter().zip(denoms).enumerate() {
            let lut = d.inv_lut().make_mul_lut();
            for (o, &num) in out[k].iter_mut().zip(q[pos].iter()) {
                *o = lut[num.into_usize()];
            }
        }
    }

    fn recover_erasure_shards(
        &self,
        received: &mut [&mut [G]],
        k_msg: usize,
        erasure_positions: &[usize],
    ) -> bool {
        let n = received.len();
        let t_parity = n - k_msg;
        let e = erasure_positions.len();

        if e > t_parity {
            return false;
        }
        if e == 0 || t_parity == 0 {
            return true;
        }

        let t_log = t_parity.trailing_zeros() as u8;
        let shard_len = received[0].len();

        let (lambda, denoms) = self.erasure_locator_and_denominators(erasure_positions);

        let mut work_backing = vec![G::zero(); n * shard_len];
        let mut work_hdrs: [MaybeUninit<&mut [G]>; FIELD_SIZE] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for (i, chunk) in work_backing.chunks_mut(shard_len).enumerate() {
            work_hdrs[i].write(chunk);
        }
        let work: &mut [&mut [G]] =
            unsafe { std::slice::from_raw_parts_mut(work_hdrs.as_mut_ptr() as *mut &mut [G], n) };

        // Chunk 0: parity block at ω_0
        for i in 0..t_parity {
            work[i].copy_from_slice(received[i]);
        }
        self.ifft_sharded(&mut work[..t_parity], t_log, G::zero());

        // Chunks 1 .. n/T-1: message blocks, each shifted by one more ωT
        {
            let mut msg_backing = vec![G::zero(); t_parity * shard_len];
            let mut msg_hdrs: [MaybeUninit<&mut [G]>; FIELD_SIZE / 2] =
                unsafe { MaybeUninit::uninit().assume_init() };
            for (i, chunk) in msg_backing.chunks_mut(shard_len).enumerate() {
                msg_hdrs[i].write(chunk);
            }
            let msg: &mut [&mut [G]] = unsafe {
                std::slice::from_raw_parts_mut(msg_hdrs.as_mut_ptr() as *mut &mut [G], t_parity)
            };

            for chunk in 1..(n / t_parity) {
                let omega = self.get_subspace_point_lut((chunk * t_parity) as u8);
                for i in 0..t_parity {
                    msg[i].copy_from_slice(received[chunk * t_parity + i]);
                }
                self.ifft_sharded(msg, t_log, omega);
                for i in 0..t_parity {
                    G::shard_add(work[i], msg[i]);
                }
            }
        }

        // Horner evaluation of λ in the monomial basis at all n Cantor subspace points
        let mut lambda_evals = [G::zero(); FIELD_SIZE];
        for (i, u) in lambda_evals.iter_mut().take(n).enumerate() {
            let p = self.get_subspace_point_lut(i as u8);
            let mut v = lambda[e]; // monic coefficient = G::one()
            for j in (0..e).rev() {
                v = v.mul_lut(p).add(lambda[j]);
            }
            *u = v;
        }

        // Evaluate s at all n points
        self.fft_sharded(work, t_log + 1, G::zero());

        // Pointwise multiply: work[i] := work[i] · λ(ω_i)
        for i in 0..n {
            let lut = lambda_evals[i].make_mul_lut();
            for b in work[i].iter_mut() {
                *b = lut[b.into_usize()];
            }
        }

        // X-basis coefficients of (s·λ); q is in work[T .. T+e]
        self.ifft_sharded(work, t_log + 1, G::zero());

        // Shift q from work[T..T+e] down to work[0..e], zero everything else
        {
            let (lo, hi) = work.split_at_mut(t_parity);
            for k in 0..e {
                lo[k].copy_from_slice(hi[k]);
                hi[k].fill(G::zero());
            }
            for l in lo.iter_mut().skip(e).take(t_parity) {
                l.fill(G::zero());
            }
        }

        // Evaluate q at all n points
        self.fft_sharded(work, n.trailing_zeros() as u8, G::zero());

        // (Forney) Eq 78: u(ω_i) = q(ω_i) / λ'(ω_i)
        for (&pos, d) in erasure_positions.iter().zip(denoms) {
            let lut = d.inv_lut().make_mul_lut();
            for (dst, &src) in received[pos].iter_mut().zip(work[pos].iter()) {
                *dst = lut[src.into_usize()];
            }
        }

        true
    }
}

impl<G: Gf2p8Lut, T: CantorBasisLut<G> + LchBasisLut<G>> Codec<G> for T {}

pub trait PolyOps<G: Gf2p8Lut>: AsRef<[G]> {
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
}

impl<G: Gf2p8Lut, T: AsRef<[G]>> PolyOps<G> for T {}

// pub struct Poly<'a, G>(pub &'a [G]);

// impl<'a, G> AsRef<[G]> for Poly<'a, G> {
//     fn as_ref(&self) -> &[G] {
//         self.0
//     }
// }

// pub struct PolyMut<'a, G>(pub &'a mut [G]);

// impl<'a, G> AsMut<[G]> for PolyMut<'a, G> {
//     fn as_mut(&mut self) -> &mut [G] {
//         self.0
//     }
// }

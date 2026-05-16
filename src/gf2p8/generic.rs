use super::bit_matrix::BitMatrix;

pub const FIELD_SIZE: usize = 256;
pub const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;

pub trait Gf2p8: Sized + Copy + From<u8> + Into<u8> + PartialEq {
    const POLY: u16;
    const PRIM_ELEM: Self;

    fn zero() -> Self {
        0u8.into()
    }

    fn one() -> Self {
        1u8.into()
    }

    fn add(self, other: Self) -> Self {
        Self::from(self.into() ^ other.into())
    }

    fn mul(self, other: Self) -> Self {
        let mut a = self.into() as u16;
        let mut b = other.into() as u16;
        let mut res = 0u16;
        for _ in 0..8 {
            if b & 1 != 0 {
                res ^= a;
            }
            a <<= 1;
            if a & 0x100 != 0 {
                a ^= Self::POLY;
            }
            b >>= 1;
        }
        (res as u8).into()
    }

    fn into_usize(self) -> usize {
        let byte: u8 = self.into();
        byte as usize
    }

    fn exp_log_tables() -> ([u8; EXP_TABLE_SIZE], [u8; FIELD_SIZE]) {
        let mut exp_table = [0u8; EXP_TABLE_SIZE];
        let mut log_table = [0u8; FIELD_SIZE];

        let mut x = 1u8;
        // build exp[0..254], log for non-zero
        for (i, e) in exp_table.iter_mut().take(FIELD_SIZE - 1).enumerate() {
            *e = x;
            log_table[x as usize] = i as u8;

            let gf_x: Self = x.into();
            x = gf_x.mul(Self::PRIM_ELEM).into();
        }

        // copy for overflow-friendly indexing
        for i in 0..FIELD_SIZE - 1 {
            exp_table[FIELD_SIZE - 1 + i] = exp_table[i];
        }

        (exp_table, log_table)
    }

    fn inv_table(
        exp_table: &[u8; EXP_TABLE_SIZE],
        log_table: &[u8; FIELD_SIZE],
    ) -> [u8; FIELD_SIZE] {
        let mut inv_table = [0u8; FIELD_SIZE];

        for (i, e) in inv_table.iter_mut().enumerate().skip(1) {
            let li = log_table[i] as usize;
            *e = exp_table[FIELD_SIZE - 1 - li];
        }

        inv_table
    }

    /// Brute-force the multiplicative inverse lookup table.
    ///
    /// 0 has no inverse. It is preserved in the output, so the case of 0 needs to be covered with checks.
    fn iter_inverses() -> impl Iterator<Item = Self> {
        std::iter::once(Self::zero()).chain((1u8..=255).map(|a| {
            let gf_a: Self = a.into();
            for b in 1u8..=255 {
                let gf_b = b.into();
                if gf_a.mul(gf_b) == Self::one() {
                    return gf_b;
                }
            }
            panic!("Cannot compute mul inverse of {a}");
        }))
    }

    /// Trace function for GF(2^8) over GF(2)
    /// Tr(x) = x + x^2 + x^4 + x^8 + x^16 + x^32 + x^64 + x^128
    fn trace(self) -> bool {
        let mut t = self;
        let mut sum: u8 = self.into();
        for _ in 0..7 {
            t = t.mul(t); // squaring
            sum ^= t.into();
        }
        (sum & 1) != 0
    }

    /// Solves the quadratic equation x^2 + x = self
    /// Returns one of the two solutions (x and x+1)
    fn solve_quadratic(self) -> Option<Self> {
        if self.trace() {
            return None;
        }
        // For GF(2^8), we can simply brute force or use the half trace.
        // Brute force is fine for a one-time basis generation.
        for i in 0u8..=255 {
            let x: Self = i.into();
            if x.mul(x).add(x) == self {
                return Some(x);
            }
        }
        None
    }

    /// Create a bit matrix for (x * self) mod POLY
    fn into_bit_matrix(self) -> BitMatrix {
        let mut m = BitMatrix([0u8; 8]);
        for i in 0..8 {
            m.0[i] = self.mul((1u8 << i).into()).into();
        }

        m.transpose()
    }
}

pub trait CantorBasis<G: Gf2p8>:
    Sized + Copy + Clone + FromIterator<G> + IntoIterator<Item = G> + AsRef<[G]>
{
    fn new() -> Self {
        let mut basis = Vec::new();

        // Start with v0 = 1
        let mut current: G = G::one();
        basis.push(current);

        // Try to extend the chain using v_i^2 + v_i = v_{i-1}
        while let Some(next) = current.solve_quadratic() {
            // We have two solutions: 'next' and 'next + 1'.
            // We must pick the one with Trace 0 to ensure the next level exists.
            if !next.trace() {
                current = next;
            } else {
                current = next.add(G::one());
            }
            basis.push(current);

            // Stop if we hit 8 elements (full field) or can't solve anymore
            if basis.len() == 8 {
                break;
            }
        }
        basis.into_iter().collect()
    }

    fn into_fft_twiddle_matrices(self) -> Vec<BitMatrix> {
        // Convert each basis element into an 8x8 bit matrix.
        // In the actual Firedancer assembly, these are the constants
        // that get loaded into ZMM registers for vgf2p8affineqb.
        self.into_iter().map(|a| a.into_bit_matrix()).collect()
    }

    /// Evaluates the erasure locator polynomial E(x) at point alpha_i.
    /// E(x) = product over missing indices j of (x ^ alpha_j).
    fn eval_erasure_locator_poly(&self, i: u8, erased_indices: &[u8]) -> G {
        let alpha_i = self.get_subspace_point(i);
        let mut eval: G = G::one();

        for &j in erased_indices {
            if i == j {
                continue;
            }
            let alpha_j = self.get_subspace_point(j);
            let factor = alpha_i.add(alpha_j);
            eval = eval.mul(factor);
        }
        eval
    }

    /// Returns the i-th point in the basis subspace.
    fn get_subspace_point(&self, i: u8) -> G {
        let mut point: G = G::zero();
        // Reverse the order of indices to match the reversed storage order.
        for (bit, elem) in (0..8).rev().zip(self.into_iter()) {
            if (i >> bit) & 1 != 0 {
                point = point.add(elem);
            }
        }
        point
    }

    fn iter_subspace_points(&self) -> (usize, impl Iterator<Item = G>) {
        let basis_len = self.into_iter().count();
        let num_points = 1 << basis_len;
        (
            num_points,
            (0..num_points).map(|i| self.get_subspace_point(i as u8)),
        )
    }

    /// Generates the LCH Twiddle Tower for an N-point FFT.
    fn generate_lch_twiddle_tower<const N: usize>(&self) -> Vec<BitMatrix> {
        let k = N.trailing_zeros() as usize;

        let mut current_basis: Vec<G> = self.into_iter().take(k).collect();
        let mut twiddles = Vec::with_capacity(k);

        // Build the tower from the bottom
        for i in (0..k).rev() {
            // The pivot for the current butterfly layer
            let beta = current_basis[i];
            twiddles.push(beta.into_bit_matrix());

            // Transform the lower-level basis elements for the sub-FFTs.
            // This is the "Subspace Polynomial" step: L(x) = x^2 + x*beta.
            // It ensures that the sub-problems see unique points.
            for j in 0..i {
                let x = current_basis[j];
                current_basis[j] = (x.mul(x)).add(x.mul(beta));
            }
        }

        twiddles.reverse();
        twiddles
    }

    fn eval_subspace_poly(&self, k: u8, x: G) -> G {
        let mut val: G = 1.into();
        for a in self.span_by_gray_code(k) {
            let sum = x.add(a);
            val = val.mul(sum);
        }
        val
    }

    fn chain_of_subspaces(&self) -> Vec<Vec<G>> {
        (0..9).map(|k| self.span(k)).collect()
    }

    fn span(&self, k: u8) -> Vec<G> {
        let size = 1 << k;
        let mut res = Vec::with_capacity(size);
        for i in 0..size {
            let mut sum: G = 0.into();
            for (j, v) in self.into_iter().take(k as usize).enumerate() {
                if (i >> j) & 1 == 1 {
                    sum = sum.add(v);
                }
            }
            res.push(sum);
        }
        res
    }

    fn span_by_gray_code(&self, k: u8) -> Vec<G> {
        let size = 1 << k;
        let mut span: Vec<G> = vec![G::zero(); size];
        for i in 1..size {
            let lsb = i.trailing_zeros() as usize;
            span[i] = span[i ^ (1 << lsb)].add(self.as_ref()[lsb]);
        }
        span
    }

    // FIXME: Broken!
    //
    // /// Generates the compact LUT for the subspace polynomial s_k(x).
    // /// The const K is the table size 2^(8-k). the level index k is derived as 8 - log2(K).
    // fn gen_compact_subspace_poly_lut<const K: usize>(&self) -> [G; K] {
    //     let log2_k = K.trailing_zeros() as usize;
    //     let k = 8 - log2_k;

    //     let mut table = [G::zero(); K];
    //     let basis = self.as_ref();

    //     // Pre-compute projections b_l = s_l(v_l) for l < k.
    //     // These are the basis point projections required to evaluate s_k.
    //     let mut basis_image = [G::from(0u8); 8];
    //     let mut w = basis.to_vec();
    //     for l in 0..k {
    //         let b_l = w[l];
    //         basis_image[l] = b_l;
    //         for w_j in w.iter_mut().take(k).skip(l + 1) {
    //             // s_{l+1}(v_j) = s_l(v_j) * (s_l(v_j) + s_l(v_l))
    //             *w_j = w_j.mul(w_j.add(b_l));
    //         }
    //     }

    //     // Compute s_k(X_{r << k}) for all r in the compressed range.
    //     for r in 0..K {
    //         // We evaluate s_k only at coset representatives of the subspace V_k.
    //         // These are points where the lower k bits are zero.
    //         let mut val = self.get_subspace_point((r << k) as u8);

    //         // Apply the chain of subspace polynomials s_0 -> s_1 -> ... -> s_k
    //         for &b_l in basis_image.iter().take(k) {
    //             val = val.mul(val.add(b_l));
    //         }
    //         table[r] = val;
    //     }

    //     table
    // }

    /// Generates a LUT for the subspace polynomial s_k(x).
    /// The table index is the field element (as u8), and the value is s_k(index).
    ///
    /// - s_0(x) = x
    /// - s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
    fn gen_subspace_poly_lut(&self, k: usize) -> [G; FIELD_SIZE] {
        let mut table = [G::zero(); FIELD_SIZE];

        // Base case: s_0(x) = x
        for x in 0..FIELD_SIZE {
            table[x] = G::from(x as u8);
        }

        let basis = self.as_ref();
        // We only need to iterate if k > 0
        for v_j in basis.iter().take(k) {
            // b_j = s_j(v_j)
            // To find this, we evaluate the current state of s_j at point v_j.
            // Since our table currently holds s_j(x) for all x,
            // we just look up the index corresponding to basis element v_j.
            let b_j = table[(*v_j).into_usize()];

            // Update the table: s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
            for x in table.iter_mut() {
                let s_j_x = *x;
                *x = s_j_x.mul(s_j_x.add(b_j));
            }
        }

        table
    }

    /// Generates all subspace polynomial LUTs from s_0 to s_8.
    /// Returns an array where [j][x] contains s_j(x).
    ///
    /// - s_0(x) = x
    /// - s_{j+1}(x) = s_j(x) * (s_j(x) + s_j(v_j))
    fn gen_all_subspace_poly_luts(&self) -> [[G; FIELD_SIZE]; 9] {
        let mut luts = [[G::zero(); FIELD_SIZE]; 9];

        // Initialize s_0(x) = x
        for (x, s_0_x) in luts[0].iter_mut().enumerate() {
            *s_0_x = G::from(x as u8);
        }

        let basis = self.as_ref();

        // Iteratively compute s_{j+1} from s_j
        for j in 0..8 {
            // b_j is the basis projection: s_j(v_j)
            // Look up the basis element v_j in the current s_j table
            let b_j = luts[j][basis[j].into_usize()];

            for x in 0..FIELD_SIZE {
                let s_j_x = luts[j][x];
                // s_{j+1}(x) = s_j(x) * (s_j(x) + b_j)
                luts[j + 1][x] = s_j_x.mul(s_j_x.add(b_j));
            }
        }

        luts
    }

    /// Generates 1/p_i values, for 0 <= i < 256.
    fn gen_normalization_factors(
        &self,
        subspace_poly_luts: &[[G; FIELD_SIZE]; 9],
        inv_lut: &[u8; FIELD_SIZE],
    ) -> [u8; FIELD_SIZE] {
        let basis = self.as_ref();
        let basis_image: Vec<G> = (0..8)
            .map(|i| subspace_poly_luts[i][basis[i].into_usize()])
            .collect();
        let mut factors = [0u8; FIELD_SIZE];

        for (i, f) in factors.iter_mut().enumerate() {
            let mut p: G = G::one();
            for (j, &b) in basis_image.iter().enumerate() {
                if (i >> j) & 1 == 1 {
                    p = p.mul(b);
                }
            }
            *f = inv_lut[p.into_usize()];
        }

        factors
    }

    /// Generates the derivatives of subspace polynomial terms.
    fn gen_deriv_subspace_poly_lut(&self) -> [G; 9] {
        let mut derivs = [G::one(); 9];

        for (i, d) in derivs.iter_mut().enumerate() {
            let span = self.span_by_gray_code(i as u8);
            for s in span {
                if s != G::zero() {
                    *d = (*d).mul(s)
                }
            }
        }

        derivs
    }

    /// Generates bitmasks of subspace polynomials $s_k$ coefficients of the $x^2^i$ terms for
    /// $2 <= i <= 8$. Coefficient of $x$ is always 1 and is thus hard-coded in `CantorBasisLut`.
    fn gen_subspace_poly_coeffs() -> impl Iterator<Item = u8> {
        let mut masks = [0u16; 9];

        // Base case: s_0(x) = x^1 (bit 0 set)
        masks[0] = 0b000000001;

        for j in 0..8 {
            // s_{j+1} = s_j^2 + s_j
            // Squaring a linearized polynomial is a bit shift.
            // Adding in GF(2^8) is XOR.
            masks[j + 1] = (masks[j] << 1) ^ masks[j];
        }

        masks.into_iter().map(|m| (m >> 1) as u8)
    }

    /// Generate the monomial coefficients of the standard form of a ring basis polynomial X_i,
    /// for 0 <= i < 256.
    fn gen_ring_basis_poly_coeffs(&self, i: u8) -> [G; FIELD_SIZE] {
        // Select the subspace poly terms s_k
        for j in 0..8 {}
        todo!()
    }
}

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

    fn fft_sharded(&self, shards: &mut [&mut [u8]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);

        let (left, right) = shards.split_at_mut(half);
        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let (s_i, s_i_half) = (&mut left[i], &mut right[i]);
            for (byte_l, byte_r) in s_i.iter_mut().zip(s_i_half.iter_mut()) {
                let d_i = G::from(*byte_l);
                let d_i_half = G::from(*byte_r);
                let g_i_0 = d_i.add(twiddle.mul(d_i_half));
                let g_i_1 = g_i_0.add(d_i_half);
                *byte_l = g_i_0.into();
                *byte_r = g_i_1.into();
            }
        }

        self.fft_sharded(left, k - 1, beta);
        self.fft_sharded(right, k - 1, beta.add(self.get_basis_point_lut(k - 1)));
    }

    fn ifft_sharded(&self, shards: &mut [&mut [u8]], k: u8, beta: G) {
        if k == 0 {
            return;
        }
        let half = 1 << (k - 1);

        let (left, right) = shards.split_at_mut(half);
        self.ifft_sharded(left, k - 1, beta);
        self.ifft_sharded(right, k - 1, beta.add(self.get_basis_point_lut(k - 1)));

        let twiddle = self.eval_subspace_poly_lut(k - 1, beta);

        for i in 0..half {
            let (s_i, s_i_half) = (&mut left[i], &mut right[i]);
            for (byte_l, byte_r) in s_i.iter_mut().zip(s_i_half.iter_mut()) {
                let g_i_0 = G::from(*byte_l);
                let g_i_1 = G::from(*byte_r);
                let d_i_half = g_i_0.add(g_i_1);
                let d_i = g_i_0.add(twiddle.mul(d_i_half));
                *byte_l = d_i.into();
                *byte_r = d_i_half.into();
            }
        }
    }

    /*
    /// Performs A = Q * B + R in Basis X.
    fn poly_div_rem<I, O>(&self, dividend: I, divisor: I, mut quotient: O, mut remainder: O, k: u8)
    where
        I: AsRef<[G]>,
        O: AsMut<[G]> + AsRef<[G]>,
    {
        let d_lc_inv = divisor.leading_coeff().inv_lut();
        let d_deg = divisor.degree();

        quotient.as_mut().fill(G::zero());
        remainder.as_mut().fill(G::zero());

        let src = dividend.as_ref();
        remainder.as_mut()[..src.len()].copy_from_slice(src);

        // Synthetic division loop
        loop {
            let (deg, lc) = {
                let deg = remainder.as_ref().degree();
                if deg < d_deg {
                    break;
                }
                let lc = remainder.leading_coeff();
                if lc == G::zero() {
                    break;
                }
                (deg, lc)
            };
            // q_term = (LC(R) / LC(B)) * X_{r_deg - d_deg}
            let q_coeff = lc.mul(d_lc_inv);
            let deg_diff = deg - d_deg;
            quotient.as_mut()[deg_diff] = q_coeff;

            // R = R - (q_coeff * X_deg_diff * B)
            self.poly_fused_mul_add(remainder.as_mut(), divisor.as_ref(), q_coeff, deg_diff, k);
        }
    }
    */

    fn init_eea_modulus(
        &self,
        r0: &mut [G], // Coefficient array for EEA (size FIELD_SIZE)
        t_log: u8,
        is_systematic: bool,
    ) {
        r0.fill(G::zero());

        // s_t(x) = sum (mask_bit_i * x^{2^i})
        let mask = self.get_subspace_poly_coeff_lut(t_log);

        // Unpack the linearized coefficients into the standard basis
        // Each bit i corresponds to the term x^(2^{i+1})
        r0[1] = G::one(); // Coefficient of x stays 1.
        for i in 0..t_log {
            if (mask >> i) & 1 == 1 {
                // Map bit i to the index 2^i
                // e.g., i=0 -> r0[1], i=1 -> r0[2], i=2 -> r0[4]
                r0[1 << (i + 1)] = G::one();
            }
        }

        // Systematic data shift (Eq 76)
        // To find errors in data blocks (indices >= T), the modulus
        // must be shifted by the subspace evaluation of the coset.
        // In the Cantor basis where s_j(v_j) = 1, this shift is always 1.
        if is_systematic {
            // r0(x) becomes s_t(x) + 1
            r0[0] = G::one();
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

    /*
    /// Step 2 of decoding: key equation solver via extended Euclidean algorithm.
    /// Solves: z_0(x) = s(x)λ(x) + q(x)s_t(x)
    ///
    /// # Arguments
    /// * `syndrome` - Input syndrome polynomial.
    /// * `t_log`    - log_2 of the parity size T (subspace dimension).
    ///
    /// # Returns
    /// * `.0` - Coeffcients of the Quotient Polynomial q.
    /// * `.1` - Coeficients of the Error Locator Polynomial λ.
    fn solve_key_equation_eea(
        &self,
        syndrome: &[G], // Size T
        t_log: u8,
    ) -> ([G; FIELD_SIZE], [G; FIELD_SIZE]) {
        let t_parity = 1 << t_log;
        let t_half = t_parity / 2;
        let k_arith = t_log + 1; // Arithmetic headroom

        // r = u * s_t + v * s
        // z0 -> r, q -> u, lambda -> v.

        let mut r0 = [G::zero(); FIELD_SIZE];
        self.init_eea_modulus(&mut r0, t_log, true);
        println!("r0={:?}", r0.iter().map(|&x| x.into()).collect::<Vec<u8>>());
        let mut u0 = [G::zero(); FIELD_SIZE];
        u0[0] = G::one(); // u associated with s_t
        let mut v0 = [G::zero(); FIELD_SIZE];

        let mut r1 = [G::zero(); FIELD_SIZE];
        r1[..syndrome.len()].copy_from_slice(syndrome);
        let mut u1 = [G::zero(); FIELD_SIZE];
        let mut v1 = [G::zero(); FIELD_SIZE];
        v1[0] = G::one(); // v associated with s

        while r1.degree() >= t_half {
            let q_coeff = r0.leading_coeff().mul(r1.leading_coeff().inv_lut());
            let deg_diff = r0.degree() - r1.degree();

            // Update all three: remainder and both auxiliaries
            self.poly_fused_mul_add(&mut r0, r1, q_coeff, deg_diff, k_arith);
            self.poly_fused_mul_add(&mut u0, u1, q_coeff, deg_diff, k_arith);
            self.poly_fused_mul_add(&mut v0, v1, q_coeff, deg_diff, k_arith);

            if r0.degree() < r1.degree() {
                r0.swap_with_slice(&mut r1);
                u0.swap_with_slice(&mut u1);
                v0.swap_with_slice(&mut v1);
            }
        }
        // Result: r1 is remainder, u1 is numerator q(x), v1 is locator lambda(x)
        (u1, v1)
    }
    */

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
        st.fill(G::zero());
        st[1 << t_log] = G::one(); // Coefficient of x stays 1.

        // // s_t(x) = sum (mask_bit_i * x^{2^i})
        // let mask = self.get_subspace_poly_coeff_lut(t_log);

        // // Unpack the linearized coefficients into the standard basis
        // // Each bit i corresponds to the term x^(2^{i+1})
        // for i in 0..t_log {
        //     if (mask >> i) & 1 == 1 {
        //         // Map bit i to the index 2^i
        //         // e.g., i=0 -> r0[1], i=1 -> r0[2], i=2 -> r0[4]
        //         st[1 << (i + 1)] = G::one();
        //     }
        // }
    }

    /// Division in the monomial basis.
    /// s_t(x) = q_t(x) * s(x) + r_t(x)
    fn poly_div_mon(&self, a: &[G], b: &[G]) -> Option<([G; FIELD_SIZE], [G; FIELD_SIZE])> {
        let a_deg = a.degree();
        if a_deg.is_none() {
            return Some(([G::zero(); FIELD_SIZE], [G::zero(); FIELD_SIZE]));
        }
        let a_deg = a_deg.unwrap();

        let mut r = [G::zero(); FIELD_SIZE];
        r.copy_from_slice(a);
        let mut q = [G::zero(); FIELD_SIZE];
        let b_deg = b.degree();

        // Division by 0
        if b_deg.is_none() {
            return None;
        }
        let b_deg = b_deg.unwrap();

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
        let deg_a = a.degree();

        if deg_a.is_none() {
            return [G::zero(); FIELD_SIZE];
        }
        let deg_a = deg_a.unwrap();

        let deg_b = b.degree();
        if deg_b.is_none() {
            return [G::zero(); FIELD_SIZE];
        }
        let deg_b = deg_b.unwrap();

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
        for i in pivot..FIELD_SIZE {
            hi[i - pivot] = p[i];
        }
        (lo, hi)
    }

    /// Multiply p by s_k = X_{2^k} by shifting coefficients up by 2^k.
    ///
    /// Valid only when every nonzero coefficient of p sits at an index
    /// where bit k is 0 — always satisfied by the HGCD invariants.
    fn poly_shift_up(&self, p: &[G; FIELD_SIZE], k: u8) -> [G; FIELD_SIZE] {
        let shift = 1usize << k;
        let mut out = [G::zero(); FIELD_SIZE];
        for i in 0..FIELD_SIZE - shift {
            out[i + shift] = p[i];
        }
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
        let q = 1usize << (g - 2); // 2^{g-2}
        let h = q * 2; // 2^{g-1}

        let (mut p_ll, mut p_lh, mut p_h) = (
            [G::zero(); FIELD_SIZE],
            [G::zero(); FIELD_SIZE],
            [G::zero(); FIELD_SIZE],
        );
        for i in 0..q {
            p_ll[i] = p[i];
            p_lh[i] = p[i + q];
            p_h[i] = p[i + h];
        }

        // c = s_{g-2}(v_{g-2}), where v_{g-2} = ω_{2^{g-2}}
        let v_gm2 = self.get_subspace_point_lut(1u8 << (g - 2));
        let c = self.eval_subspace_poly_lut(g - 2, v_gm2);

        // p_m[0..q)  = p_lh + c · p_h  (low part)
        // p_m[q..2q) = p_h             (shift of p_h by s_{g-2})
        let mut p_m = [G::zero(); FIELD_SIZE];
        for i in 0..q {
            p_m[i] = p_lh[i].add(c.mul_lut(p_h[i]));
            p_m[i + q] = p_h[i];
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
            .expect("z_m1 nonzero: guaranteed by the HGCD degree invariant");

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

/// Computes the formal derivative of a polynomial in the basis X.
/// Follows Eq 82 from the LNH paper.
pub fn deriv_poly<G: Gf2p8>(coeffs: &[G], out: &mut [G], k: u8) {
    if k == 0 {
        out[0] = G::zero();
        return;
    }

    let half = 1 << (k - 1);
    let (low, high) = coeffs.split_at(half);
    let (out_low, out_high) = out.split_at_mut(half);

    // Compute [D0]' and [D1]'
    deriv_poly(low, out_low, k - 1);
    deriv_poly(high, out_high, k - 1);

    // Combine the results according to Eq 82
    // out_low has [D0]'
    // out_high has [D1]'

    // Term 2: Add s'_{k-1} * D1. Assuming s' = 1, we XOR high into out_low.
    for (l, h) in out_low.iter_mut().zip(high.iter()) {
        *l = l.add(*h);
    }

    // Term 3: s_{k-1} * [D1]' is already in out_high.  Multiplying by s_{k-1} in basis X is
    // just a shift into the upper half of the coefficient vector.
}

pub fn deriv_poly_iterative<G: Gf2p8>(coeffs: &[G], out: &mut [G]) {
    let n = coeffs.len();
    let m = n.trailing_zeros() as usize;

    // TODO: eliminate double init
    out.fill(G::zero());

    for j in (1..=m).rev() {
        let step = 1 << j;
        let half = 1 << (j - 1);

        // Iterate through each subspace coset at this level
        for start in (0..n).step_by(step) {
            let (low_out, _high_out) = out[start..start + step].split_at_mut(half);
            let high_in = &coeffs[start + half..start + step];

            // Eq 82: Low Part += s'_{j-1} * High Part
            // The high part of 'out' is updated automatically in subsequent iterations
            // (smaller j) acting on the high blocks.
            for (l, &h) in low_out.iter_mut().zip(high_in.iter()) {
                *l = l.add(h);
            }
        }
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
        // debug_assert!((0..=8).any(|k| coeffs.len() == 2usize.pow(k) - 1));
        // let k = coeffs.len().trailing_zeros() as usize;
        let mut result: G = G::zero();

        for (i, d) in coeffs.iter().enumerate() {
            let term = d.mul(self.eval_lch_basis_poly(i as u8, x));
            result = result.add(term);
        }

        result
    }
}

pub trait Codec<G: Gf2p8Lut>: CantorBasisLut<G> + LchBasisLut<G> {
    /// Systematic Reed-Solomon encoding: The message stays the same. Only parity is modified.
    ///
    /// This method is for the special case when the number of message shards equals the number of
    /// parity shards, and is a power of 2.
    fn encode_systematic(&self, message_shards: &[&[u8]], parity_shards: &mut [&mut [u8]]) {
        let num_parity = parity_shards.len();
        let log_num_parity = num_parity.trailing_zeros() as u8;

        // TODO: this can be done by the caller, in which case only the mut buffer is required.
        for (m_shard, p_shard) in message_shards.iter().zip(parity_shards.iter_mut()) {
            p_shard.copy_from_slice(m_shard);
        }

        let beta = self.get_subspace_point_lut(num_parity as u8);
        self.ifft_sharded(parity_shards, log_num_parity, beta);
        self.fft_sharded(parity_shards, log_num_parity, G::zero());
    }

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
            for (w, &r) in workspace.iter_mut().zip(chunk.iter()) {
                *w = r;
            }

            // Perform the partial IFFT (Algorithm 2)
            // This moves the chunk from Evaluation Space -> Novel Basis Coefficients
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

    /// Computes the syndrome $\mathbf{s}(x)$ per LCH formalism:
    /// Parity shards are at the beginning (indices 0..T-1).
    ///
    /// # Arguments
    /// * `received_shards` - Full array of n shards [parity_0..parity_{T-1}, data_0..data_{k-1}]
    /// * `syndrome_shards` - Output buffer of size T (coefficients of s(x))
    /// * `scratchpad`      - Workspace of size T shards
    fn compute_syndrome_sharded(
        &self,
        received_shards: &[&[u8]],
        syndrome_shards: &mut [&mut [u8]],
        scratchpad: &mut [&mut [u8]],
    ) {
        let t_parity = syndrome_shards.len(); // T = 2^t
        let t_log = t_parity.trailing_zeros() as u8;

        for s in syndrome_shards.iter_mut() {
            s.fill(0);
        }

        // Equation (67): s = sum_{i=0}^{n/T - 1} IFFT(r_i, t, omega_{i*T})
        // i=0 corresponds to the parity chunk (v_0)
        for (i, r_chunk) in received_shards.chunks(t_parity).enumerate() {
            let omega_idx = i * t_parity;
            // omega_{i * T}
            let omega = self.get_subspace_point_lut(omega_idx as u8);

            for s in scratchpad.iter_mut() {
                s.fill(0);
            }
            for (src, dst) in r_chunk.iter().zip(scratchpad.iter_mut()) {
                dst.copy_from_slice(src);
            }

            self.ifft_sharded(scratchpad, t_log, omega);

            // Linearity of IFFT allows to sum in the coefficient domain
            for (s_shard, temp_shard) in syndrome_shards.iter_mut().zip(scratchpad.iter()) {
                for (s_byte, t_byte) in s_shard.iter_mut().zip(temp_shard.iter()) {
                    *s_byte ^= *t_byte;
                }
            }
        }
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
            println!("Syndrome is zero.");
            return true;
        }

        // Step 2: key equation
        let (v1, lambda) = match self.solve_key_equation_eea(&syndrome, t_log) {
            Some(pair) => pair,
            None => {
                // TODO: error type enum
                println!("Key equation has no solution.");
                return false;
            }
        };

        let deg_lambda = match lambda.degree() {
            Some(d) => d,
            None => {
                println!("Zero locator - no errors.");
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
            println!(
                "Too few roots. deg_lambda={deg_lambda}, error_indices={error_indices:?}, syndrome={:?}, v1={:?}, lambda={:?}",
                syndrome.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
                v1.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
                lambda.iter().map(|&x| x.into()).collect::<Vec<u8>>(),
            );
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

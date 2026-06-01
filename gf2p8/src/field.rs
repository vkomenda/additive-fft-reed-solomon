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

    fn mul_with_tables(
        self,
        other: Self,
        exp: &[u8; EXP_TABLE_SIZE],
        log: &[u8; FIELD_SIZE],
    ) -> Self {
        if self == Self::zero() || other == Self::zero() {
            return Self::zero();
        }
        exp[log[self.into_usize()] as usize + log[other.into_usize()] as usize].into()
    }

    fn make_mul_table(
        self,
        exp: &[u8; EXP_TABLE_SIZE],
        log: &[u8; FIELD_SIZE],
    ) -> [u8; FIELD_SIZE] {
        let mut mul_table = [0u8; 256];
        for x in 0..=255u8 {
            mul_table[x as usize] = self.mul_with_tables(x.into(), exp, log).into();
        }
        mul_table
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
    fn into_mul_matrix(self) -> BitMatrix {
        let mut m = BitMatrix([0u8; 8]);
        for i in 0..8 {
            m.0[i] = self.mul((1u8 << i).into()).into();
        }

        m.transpose()
    }

    fn iter_gfni_mul_matrices() -> impl Iterator<Item = u64> {
        (0..FIELD_SIZE).map(|i| Self::from(i as u8).into_mul_matrix().to_gfni_u64())
    }

    // TODO: vectorized ops need to move to a dedicated trait.
    fn shard_add(a: &mut [Self], b: &[Self]) {
        for (x, y) in a.iter_mut().zip(b) {
            *x = x.add(*y);
        }
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
        for (bit, elem) in (0..8).zip(*self) {
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
}

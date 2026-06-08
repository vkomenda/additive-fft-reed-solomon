use crate::{
    gf2p8lut::{CantorBasisLut, Gf2p8Lut},
    kernel::Kernel,
    poly_arith::{
        CantorBasisPolyArith, CantorBasisPolySliceArith, PolyMutSliceArith, PolySliceArith, poly,
    },
};
use std::marker::PhantomData;

#[derive(Copy, Clone)]
pub struct Codec<G, B, K, const N: usize, const T: usize> {
    basis: B,
    _kernel: PhantomData<(G, K)>,
}

impl<G, B, K, const N: usize, const T: usize> Codec<G, B, K, N, T>
where
    G: Gf2p8Lut,
    B: CantorBasisLut<G> + Default,
    K: Kernel<G>,
{
    pub fn new() -> Self {
        Self {
            basis: B::default(),
            _kernel: PhantomData,
        }
    }

    #[allow(unused)]
    pub(crate) fn solve_key_equation_eea(&self, syndrome: &[G; N]) -> Option<([G; N], [G; N])> {
        let mut st = [G::zero(); N];
        st[T] = G::one();
        let (qt, rt) = self.basis.poly_div_lnh(&st, syndrome)?;
        let (u1, v1, _z1) = self.eea(syndrome, &rt);
        let lambda = poly::add(&u1, &self.basis.poly_mul_lnh(&v1, &qt));
        Some((v1, lambda))
    }

    /// Extended Euclidean Algorithm.
    pub(crate) fn eea(
        &self,
        a: &[G], // syndrome
        b: &[G], // remainder r_t(x) from the initial division
    ) -> ([G; N], [G; N], [G; N]) {
        let target_deg = T / 2; // Stop when deg(z) < T/2

        let mut z0 = [G::zero(); N];
        z0.copy_from_slice(a);
        let mut z1 = [G::zero(); N];
        z1.copy_from_slice(b);

        let mut u0 = [G::zero(); N];
        u0[0] = G::one();
        let mut u1 = [G::zero(); N];

        let mut v0 = [G::zero(); N];
        let mut v1 = [G::zero(); N];
        v1[0] = G::one();

        while z1.degree().is_some_and(|d| d >= target_deg) {
            let (q, remainder) = self
                .basis
                .poly_div_lnh(&z0, &z1)
                .expect("z1 is non-zero thanks to while condition");

            // Update r: r = r0 - q * r1
            z0 = z1;
            z1 = remainder;

            // Update u: u = u0 - q * u1
            let q_u1 = self.basis.poly_mul_lnh(&q, &u1);
            let next_u = poly::add(&u0, &q_u1);
            u0 = u1;
            u1 = next_u;

            // Update v: v = v0 - q * v1
            let q_v1 = self.basis.poly_mul_lnh(&q, &v1);
            let next_v = poly::add(&v0, &q_v1);
            v0 = v1;
            v1 = next_v;
        }

        // LNH page 9:
        // u1 = u auxiliary, v1 = v auxiliary, z1 = z error evaluator
        (u1, v1, z1)
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
    pub(crate) fn poly_hgcd_middle(&self, p: &[G; N], g: u8) -> ([G; N], [G; N]) {
        debug_assert!(g >= 2);
        let q = 1usize << (g - 2); // 2^{g-2}
        let h = 1usize << (g - 1); // 2^{g-1}
        let c = self
            .basis
            .eval_subspace_poly_lut(g - 2, self.basis.get_subspace_point_lut(1u8 << (g - 2)));

        let mut p_ll = [G::zero(); N];
        let mut p_m = [G::zero(); N];
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

    /// Half-GCD algorithm (Algorithm 5, LNH).
    ///
    /// Preconditions: $deg(b) \le deg(a),  2^{g-1} \le deg(a) < 2^g$.
    ///
    /// Returns (z0, z1, M) where M = [m00, m01, m10, m11] (row-major) satisfies
    /// - $[z_0, z_1]^T = M · [a, b]^T$,
    /// - $deg(z_0) \ge 2^{g-1}, deg(z_1) < 2^{g-1}$.
    pub(crate) fn hgcd(&self, a: &[G; N], b: &[G; N], g: u8) -> ([G; N], [G; N], [[G; N]; 4]) {
        let zero = [G::zero(); N];
        let one = {
            let mut p = zero;
            p[0] = G::one();
            p
        };
        let half = 1 << (g - 1);

        // Base case (Algorithm 5 lines 1-2)
        // deg(b) < 2^{g-1}: Z = [a, b], M = I.
        if b.degree().is_none_or(|d| d < half) {
            return (*a, *b, [one, zero, zero, one]);
        }

        // Step 3: split at 2^{g-1}, recurse on high halves
        let (a_l, a_h) = poly::split_at(a, g - 1);
        let (b_l, b_h) = poly::split_at(b, g - 1);
        let (z_h0, z_h1, m_h) = self.hgcd(&a_h, &b_h, g - 1);

        // Step 4: z_M = Z_H · s_{g-1} + M_H · [a_L, b_L]^T
        // Equivalently M_H · (s_{g-1}·[a_H,b_H] + [a_L,b_L]) = M_H · [a, b].
        let (mv0, mv1) = self.basis.mat_vec_lnh(&m_h, &a_l, &b_l);
        let z_m0 = poly::add(&poly::shift_up(&z_h0, g - 1), &mv0);
        let z_m1 = poly::add(&poly::shift_up(&z_h1, g - 1), &mv1);

        // Step 5: early return if deg(z_M1) < 2^{g-1} (lines 5-6)
        if z_m1.degree().is_none_or(|d| d < half) {
            return (z_m0, z_m1, m_h);
        }

        // Step 7: divide z_M0 by z_M1
        let (q_m, r_m) = self
            .basis
            .poly_div_lnh(&z_m0, &z_m1)
            .expect("z_m1 non-zero: guaranteed by the HGCD degree invariant");

        // Step 8: decompose z_M1 and r_M into LL and M parts
        let (z_m1_ll, z_m1_m) = self.poly_hgcd_middle(&z_m1, g);
        let (r_m_ll, r_m_m) = self.poly_hgcd_middle(&r_m, g);

        // Step 9: second recursive call
        let (y_m0, y_m1, m_m) = self.hgcd(&z_m1_m, &r_m_m, g - 1);

        // Step 10
        let swap = [zero, one, one, q_m];
        let m_r = self
            .basis
            .mat_mul_lnh(&m_m, &self.basis.mat_mul_lnh(&swap, &m_h));

        // Y_M · s_{g-2}: y_m1 is safe to shift (degree < 2^{g-2}, bit g-2 always 0),
        // but y_m0 can have coefficients at index 2^{g-2} (bit g-2 set), so it
        // requires a proper polynomial multiplication rather than a plain index shift.
        let sg_minus_2 = {
            let mut p = [G::zero(); N];
            p[1 << (g - 2)] = G::one(); // s_{g-2} = X_{2^{g-2}} in basis X
            p
        };
        let (mv0, mv1) = self.basis.mat_vec_lnh(&m_m, &z_m1_ll, &r_m_ll);
        let z_r0 = poly::add(&self.basis.poly_mul_lnh(&y_m0, &sg_minus_2), &mv0);
        let z_r1 = poly::add(&poly::shift_up(&y_m1, g - 2), &mv1); // y_m1 safe

        (z_r0, z_r1, m_r)
    }

    /// This is functionally equivalent to `solve_key_equation_eea` and is what the LNH paper
    /// has.
    pub(crate) fn solve_key_equation_hgcd(
        &self,
        syndrome: &[G; N],
        t_log: u8,
    ) -> Option<([G; N], [G; N])> {
        let mut st = [G::zero(); N];
        st[T] = G::one();

        // s_t = q_t · s + r_t
        let (q_t, r_t) = self.basis.poly_div_lnh(&st, syndrome)?;

        // HGCD(s, r_t, t_log) => M = [m00,m01,m10,m11] with
        //   z1 = m10·s + m11·r_t
        //      = m11·s_t + (m10 + m11·q_t)·s     [since r_t = s_t − q_t·s]
        // so the error locator is λ = m10 + m11·q_t  (eq. 79, GF(2): - is +)
        // and the error evaluator is v1 = m11
        let (_z0, _z1, m) = self.hgcd(syndrome, &r_t, t_log);

        let u1 = m[2]; // m10
        let v1 = m[3]; // m11
        let lambda = poly::add(&u1, &self.basis.poly_mul_lnh(&v1, &q_t));

        Some((v1, lambda))
    }

    pub fn encode_systematic_scalar(&self, message: &[G], parity: &mut [G]) {
        let t_log = T.trailing_zeros() as u8;
        let k_msg = N - T;

        // Compute parity image (v0') using LNH Eq 68
        parity.fill(G::zero());
        let mut workspace = [G::zero(); T];

        for i in 0..k_msg / T {
            workspace[..T].copy_from_slice(&message[i * T..(i + 1) * T]);
            let omega = self.basis.get_subspace_point_lut(((i + 1) * T) as u8);
            self.basis.ifft_scalar(&mut workspace[..T], t_log, omega);
            parity.poly_add_in_place(&workspace[..T]);
        }

        // Compute parity (v0)
        self.basis.fft_scalar(parity, t_log, G::zero());
    }

    pub fn encode_systematic_sharded(
        &self,
        message: &[G],
        parity: &mut [G],
        workspace: &mut [G],
        shard_len: usize,
    ) {
        debug_assert_eq!(message.len(), (N - T) * shard_len);
        debug_assert_eq!(parity.len(), T * shard_len);
        debug_assert_eq!(workspace.len(), T * shard_len);

        let t_log = T.trailing_zeros() as u8;
        let k = N - T;

        for i in 0..k / T {
            workspace.copy_from_slice(&message[i * T * shard_len..(i + 1) * T * shard_len]);
            let omega = self.basis.get_subspace_point_lut(((i + 1) * T) as u8);
            K::ifft_sharded(&self.basis, workspace, shard_len, t_log, omega);
            for (p, w) in parity.iter_mut().zip(workspace.iter()) {
                *p = p.add(*w);
            }
        }

        K::fft_sharded(&self.basis, parity, shard_len, t_log, G::zero());
    }

    /// Syndrome calculation (scalar).
    /// Computes s = sum_{i=0}^{n/T-1} IFFT(r_i, t, omega_{i*T})
    fn compute_syndrome_scalar(
        &self,
        received: &[G], // Size n (e.g., 256)
    ) -> [G; N] {
        // Reserve the extra bit for the key equation solver (EEA requirement).
        let mut syndrome = [G::zero(); N];
        let mut workspace = [G::zero(); T];

        for (i, chunk) in received.chunks(T).enumerate() {
            // beta corresponds to the starting point of the i-th chunk: omega_{i*T}
            let omega_idx = (i * T) as u8;
            let beta = self.basis.get_subspace_point_lut(omega_idx);

            // Copy received chunk into workspace
            // Pad with zeros if the last chunk is partial (Eq 63)
            workspace[..T].fill(G::zero());
            for (w, &r) in workspace[..T].iter_mut().zip(chunk.iter()) {
                *w = r;
            }

            let t_log = T.trailing_zeros() as u8;
            // Perform the partial IFFT (Algorithm 2)
            // This moves the chunk from evaluation space to basis X coefficients
            self.basis.ifft_scalar(&mut workspace[..T], t_log, beta);

            // Accumulate into the syndrome buffer
            for (s, &w) in syndrome.iter_mut().take(T).zip(workspace[..T].iter()) {
                *s = s.add(w);
            }
        }

        syndrome
    }

    /// Recomputes data from parity when $n = 2T$.
    pub fn recompute_data_from_parity(&self, received: &mut [G]) {
        let t_log = T.trailing_zeros() as u8;

        let mut workspace = [G::zero(); T];

        workspace[..T].copy_from_slice(&received[..T]);
        self.basis.ifft_scalar(&mut workspace, t_log, G::zero());
        self.basis.fft_scalar(
            &mut workspace,
            t_log,
            self.basis.get_subspace_point_lut(T as u8),
        );

        received[T..].copy_from_slice(&workspace[..T]);
    }

    /// Recomputes data shards from parity shards when $n = 2T$.
    pub fn recompute_data_from_parity_sharded(
        &self,
        received: &mut [G],
        workspace: &mut [G],
        shard_len: usize,
    ) {
        debug_assert_eq!(N, 2 * T);
        debug_assert_eq!(received.len(), N * shard_len);
        debug_assert_eq!(workspace.len(), T * shard_len);

        let t_log = T.trailing_zeros() as u8;

        // Copy parity shards into workspace, then recover polynomial coefficients
        workspace[0..T * shard_len].copy_from_slice(&received[0..T * shard_len]);
        K::ifft_sharded(&self.basis, workspace, shard_len, t_log, G::zero());
        let omega = self.basis.get_subspace_point_lut(T as u8);
        K::fft_sharded(&self.basis, workspace, shard_len, t_log, omega);

        received[T * shard_len..].copy_from_slice(&workspace[..T * shard_len]);
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
    pub fn decode_systematic_scalar(&self, received: &mut [G]) -> bool {
        if T == 0 {
            return true;
        }
        let t_log = T.trailing_zeros() as u8;

        // Step 1: syndrome
        let syndrome = self.compute_syndrome_scalar(received);
        if syndrome.iter().take(T).all(|&c| c == G::zero()) {
            return true;
        }

        // Step 2: key equation
        let (v1, lambda) = match self.solve_key_equation_hgcd(&syndrome, t_log) {
            Some(pair) => pair,
            None => {
                // TODO: error type enum
                return false;
            }
        };

        let deg_lambda = match lambda.degree() {
            Some(d) => d,
            None => {
                // Zero locator - no errors.
                return true;
            }
        };

        // Step 3: root-finding - one T-point FFT per chunk
        let mut error_indices: Vec<usize> = Vec::with_capacity(deg_lambda);

        'root: for chunk in 0..(N / T) {
            let beta = self.basis.get_subspace_point_lut((chunk * T) as u8);
            let mut evals = lambda;
            self.basis.fft_scalar(&mut evals[..T], t_log, beta);

            for offset in 0..T {
                if evals[offset] == G::zero() {
                    error_indices.push(chunk * T + offset);
                    if error_indices.len() == deg_lambda {
                        break 'root; // found all roots, stop scanning
                    }
                }
            }
        }

        if error_indices.len() < deg_lambda {
            return false; // too few roots, uncorrectable
        }

        // Step 4: error values - same per-chunk FFT structure as step 3
        let lambdap = poly::deriv_lnh(&lambda);

        for chunk in 0..(N / T) {
            let chunk_errors: Vec<(usize, usize)> = error_indices
                .iter()
                .filter(|&&g| g / T == chunk)
                .map(|&g| (g, g % T))
                .collect();

            if chunk_errors.is_empty() {
                continue;
            }

            let beta = self.basis.get_subspace_point_lut((chunk * T) as u8);

            let mut q_evals = v1;
            let mut lp_evals = lambdap;
            self.basis.fft_scalar(&mut q_evals[..T], t_log, beta);
            self.basis.fft_scalar(&mut lp_evals[..T], t_log, beta);

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

    pub(crate) fn erasure_locator_and_denominators(
        &self,
        erasure_positions: &[u8],
    ) -> ([G; N], [G; N]) {
        let erasure_count = erasure_positions.len();

        let mut pts = [G::zero(); N];
        for (k, &pos) in erasure_positions.iter().enumerate() {
            pts[k] = self.basis.get_subspace_point_lut(pos);
        }

        let mut lambda = [G::zero(); N];
        lambda[0] = G::one();
        for k in 0..erasure_count {
            let p = pts[k];
            for j in (1..=k + 1).rev() {
                lambda[j] = lambda[j - 1].add(p.mul_lut(lambda[j]));
            }
            lambda[0] = p.mul_lut(lambda[0]);
        }

        let mut denoms = [G::zero(); N];
        for j in 0..erasure_count {
            denoms[j] = (0..erasure_count)
                .filter(|&i| i != j)
                .fold(G::one(), |acc, i| acc.mul_lut(pts[j].add(pts[i])));
        }

        (lambda, denoms)
    }

    #[allow(unused)]
    pub(crate) fn forney_sharded(
        q: &[&[G]],
        erasure_positions: &[u8],
        denoms: &[G],
        out: &mut [&mut [G]], // one shard per erased position, in order
    ) {
        for (k, (&pos, &d)) in erasure_positions.iter().zip(denoms).enumerate() {
            K::scale(q[pos as usize], out[k], d.inv_lut());
        }
    }

    pub fn recover_erasure_shards(
        &self,
        received: &mut [G],
        workspace: &mut [G],
        shard_len: usize,
        erasure_positions: &[u8],
    ) -> bool {
        debug_assert_eq!(received.len(), N * shard_len);
        debug_assert_eq!(workspace.len(), N * shard_len);

        let e = erasure_positions.len();

        if e > T {
            return false;
        }
        if e == 0 || T == 0 {
            return true;
        }

        let t_log = T.trailing_zeros() as u8;
        let n_log = N.trailing_zeros() as u8;

        let (lambda, denoms) = self.erasure_locator_and_denominators(erasure_positions);

        // Chunk 0: parity block at ω_0
        workspace[..T * shard_len].copy_from_slice(&received[..T * shard_len]);
        K::ifft_sharded(
            &self.basis,
            &mut workspace[..T * shard_len],
            shard_len,
            t_log,
            G::zero(),
        );

        // Message chunks/shards, each shifted by one more ωT
        for chunk in 1..(N / T) {
            let omega = self.basis.get_subspace_point_lut((chunk * T) as u8);
            let (acc, rest) = workspace.split_at_mut(T * shard_len);
            let tmp = &mut rest[..T * shard_len];
            tmp.copy_from_slice(&received[chunk * T * shard_len..(chunk + 1) * T * shard_len]);
            K::ifft_sharded(&self.basis, tmp, shard_len, t_log, omega);
            for (w, t) in acc.iter_mut().zip(tmp.iter()) {
                *w = w.add(*t);
            }
        }

        workspace[T * shard_len..].fill(G::zero());

        // Horner evaluation of λ in the monomial basis at all N Cantor subspace points
        let mut lambda_evals = [G::zero(); N];
        for (i, u) in lambda_evals.iter_mut().enumerate() {
            let p = self.basis.get_subspace_point_lut(i as u8);
            let mut v = lambda[e]; // monic coefficient = G::one()
            for j in (0..e).rev() {
                v = v.mul_lut(p).add(lambda[j]);
            }
            *u = v;
        }

        // Evaluate s at all n points
        K::fft_sharded(&self.basis, workspace, shard_len, n_log, G::zero());

        // Pointwise multiply: work[i] := work[i] · λ(ω_i)
        for i in 0..N {
            K::scale_in_place(
                &mut workspace[i * shard_len..(i + 1) * shard_len],
                lambda_evals[i],
            );
        }

        // X-basis coefficients of (s·λ); q is in work[T .. T+e]
        K::ifft_sharded(&self.basis, workspace, shard_len, n_log, G::zero());

        // Shift q from work[T..T+e] down to work[0..e], zero everything else
        workspace.copy_within(T * shard_len..(T + e) * shard_len, 0);
        workspace[e * shard_len..].fill(G::zero());

        // Evaluate q at all n points
        K::fft_sharded(&self.basis, workspace, shard_len, n_log, G::zero());

        // (Forney) Eq 78: u(ω_i) = q(ω_i) / λ'(ω_i)
        for (&pos, d) in erasure_positions.iter().zip(denoms) {
            K::scale(
                &workspace[pos as usize * shard_len..(pos as usize + 1) * shard_len],
                &mut received[pos as usize * shard_len..(pos as usize + 1) * shard_len],
                d.inv_lut(),
            );
        }

        true
    }
}

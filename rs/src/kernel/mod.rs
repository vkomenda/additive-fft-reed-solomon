#[cfg(test)]
#[macro_use]
mod tests;

#[cfg(target_feature = "avx2")]
pub mod avx2_kernel;
#[cfg(avx512_gfni)]
pub mod gfni_kernel;
pub mod lut_kernel;
#[cfg(target_feature = "neon")]
pub mod neon_kernel;
pub mod node;
#[cfg(target_feature = "sse2")]
pub mod sse2_kernel;
#[cfg(target_feature = "ssse3")]
pub mod ssse3_kernel;

use crate::gf2p8lut::{Gf2p8Lut, SubspaceBasisLut};
use crate::poly_arith::PolyMutSliceArith;
use additive_fft_reed_solomon_gf2p8::Z255;
use node::{Dit2Node, Dit4Node, Node};

pub trait Kernel<G: Gf2p8Lut, B: SubspaceBasisLut<G>> {
    const ALIGN: usize;

    /// Multiplication by a fixed element of G. Plain, non-SIMD data type.
    type Multiplier: 'static + Copy;

    const NODE_MUL: &'static [Self::Multiplier; 510];

    fn mul_table(twiddle: G) -> Self::Multiplier;

    fn is_zero_mul(m: &Self::Multiplier) -> bool;

    fn butterfly_fwd_dit2(node: Dit2Node<'_, G>, m: Self::Multiplier);

    fn butterfly_fwd_dit2_zero(node: Dit2Node<'_, G>);

    /// Applies two butterfly levels to the node of `4 * d` shards beginning at shard `base`. The
    /// node consists of `d` independent groups. Group `i` consists of shards `base + i`, `base + i
    /// + d`, `base + i + 2 * d` and `base + i + 3 * d`. All groups share the multiplication tables
    /// `m01`, `m23` and `m02`.
    fn butterfly_fwd_dit4(
        node: Dit4Node<'_, G>,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    );

    /// An optimization of `butterfly_fwd_dit4` for the common special case `m01 == m02 == 0` that
    /// requires only 1 out of 4 multiply operations as well as fewer respective loads and stores.
    fn butterfly_fwd_dit4_zero(node: Dit4Node<'_, G>, m23: Self::Multiplier);

    fn butterfly_inv_dit2(node: Dit2Node<'_, G>, m: Self::Multiplier);

    fn butterfly_inv_dit4(
        node: Dit4Node<'_, G>,
        m01: Self::Multiplier,
        m23: Self::Multiplier,
        m02: Self::Multiplier,
    );

    fn butterfly_inv_dit4_zero(node: Dit4Node<'_, G>, m23: Self::Multiplier);

    /// Forward transform of 2^k shards whose input is nonzero only in the
    /// first 2^log_support. Shards past the support are overwritten and need
    /// not be initialised.
    fn fft_sharded_zero_padded(shards: &mut [G], shard_len: usize, k: u8, log_support: u8) {
        debug_assert!(log_support < k);
        let block = (1usize << log_support) * shard_len;

        // Every pruned level is a copy; together they replicate the first
        // block into all the others.
        let (first, rest) = shards.split_at_mut(block);
        for dst in rest.chunks_exact_mut(block) {
            dst.copy_from_slice(first);
        }

        // Each block is then an ordinary transform shifted by its offset.
        for (b, chunk) in shards.chunks_exact_mut(block).enumerate() {
            Self::fft_sharded_dit4_indexed(chunk, shard_len, log_support, b << log_support);
        }
    }

    /// FFT method selector.
    #[inline]
    fn fft_sharded(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        Self::fft_sharded_dit4_outer_tail(shards, shard_len, k, coords);
    }

    /// IFFT method selector.
    #[inline]
    fn ifft_sharded(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        Self::ifft_sharded_dit4_outer_tail(shards, shard_len, k, coords);
    }

    /// Forward transform that calls DIT4 transform on the outermost pairs of levels and DIT2
    /// transform on the innermost level if k is odd.
    fn fft_sharded_dit4_inner_tail(shards: &mut [G], shard_len: usize, k: u8, coords: usize);

    /// Forward transform that calls DIT2 transform on the innermost level if k is odd, and DIT4
    /// transform on the innermost pairs of levels.
    fn fft_sharded_dit4_outer_tail(shards: &mut [G], shard_len: usize, k: u8, coords: usize);

    /// Inverse transform. Comparison with the innermost DIT2 tail IFFT version is a research question
    /// which hasn't been settled yet.
    fn ifft_sharded_dit4_outer_tail(shards: &mut [G], shard_len: usize, k: u8, coords: usize);

    fn scale(src: &[G], dst: &mut [G], scalar: G);

    fn scale_in_place(dst: &mut [G], scalar: G);

    fn scale_add(src: &[G], dst: &mut [G], scalar: G);

    fn scale_by_log(src: &[G], dst: &mut [G], log_m: Z255);

    fn scale_in_place_by_log(dst: &mut [G], log_m: Z255);

    #[inline]
    fn twiddle(level: u8, beta: G) -> G {
        if level == 0 {
            beta
        } else {
            B::eval_subspace_poly_lut(level, beta)
        }
    }

    #[inline]
    fn fft_sharded_dit2(shards: &mut [G], shard_len: usize, k: u8, beta: G) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;

        for l in (0..k).rev() {
            let d = 1 << l;
            for base in (0..n).step_by(2 * d) {
                let node_beta = beta.add(B::get_subspace_point_lut(base as u8));
                Self::butterfly_fwd_dit2(
                    Node::new(shards, shard_len, base, d),
                    Self::mul_table(Self::twiddle(l, node_beta)),
                );
            }
        }
    }

    #[inline]
    fn fft_sharded_dit4(shards: &mut [G], shard_len: usize, k: u8, beta: G) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let node_beta = |start: usize| beta.add(B::get_subspace_point_lut(start as u8));
        let mt = |level: u8, beta: G| Self::mul_table(Self::twiddle(level, beta));

        let mut l = k;
        while l >= 2 {
            let hi = l - 1; // wide level, stride 2d
            let lo = l - 2; // narrow level, stride d
            let d = 1 << lo;
            for base in (0..n).step_by(4 * d) {
                let beta_l = node_beta(base);
                let beta_r = node_beta(base + 2 * d);
                let m01 = mt(lo, beta_l);
                let m23 = mt(lo, beta_r);
                let m02 = mt(hi, beta_l);
                let node = Node::new(shards, shard_len, base, d);
                if Self::is_zero_mul(&m01) && Self::is_zero_mul(&m02) {
                    Self::butterfly_fwd_dit4_zero(node, m23);
                } else {
                    Self::butterfly_fwd_dit4(node, m01, m23, m02);
                }
            }
            l -= 2;
        }

        if l == 1 {
            for base in (0..n).step_by(2) {
                let m = mt(0, node_beta(base));
                Self::butterfly_fwd_dit2(Node::new(shards, shard_len, base, 1), m);
            }
        }
    }

    /// FFT with the radix-2 last traversal order.
    #[inline]
    fn fft_sharded_dit4_inner_tail_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::Multiplier; 3]],
        tail_mul: &[Self::Multiplier],
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let mut body_mul = body_mul.iter();

        let mut l = k;
        while l >= 2 {
            let d = 1 << (l - 2);
            for base in (0..n).step_by(4 * d) {
                let [m01, m23, m02] = body_mul.next().unwrap();
                let node = Node::new(shards, shard_len, base, d);
                if Self::is_zero_mul(m01) && Self::is_zero_mul(m02) {
                    Self::butterfly_fwd_dit4_zero(node, *m23);
                } else {
                    Self::butterfly_fwd_dit4(node, *m01, *m23, *m02);
                }
            }
            l -= 2;
        }
        debug_assert!(body_mul.next().is_none());

        if l == 1 {
            debug_assert_eq!(tail_mul.len(), n / 2);
            for (base, m) in (0..n).step_by(2).zip(tail_mul) {
                Self::butterfly_fwd_dit2(Node::new(shards, shard_len, base, 1), *m);
            }
        }
    }

    /// FFT that mirrors the `ifft_sharded_dit4_with` outermost level radix-2 first traversal order
    /// and thus reuses IFFT body and tail Multiplier schedules.
    #[inline]
    fn fft_sharded_dit4_outer_tail_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::Multiplier; 3]], // IFFT schedule
        tail_mul: Option<&Self::Multiplier>,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;

        if k % 2 == 1 {
            let m = tail_mul.expect("odd k needs a tail multiplier");
            let d = n / 2;
            let node = Node::new(shards, shard_len, 0, d);
            if Self::is_zero_mul(m) {
                Self::butterfly_fwd_dit2_zero(node);
            } else {
                Self::butterfly_fwd_dit2(node, *m);
            }
        } else {
            debug_assert!(tail_mul.is_none());
        }

        let mut end = body_mul.len();
        let mut l = k as i32 - 2 - (k % 2) as i32;
        while l >= 0 {
            let d = 1 << l;
            let count = n / (4 * d);
            let round = &body_mul[end - count..end];
            for (base, [m01, m23, m02]) in (0..n).step_by(4 * d).zip(round) {
                let node = Node::new(shards, shard_len, base, d);
                if Self::is_zero_mul(m01) && Self::is_zero_mul(m02) {
                    Self::butterfly_fwd_dit4_zero(node, *m23);
                } else {
                    Self::butterfly_fwd_dit4(node, *m01, *m23, *m02);
                }
            }
            end -= count;
            l -= 2;
        }
        debug_assert_eq!(end, 0);
    }

    #[inline]
    fn ifft_sharded_dit2(shards: &mut [G], shard_len: usize, k: u8, beta: G) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;

        for l in 0..k {
            let d = 1 << l;
            for base in (0..n).step_by(2 * d) {
                let node_beta = beta.add(B::get_subspace_point_lut(base as u8));
                Self::butterfly_inv_dit2(
                    Node::new(shards, shard_len, base, d),
                    Self::mul_table(Self::twiddle(l, node_beta)),
                );
            }
        }
    }

    #[inline]
    fn ifft_sharded_dit4(shards: &mut [G], shard_len: usize, k: u8, beta: G) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let node_beta = |start: usize| beta.add(B::get_subspace_point_lut(start as u8));
        let mt = |level: u8, beta: G| Self::mul_table(Self::twiddle(level, beta));

        let mut lo = 0u8;
        while lo + 1 < k {
            let hi = lo + 1;
            let d = 1 << lo;
            for base in (0..n).step_by(4 * d) {
                let beta_l = node_beta(base);
                let beta_r = node_beta(base + 2 * d);
                let m01 = mt(lo, beta_l);
                let m23 = mt(lo, beta_r);
                let m02 = mt(hi, beta_l);
                let node = Node::new(shards, shard_len, base, d);
                if Self::is_zero_mul(&m01) && Self::is_zero_mul(&m02) {
                    Self::butterfly_inv_dit4_zero(node, m23);
                } else {
                    Self::butterfly_inv_dit4(node, m01, m23, m02);
                }
            }
            lo += 2;
        }

        if lo + 1 == k {
            let m = mt(lo, beta);
            Self::butterfly_inv_dit2(Node::new(shards, shard_len, 0, n / 2), m);
        }
    }

    /// Common IFFT driver for Multiplier schedules that match the radix-2 first traversal order.
    #[inline]
    fn ifft_sharded_dit4_outer_tail_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::Multiplier; 3]],
        tail_mul: Option<&Self::Multiplier>,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let mut body_mul = body_mul.iter();

        let mut lo = 0u8;
        while lo + 1 < k {
            let d = 1 << lo;
            for base in (0..n).step_by(4 * d) {
                let [m01, m23, m02] = body_mul.next().unwrap();
                let node = Node::new(shards, shard_len, base, d);
                if Self::is_zero_mul(m01) && Self::is_zero_mul(m02) {
                    Self::butterfly_inv_dit4_zero(node, *m23);
                } else {
                    Self::butterfly_inv_dit4(node, *m01, *m23, *m02);
                };
            }
            lo += 2;
        }
        debug_assert!(body_mul.next().is_none());

        if lo + 1 == k {
            let m = tail_mul.expect("odd k needs a tail multiplier");
            Self::butterfly_inv_dit2(Node::new(shards, shard_len, 0, n / 2), *m);
        } else {
            debug_assert!(tail_mul.is_none());
        }
    }

    /// Outer-tail indexed FFT - using basis coordinates instead of a subspace point.
    #[inline]
    fn fft_sharded_dit4_indexed(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        let n = 1 << k;
        let m = |l: u8, x: usize| &Self::NODE_MUL[node_mul_index(l, x)];

        if k % 2 == 1 {
            let hi = k - 1;
            let d = n / 2;
            let node = Node::new(shards, shard_len, 0, d);
            if coords < (1 << hi) {
                Self::butterfly_fwd_dit2_zero(node);
            } else {
                Self::butterfly_fwd_dit2(node, *m(hi, coords));
            }
        }

        if k >= 2 {
            let top = k - 2 - (k % 2);
            for lo in (0..=top).rev().step_by(2) {
                let d = 1usize << lo;
                for base in (0..n).step_by(4 * d) {
                    let x = coords ^ base;
                    let node = Node::new(shards, shard_len, base, d);
                    if x < d {
                        Self::butterfly_fwd_dit4_zero(node, *m(lo, x ^ (2 * d)));
                    } else {
                        let (m01, m23, m02) = (m(lo, x), m(lo, x ^ (2 * d)), m(lo + 1, x));
                        Self::butterfly_fwd_dit4(node, *m01, *m23, *m02);
                    }
                }
            }
        }
    }

    /// Outer-tail indexed IFFT - using basis coordinates instead of a subspace point.
    #[inline]
    fn ifft_sharded_dit4_indexed(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        let n = 1usize << k;
        let m = |l: u8, x: usize| &Self::NODE_MUL[node_mul_index(l, x)];

        let mut lo = 0u8;
        while lo + 1 < k {
            let d = 1usize << lo;
            for base in (0..n).step_by(4 * d) {
                let x = coords ^ base;
                let node = Node::new(shards, shard_len, base, d);
                if x < d {
                    Self::butterfly_inv_dit4_zero(node, *m(lo, x ^ (2 * d)));
                } else {
                    let (m01, m23, m02) = (m(lo, x), m(lo, x ^ (2 * d)), m(lo + 1, x));
                    Self::butterfly_inv_dit4(node, *m01, *m23, *m02);
                };
            }
            lo += 2;
        }
        if lo + 1 == k {
            Self::butterfly_inv_dit2(Node::new(shards, shard_len, 0, n / 2), *m(lo, coords));
        }
    }

    #[inline]
    fn fft_sharded_dit2_indexed(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        let n = 1 << k;
        for l in (0..k).rev() {
            let d = 1 << l;
            for base in (0..n).step_by(2 * d) {
                let x = coords ^ base;
                Self::butterfly_fwd_dit2(
                    Node::new(shards, shard_len, base, d),
                    Self::NODE_MUL[node_mul_index(l, x)],
                );
            }
        }
    }

    #[inline]
    fn ifft_sharded_dit2_indexed(shards: &mut [G], shard_len: usize, k: u8, coords: usize) {
        let n = 1usize << k;
        for l in 0..k {
            let d = 1usize << l;
            for base in (0..n).step_by(2 * d) {
                let x = coords ^ base;
                Self::butterfly_inv_dit2(
                    Node::new(shards, shard_len, base, d),
                    Self::NODE_MUL[node_mul_index(l, x)],
                );
            }
        }
    }

    fn formal_derivative(shards: &mut [G], shard_len: usize, n_log: u8) {
        let n = 1usize << n_log;
        debug_assert_eq!(shards.len(), n * shard_len);

        if B::UNIT_DERIVS {
            // In a Cantor basis every subspace polynomial derivative is one, so this becomes an XOR
            // linear map.
            for u in 1..n {
                let w = 1 << u.trailing_zeros();
                let (lo, hi) = shards.split_at_mut(u * shard_len);
                lo[(u - w) * shard_len..].poly_add_in_place(&hi[..w * shard_len]);
            }
        } else {
            let derivs = B::subspace_poly_deriv_lut();
            for u in 0..n {
                let dst = u * shard_len;
                let mut first = true;
                for i in 0..n_log {
                    if u >> i & 1 == 1 {
                        continue;
                    }
                    let src = (u | 1 << i) * shard_len;
                    let (lo, hi) = shards.split_at_mut(src);
                    let (d, s) = (&mut lo[dst..dst + shard_len], &hi[..shard_len]);
                    if first {
                        Self::scale(s, d, derivs[i as usize]); // d = c_i · s
                        first = false;
                    } else {
                        Self::scale_add(s, d, derivs[i as usize]); // d ^= c_i · s
                    }
                }
                if first {
                    shards[dst..dst + shard_len].fill(G::zero()); // u = n-1 has no terms
                }
            }
        }
    }
}

/// Multiplier for level l node whose first shard has Cantor subspace index x. W_l(ω_x) depends only
/// on x >> l, so level l needs 256 >> l entries.
#[inline]
fn node_mul_index(l: u8, x: usize) -> usize {
    (512 - (512 >> l)) + (x >> l)
}

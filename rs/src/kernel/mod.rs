#[cfg(any(native_avx2, feature = "compile_avx2"))]
pub mod avx2_kernel;
#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub mod gfni_kernel;
pub mod lut_kernel;
#[cfg(native_neon)]
pub mod neon_kernel;

use additive_fft_reed_solomon_gf2p8::Z255;

use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};

pub trait Kernel<G: Gf2p8Lut> {
    /// Shard buffer byte alignment multiple to maximize SIMD performance. Every buffer must start
    /// at an address which is a multiple of this constant. The length of every shard in the buffer
    /// must be a multiple of this constant. Unaligned buffers will work but may be noticeably
    /// slower.
    const ALIGN: usize;

    /// Multiplication by a fixed element of G. Plain, non-SIMD data type.
    type MulTable: Copy;

    fn mul_table(twiddle: G) -> Self::MulTable;

    fn butterfly_fwd_dit2(
        shards: &mut [G],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::MulTable,
    );

    /// Applies two butterfly levels to the node of `4 * d` shards beginning at shard `base`. The
    /// node consists of `d` independent groups. Group `i` consists of shards `base + i`, `base + i
    /// + d`, `base + i + 2 * d` and `base + i + 3 * d`. All groups share the multiplication tables
    /// `m01`, `m23` and `m02`.
    fn butterfly_fwd_dit4(
        shards: &mut [G],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::MulTable,
        m23: Self::MulTable,
        m02: Self::MulTable,
    );

    fn butterfly_inv_dit2(
        shards: &mut [G],
        shard_len: usize,
        base: usize,
        d: usize,
        m: Self::MulTable,
    );

    fn butterfly_inv_dit4(
        shards: &mut [G],
        shard_len: usize,
        base: usize,
        d: usize,
        m01: Self::MulTable,
        m23: Self::MulTable,
        m02: Self::MulTable,
    );

    /// Forward transform.
    fn fft_sharded(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    );

    /// Forward transform where shards [1 << log_support..] are treated as zeros.
    fn fft_sharded_zero_padded(shards: &mut [G], shard_len: usize, k: u8, log_support: u8);

    /// Inverse transform.
    fn ifft_sharded(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    );

    fn scale(src: &[G], dst: &mut [G], scalar: G);

    fn scale_in_place(dst: &mut [G], scalar: G);

    fn scale_by_log(src: &[G], dst: &mut [G], log_m: Z255);

    fn scale_in_place_by_log(dst: &mut [G], log_m: Z255);

    #[inline]
    fn twiddle(basis: &impl CantorBasisLut<G>, level: u8, beta: G) -> G {
        if level == 0 {
            beta
        } else {
            basis.eval_subspace_poly_lut(level, beta)
        }
    }

    fn fft_sharded_dit2(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);

        // betas[b] is the shift for block b at the current level pair.
        let mut betas = [G::zero(); 256];
        betas[0] = beta;
        let mut blocks = 1;

        let mut l = k;
        while l >= 1 {
            let hi = l - 1;
            let d = 1 << hi;
            let step = 1 << l; // block span, in shards
            let basis_hi = basis.get_basis_point_lut(hi);

            for b in (0..blocks).rev() {
                let beta_l = betas[b];
                let beta_r = beta_l.add(basis_hi);
                let start = b * step;
                for i in 0..d {
                    Self::butterfly_fwd_dit2(
                        shards,
                        shard_len,
                        start + i,
                        d,
                        Self::mul_table(Self::twiddle(basis, hi, beta_l)),
                    );
                }

                betas[2 * b] = beta_l;
                betas[2 * b + 1] = beta_r;
            }

            blocks <<= 1;
            l -= 1;
        }
    }

    fn fft_sharded_dit4(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);

        // betas[b] is the shift for block b at the current level pair.
        let mut betas = [G::zero(); 256];
        betas[0] = beta;
        let mut blocks = 1;

        let mut l = k;
        while l >= 2 {
            let hi = l - 1; // wide level, stride 2d
            let lo = l - 2; // narrow level, stride d
            let d = 1 << lo;
            let step = 1 << l; // block span, in shards
            let basis_hi = basis.get_basis_point_lut(hi);
            let basis_lo = basis.get_basis_point_lut(lo);

            for b in (0..blocks).rev() {
                let beta_l = betas[b];
                let beta_r = beta_l.add(basis_hi);
                let start = b * step;
                for i in 0..d {
                    Self::butterfly_fwd_dit4(
                        shards,
                        shard_len,
                        start + i,
                        d,
                        Self::mul_table(Self::twiddle(basis, lo, beta_l)),
                        Self::mul_table(Self::twiddle(basis, lo, beta_r)),
                        Self::mul_table(Self::twiddle(basis, hi, beta_l)),
                    );
                }

                betas[4 * b] = beta_l;
                betas[4 * b + 1] = beta_l.add(basis_lo);
                betas[4 * b + 2] = beta_r;
                betas[4 * b + 3] = beta_r.add(basis_lo);
            }

            blocks <<= 2;
            l -= 2;
        }

        if l == 1 {
            for b in 0..blocks {
                let m = Self::mul_table(Self::twiddle(basis, 0, betas[b]));
                Self::butterfly_fwd_dit2(&mut shards[..], shard_len, 2 * b, 1, m);
            }
        }
    }
}

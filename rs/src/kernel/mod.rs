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

    fn fft_sharded_iterated(shards: &mut [G], shard_len: usize, k: u8, beta: G);

    fn ifft_sharded_iterated(shards: &mut [G], shard_len: usize, k: u8, beta: G);

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
        let n = 1 << k;

        for l in (0..k).rev() {
            let d = 1 << l;
            for start in (0..n).step_by(2 * d) {
                let node_beta = beta.add(basis.get_subspace_point_lut(start as u8));
                Self::butterfly_fwd_dit2(
                    shards,
                    shard_len,
                    start,
                    d,
                    Self::mul_table(Self::twiddle(basis, l, node_beta)),
                );
            }
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
        let n = 1 << k;
        let node_beta = |start: usize| beta.add(basis.get_subspace_point_lut(start as u8));

        let mut l = k;
        while l >= 2 {
            let hi = l - 1; // wide level, stride 2d
            let lo = l - 2; // narrow level, stride d
            let d = 1 << lo;
            for start in (0..n).step_by(4 * d) {
                let beta_l = node_beta(start);
                let beta_r = node_beta(start + 2 * d);
                Self::butterfly_fwd_dit4(
                    shards,
                    shard_len,
                    start,
                    d,
                    Self::mul_table(Self::twiddle(basis, lo, beta_l)),
                    Self::mul_table(Self::twiddle(basis, lo, beta_r)),
                    Self::mul_table(Self::twiddle(basis, hi, beta_l)),
                );
            }
            l -= 2;
        }

        if l == 1 {
            for start in (0..n).step_by(2) {
                let m = Self::mul_table(Self::twiddle(basis, 0, node_beta(start)));
                Self::butterfly_fwd_dit2(shards, shard_len, start, 1, m);
            }
        }
    }

    fn fft_sharded_dit4_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::MulTable; 3]],
        tail_mul: &[Self::MulTable],
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let mut body_mul = body_mul.iter();

        let mut l = k;
        while l >= 2 {
            let d = 1 << (l - 2);
            for start in (0..n).step_by(4 * d) {
                let [m01, m23, m02] = body_mul.next().unwrap();
                Self::butterfly_fwd_dit4(shards, shard_len, start, d, *m01, *m23, *m02);
            }
            l -= 2;
        }
        debug_assert!(body_mul.next().is_none());

        if l == 1 {
            debug_assert_eq!(tail_mul.len(), n / 2);
            for (start, m) in (0..n).step_by(2).zip(tail_mul) {
                Self::butterfly_fwd_dit2(shards, shard_len, start, 1, *m);
            }
        }
    }

    fn fft_sharded_dit4_mirrored_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::MulTable; 3]], // IFFT schedule
        tail_mul: Option<&Self::MulTable>,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;

        if k % 2 == 1 {
            let m = *tail_mul.expect("odd k needs a tail multiplier");
            Self::butterfly_fwd_dit2(shards, shard_len, 0, n / 2, m);
        } else {
            debug_assert!(tail_mul.is_none());
        }

        let mut end = body_mul.len();
        let mut l = k as i32 - 2 - (k % 2) as i32;
        while l >= 0 {
            let d = 1 << l;
            let count = n / (4 * d);
            let round = &body_mul[end - count..end];
            for (start, [m01, m23, m02]) in (0..n).step_by(4 * d).zip(round) {
                Self::butterfly_fwd_dit4(shards, shard_len, start, d, *m01, *m23, *m02);
            }
            end -= count;
            l -= 2;
        }
        debug_assert_eq!(end, 0);
    }

    fn ifft_sharded_dit2(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;

        for l in 0..k {
            let d = 1 << l;
            for start in (0..n).step_by(2 * d) {
                let node_beta = beta.add(basis.get_subspace_point_lut(start as u8));
                Self::butterfly_inv_dit2(
                    shards,
                    shard_len,
                    start,
                    d,
                    Self::mul_table(Self::twiddle(basis, l, node_beta)),
                );
            }
        }
    }

    fn ifft_sharded_dit4(
        basis: &impl CantorBasisLut<G>,
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        beta: G,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let node_beta = |start: usize| beta.add(basis.get_subspace_point_lut(start as u8));

        let mut lo = 0u8;
        while lo + 1 < k {
            let hi = lo + 1;
            let d = 1 << lo;
            for start in (0..n).step_by(4 * d) {
                let beta_l = node_beta(start);
                let beta_r = node_beta(start + 2 * d);
                Self::butterfly_inv_dit4(
                    shards,
                    shard_len,
                    start,
                    d,
                    Self::mul_table(Self::twiddle(basis, lo, beta_l)),
                    Self::mul_table(Self::twiddle(basis, lo, beta_r)),
                    Self::mul_table(Self::twiddle(basis, hi, beta_l)),
                );
            }
            lo += 2;
        }

        if lo + 1 == k {
            let m = Self::mul_table(Self::twiddle(basis, lo, beta));
            Self::butterfly_inv_dit2(shards, shard_len, 0, n / 2, m);
        }
    }

    fn ifft_sharded_dit4_with(
        shards: &mut [G],
        shard_len: usize,
        k: u8,
        body_mul: &[[Self::MulTable; 3]],
        tail_mul: Option<&Self::MulTable>,
    ) {
        debug_assert_eq!(shards.len(), (1 << k) * shard_len);
        let n = 1 << k;
        let mut body_mul = body_mul.iter();

        let mut lo = 0u8;
        while lo + 1 < k {
            let d = 1 << lo;
            for start in (0..n).step_by(4 * d) {
                let [m01, m23, m02] = body_mul.next().unwrap();
                Self::butterfly_inv_dit4(shards, shard_len, start, d, *m01, *m23, *m02);
            }
            lo += 2;
        }
        debug_assert!(body_mul.next().is_none());

        if lo + 1 == k {
            let m = tail_mul.expect("odd k needs a tail multiplier");
            Self::butterfly_inv_dit2(shards, shard_len, 0, n / 2, *m);
        } else {
            debug_assert!(tail_mul.is_none());
        }
    }
}

/// Splits the node of `R * d` shards starting at shard `base` into `R` equal
/// parts and yields the `d` groups. Group `i` holds shard `i` of each part,
/// i.e. shards `base + i + j * d` for `j` in `0..R`.
pub(crate) fn shard_groups<G, const R: usize>(
    shards: &mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
) -> impl Iterator<Item = [&mut [G]; R]> {
    let q = d * shard_len;
    let node = &mut shards[base * shard_len..][..R * q];
    let mut parts = node
        .chunks_exact_mut(q)
        .map(|p| p.chunks_exact_mut(shard_len));
    let mut parts: [_; R] = std::array::from_fn(|_| parts.next().unwrap());
    (0..d).map(move |_| parts.each_mut().map(|p| p.next().unwrap()))
}

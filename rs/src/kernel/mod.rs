#[cfg(any(native_avx2, feature = "compile_avx2"))]
pub mod avx2_kernel;
#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub mod gfni_kernel;
pub mod lut_kernel;
#[cfg(native_neon)]
pub mod neon_kernel;

use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};

pub trait Kernel<G: Gf2p8Lut> {
    /// Shard buffer byte alignment multiple to maximize SIMD performance. Every buffer must start
    /// at an address which is a multiple of this constant. The length of every shard in the buffer
    /// must be a multiple of this constant. Unaligned buffers will work but may be noticeably
    /// slower.
    const ALIGN: usize;

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
}

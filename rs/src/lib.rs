mod codec;
mod gf2p8lut;
mod kernel;
mod poly_11d_lut;

//use crate::kernel::Kernel;
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;
use codec::Codec;
use kernel::{gfni_kernel::GfniKernel, lut_kernel::LutKernel};
use poly_11d_lut::BasesLut11d;

/// Reed-Solomon codec interface type with precomputed lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
pub type RsLut<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, BasesLut11d, LutKernel<Gf2p8_11d>, N, T>;

/// Reed-Solomon codec interface type accelerated with GFNI instructions and using the precomputed
/// lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
pub type RsGfni<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, BasesLut11d, GfniKernel<Gf2p8_11d>, N, T>;

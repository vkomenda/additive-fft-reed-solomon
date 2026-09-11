pub mod codec;
pub mod gf2p8lut;
pub mod kernel;
pub mod poly_11d_lut;
pub mod poly_arith;

use codec::Codec;
#[cfg(any(native_avx2, feature = "compile_avx2"))]
use kernel::avx2_kernel::Avx2Kernel;
#[cfg(any(native_gfni, feature = "compile_gfni"))]
use kernel::gfni_kernel::GfniKernel;
use kernel::lut_kernel::LutKernel;
#[cfg(native_neon)]
use kernel::neon_kernel::NeonKernel;
use poly_11d_lut::CantorBasisLut11d;

pub use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};

/// Reed-Solomon codec interface type with precomputed lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
pub type RsLut<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, LutKernel<Gf2p8_11d>, N, T>;

/// Reed-Solomon codec interface type accelerated with GFNI instructions and using the precomputed
/// lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub type RsGfni<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, GfniKernel<Gf2p8_11d>, N, T>;

#[cfg(any(native_avx2, feature = "compile_avx2"))]
pub type RsAvx2<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, Avx2Kernel<Gf2p8_11d>, N, T>;

#[cfg(native_neon)]
pub type RsNeon<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, NeonKernel<Gf2p8_11d>, N, T>;

cfg_if::cfg_if! {
    if #[cfg(feature = "compile_gfni")] {
        pub type Rs<const N: usize, const T: usize> = RsGfni<N, T>;
    } else if #[cfg(feature = "compile_avx2")] {
        pub type Rs<const N: usize, const T: usize> = RsAvx2<N, T>;
    } else if #[cfg(native_gfni)] {
        pub type Rs<const N: usize, const T: usize> = RsGfni<N, T>;
    } else if #[cfg(native_avx2)] {
        pub type Rs<const N: usize, const T: usize> = RsAvx2<N, T>;
    } else if #[cfg(native_neon)] {
        pub type Rs<const N: usize, const T: usize> = RsNeon<N, T>;
    } else {
        pub type Rs<const N: usize, const T: usize> = RsLut<N, T>;
    }
}

pub mod codec;
pub mod gf2p8lut;
pub mod kernel;
pub mod poly_11d_lut;
pub mod poly_arith;

use codec::Codec;
#[cfg(any(native_gfni, feature = "compile_gfni"))]
use kernel::gfni_kernel::GfniKernel;
use kernel::lut_kernel::LutKernel;
use poly_11d_lut::CantorBasisLut11d;

pub use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;

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

cfg_if::cfg_if! {
    if #[cfg(feature = "compile_gfni")] {
        pub type Rs<const N: usize, const T: usize> = RsGfni<N, T>;
    // } else if #[cfg(feature = "compile_avx2")] {
    //     todo!();
    } else if #[cfg(native_gfni)] {
        pub type Rs<const N: usize, const T: usize> = RsGfni<N, T>;
    // } else if #[cfg(native_avx2)] {
    //     todo!();
    } else {
        pub type Rs<const N: usize, const T: usize> = RsLut<N, T>;
    }
}

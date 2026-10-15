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
use poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d};

pub use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};

/// Aligned buffer type helper to make good use of SIMD ops and avoid memory accesses across
/// cache-line boundaries. Specifically useful for shard buffers.
#[repr(align(64))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedArray<G, const L: usize>([G; L]);

/// Reed-Solomon codec interface type with precomputed lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
pub type RsLutCantor<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, LutKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

pub type RsLutStandard<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, StandardBasisLut11d, LutKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

/// Reed-Solomon codec interface type accelerated with GFNI instructions and using the precomputed
/// lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub type RsGfniCantor<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, GfniKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub type RsGfniStandard<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, StandardBasisLut11d, GfniKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(any(native_avx2, feature = "compile_avx2"))]
pub type RsAvx2Cantor<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(any(native_avx2, feature = "compile_avx2"))]
pub type RsAvx2Standard<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, StandardBasisLut11d, Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(native_neon)]
pub type RsNeonCantor<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, NeonKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(native_neon)]
pub type RsNeonStandard<const N: usize, const T: usize> =
    Codec<Gf2p8_11d, StandardBasisLut11d, NeonKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

cfg_if::cfg_if! {
    if #[cfg(feature = "compile_gfni")] {
        pub type RsCantor<const N: usize, const T: usize> = RsGfniCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsGfniStandard<N, T>;
    } else if #[cfg(feature = "compile_avx2")] {
        pub type RsCantor<const N: usize, const T: usize> = RsAvx2Cantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsAvx2Standard<N, T>;
    } else if #[cfg(native_gfni)] {
        pub type RsCantor<const N: usize, const T: usize> = RsGfniCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsGfniStandard<N, T>;
    } else if #[cfg(native_avx2)] {
        pub type RsCantor<const N: usize, const T: usize> = RsAvx2Cantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsAvx2Standard<N, T>;
    } else if #[cfg(native_neon)] {
        pub type RsCantor<const N: usize, const T: usize> = RsNeonCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsNeonStandard<N, T>;
    } else {
        pub type RsCantor<const N: usize, const T: usize> = RsLutCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsLutStandard<N, T>;
    }
}

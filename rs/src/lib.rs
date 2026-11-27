mod codec;
mod gf2p8lut;
mod kernel;
mod poly_11d_lut;
mod poly_arith;

#[cfg(feature = "unstable-internals")]
#[doc(hidden)]
pub mod unstable {
    pub use crate::codec::ShardLayout;
    pub use crate::gf2p8lut::SubspaceBasisLut;
    pub use crate::kernel::{Kernel, lut_kernel::LutKernel};

    #[cfg(target_feature = "avx2")]
    pub use crate::kernel::avx2_kernel::Avx2Kernel;
    #[cfg(avx512_gfni)]
    pub use crate::kernel::gfni_kernel::GfniKernel;
    #[cfg(neon)]
    pub use crate::kernel::neon_kernel::NeonKernel;
    #[cfg(target_feature = "sse2")]
    pub use crate::kernel::sse2_kernel::Sse2Kernel;
    #[cfg(target_feature = "ssse3")]
    pub use crate::kernel::ssse3_kernel::Ssse3Kernel;
}

#[cfg(target_feature = "avx2")]
use kernel::avx2_kernel::Avx2Kernel;
#[cfg(avx512_gfni)]
use kernel::gfni_kernel::GfniKernel;
use kernel::lut_kernel::LutKernel;
#[cfg(neon)]
use kernel::neon_kernel::NeonKernel;
#[cfg(target_feature = "sse2")]
use kernel::sse2_kernel::Sse2Kernel;
#[cfg(target_feature = "ssse3")]
use kernel::ssse3_kernel::Ssse3Kernel;

pub use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
pub use codec::{Codec, ParityFirst, ParityLast};
pub use poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d};

pub(crate) type CodecCantor<K, const N: usize, const T: usize> =
    Codec<Gf2p8_11d, CantorBasisLut11d, K, ParityFirst, N, T>;

pub(crate) type CodecStandard<K, const N: usize, const T: usize> =
    Codec<Gf2p8_11d, StandardBasisLut11d, K, ParityLast, N, T>;

/// Reed-Solomon codec interface type with precomputed lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
pub type RsLutCantor<const N: usize, const T: usize> =
    CodecCantor<LutKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

pub type RsLutStandard<const N: usize, const T: usize> =
    CodecStandard<LutKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

/// Reed-Solomon codec interface type accelerated with GFNI instructions and using the precomputed
/// lookup tables.
///
/// ## Arguments
/// - N ≤ 256, is a power of 2.
/// - 1 ≤ T < N, is a power of 2 as well.
#[cfg(avx512_gfni)]
pub type RsGfniCantor<const N: usize, const T: usize> =
    CodecCantor<GfniKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(avx512_gfni)]
pub type RsGfniStandard<const N: usize, const T: usize> =
    CodecStandard<GfniKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(target_feature = "avx2")]
pub type RsAvx2Cantor<const N: usize, const T: usize> =
    CodecCantor<Avx2Kernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(target_feature = "avx2")]
pub type RsAvx2Standard<const N: usize, const T: usize> =
    CodecStandard<Avx2Kernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(target_feature = "ssse3")]
pub type RsSsse3Cantor<const N: usize, const T: usize> =
    CodecCantor<Ssse3Kernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(target_feature = "ssse3")]
pub type RsSsse3Standard<const N: usize, const T: usize> =
    CodecStandard<Ssse3Kernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(target_feature = "sse2")]
pub type RsSse2Cantor<const N: usize, const T: usize> =
    CodecCantor<Sse2Kernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(target_feature = "sse2")]
pub type RsSse2Standard<const N: usize, const T: usize> =
    CodecStandard<Sse2Kernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

#[cfg(neon)]
pub type RsNeonCantor<const N: usize, const T: usize> =
    CodecCantor<NeonKernel<Gf2p8_11d, CantorBasisLut11d>, N, T>;

#[cfg(neon)]
pub type RsNeonStandard<const N: usize, const T: usize> =
    CodecStandard<NeonKernel<Gf2p8_11d, StandardBasisLut11d>, N, T>;

cfg_if::cfg_if! {
    if #[cfg(avx512_gfni)] {
        pub type RsCantor<const N: usize, const T: usize> = RsGfniCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsGfniStandard<N, T>;
    } else if #[cfg(target_feature = "avx2")] {
        pub type RsCantor<const N: usize, const T: usize> = RsAvx2Cantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsAvx2Standard<N, T>;
    } else if #[cfg(target_feature = "ssse3")] {
        pub type RsCantor<const N: usize, const T: usize> = RsSsse3Cantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsSsse3Standard<N, T>;
    } else if #[cfg(target_feature = "sse2")] {
        pub type RsCantor<const N: usize, const T: usize> = RsSse2Cantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsSse2Standard<N, T>;
    } else if #[cfg(neon)] {
        pub type RsCantor<const N: usize, const T: usize> = RsNeonCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsNeonStandard<N, T>;
    } else {
        pub type RsCantor<const N: usize, const T: usize> = RsLutCantor<N, T>;
        pub type RsStandard<const N: usize, const T: usize> = RsLutStandard<N, T>;
    }
}

#[cfg(any(native_gfni, feature = "compile_gfni"))]
pub mod gfni_kernel;
pub mod lut_kernel;

use crate::gf2p8lut::{CantorBasisLut, Gf2p8Lut};

pub trait Kernel<G: Gf2p8Lut> {
    fn fft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G);
    fn ifft_sharded(basis: &impl CantorBasisLut<G>, shards: &mut [&mut [G]], k: u8, beta: G);
    fn scale(dst: &mut [G], src: &[G], scalar: G);
    fn scale_in_place(dst: &mut [G], scalar: G);
}

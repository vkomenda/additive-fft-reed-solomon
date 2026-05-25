use crate::gf2p8::generic::{CantorBasisLut, Gf2p8Lut};

pub trait Kernel<G: Gf2p8Lut>: CantorBasisLut<G> {
    fn fft_sharded(&self, shards: &mut [&mut [G]], k: u8, beta: G);
    fn ifft_sharded(&self, shards: &mut [&mut [G]], k: u8, beta: G);
    fn scale(dst: &mut [G], src: &[G], scalar: G);
}

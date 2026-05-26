mod codec;
mod gf2p8lut;
mod kernel;
mod poly_11d_lut;

//use crate::kernel::Kernel;
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;
// use codec::Codec;
use poly_11d_lut::BasesLut11d;

// /// Reed-Solomon codec interface type. N ≤ 256 and is a power of 2. 1 ≤ T < N and is a power of 2 as
// /// well.
// pub type Rs<const N: usize, const T: usize> = Codec<BasesLut11d, Gf2p8_11d, N, T>;

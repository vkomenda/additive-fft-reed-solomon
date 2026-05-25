use crate::{
    gf2p8::generic::{Gf2p8Lut, LchBasisLut},
    kernel::Kernel,
};
use std::marker::PhantomData;

pub struct Codec<B, G, const N: usize, const T: usize>(B, PhantomData<G>);

impl<B, G, const N: usize, const T: usize> Codec<B, G, N, T>
where
    G: Gf2p8Lut,
    B: Kernel<G>,
{
    pub fn new(basis: B) -> Self {
        Self(basis, PhantomData)
    }
}

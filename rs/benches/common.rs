use additive_fft_reed_solomon::{codec::Codec, gf2p8lut::CantorBasisLut, kernel::Kernel};
use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};
use rand::Rng;
use rand::rngs::SmallRng;
use std::alloc::{Layout, alloc};

pub fn aligned_buffer(len: usize) -> Vec<Gf2p8_11d> {
    let layout = Layout::from_size_align(len, 64).unwrap();
    let buf = unsafe { alloc(layout) };
    let codeword: Vec<Gf2p8_11d> = unsafe { Vec::from_raw_parts(buf as *mut Gf2p8_11d, len, len) };
    codeword
}

fn create_buffer(
    num_shards: usize,
    shard_len: usize,
    rng: Option<&mut SmallRng>,
    is_aligned: bool,
) -> (Vec<Gf2p8_11d>, usize) {
    let mut backing = vec![Gf2p8_11d::zero(); (num_shards + 1) * shard_len];
    let aligned_off = (64 - (backing.as_ptr() as usize % 64)) % 64;
    let start = if is_aligned {
        aligned_off
    } else {
        aligned_off + 1
    };

    if let Some(rng) = rng {
        let buffer = &mut backing[start..][..num_shards * shard_len];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, buffer.len()) };
        rng.fill_bytes(bytes);
    }

    (backing, start)
}

// pub fn create_buffer(
//     num_shards: usize,
//     shard_len: usize,
//     rng: &mut impl Rng,
//     is_aligned: bool,
// ) -> (Vec<Gf2p8_11d>, usize) {
//     let mut backing = vec![Gf2p8_11d::zero(); (num_shards + 1) * shard_len];
//     let aligned_off = (64 - (backing.as_ptr() as usize % 64)) % 64;
//     let start = if is_aligned {
//         aligned_off
//     } else {
//         aligned_off + 1
//     };

//     let buffer = &mut backing[start..][..num_shards * shard_len];
//     let bytes =
//         unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, buffer.len()) };
//     rng.fill_bytes(bytes);

//     (backing, start)
// }

pub fn generate_random_codeword<B, K, const N: usize, const T: usize>(
    rs: &Codec<Gf2p8_11d, B, K, N, T>,
    shard_len: usize,
    rng: &mut impl Rng,
    is_aligned: bool,
) -> (Vec<Gf2p8_11d>, usize)
where
    B: CantorBasisLut<Gf2p8_11d> + Default,
    K: Kernel<Gf2p8_11d>,
{
    let k = N - T;
    let message_len = k * shard_len;
    let mut message = vec![Gf2p8_11d::zero(); message_len];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(message.as_mut_ptr() as *mut u8, message.len()) };
    rng.fill_bytes(bytes);

    let parity_len = T * shard_len;
    let mut parity = vec![Gf2p8_11d::zero(); parity_len];
    let mut workspace = vec![Gf2p8_11d::zero(); parity_len];

    rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, shard_len);

    let (mut codeword, start) = create_buffer(N, shard_len, rng, is_aligned);

    codeword[start..start + parity_len].clone_from_slice(&parity);
    codeword[start + parity_len..start + parity_len + message_len].clone_from_slice(&message);
    (codeword, start)
}

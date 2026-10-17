//! Cross-checks the ParityLast + StandardBasisLut11d encoder against reed-solomon-erasure, and prints
//! the parity as a golden fixture.  Run with: cargo test --test rse_compat -- --nocapture

use additive_fft_reed_solomon::{Gf2p8, Gf2p8_11d, RsLutStandard};
use reed_solomon_erasure::galois_8::ReedSolomon;

const DATA: usize = 32;
const PARITY: usize = 32;
const SHARD_LEN: usize = 64; // multiple of every kernel's alignment

fn data_byte(shard: usize, byte: usize) -> u8 {
    (shard as u8).wrapping_mul(31) ^ (byte as u8).wrapping_mul(7) ^ 0x5a
}

#[test]
fn rse_compat() {
    let mut shards: Vec<Vec<u8>> = (0..DATA + PARITY)
        .map(|s| {
            (0..SHARD_LEN)
                .map(|b| if s < DATA { data_byte(s, b) } else { 0 })
                .collect()
        })
        .collect();
    ReedSolomon::new(DATA, PARITY)
        .unwrap()
        .encode(&mut shards)
        .unwrap();

    for (j, shard) in shards[DATA..].iter().enumerate() {
        println!("parity[{j:2}] = {}", hex(shard));
    }

    let message: Vec<Gf2p8_11d> = (0..DATA)
        .flat_map(|s| (0..SHARD_LEN).map(move |b| Gf2p8_11d::from(data_byte(s, b))))
        .collect();
    let mut parity = vec![Gf2p8_11d::zero(); PARITY * SHARD_LEN];
    let mut workspace = vec![Gf2p8_11d::zero(); PARITY * SHARD_LEN];
    RsLutStandard::<64, 32>::new().encode_systematic_sharded(
        &message,
        &mut parity,
        &mut workspace,
        SHARD_LEN,
    );

    for j in 0..PARITY {
        let ours: Vec<u8> = parity[j * SHARD_LEN..(j + 1) * SHARD_LEN]
            .iter()
            .map(|&x| u8::from(x))
            .collect();
        assert_eq!(ours, shards[DATA + j], "parity shard {j} differs");
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

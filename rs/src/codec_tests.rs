use additive_fft_reed_solomon_gf2p8::{Gf2p8, Gf2p8_11d};

fn generate_lch_codeword<const N: usize, const T: usize>(rs: &Rs<N, T>) -> Vec<Gf2p8_11d> {
    let k = N - T;
    let mut message = vec![Gf2p8_11d::zero(); k];
    let mut parity = vec![Gf2p8_11d::zero(); T];

    for i in 0..k {
        message[i] = Gf2p8_11d::from((T + i) as u8);
    }

    rs.encode_systematic_scalar(&message, &mut parity);

    let mut codeword = vec![Gf2p8_11d::zero(); N];
    codeword[..T].copy_from_slice(&parity);
    codeword[T..].copy_from_slice(&message);
    codeword
}

fn generate_sharded_lch_codeword<const N: usize, const T: usize>(
    rs: &Rs<N, T>,
    shard_len: usize,
) -> Vec<Gf2p8_11d> {
    let k = N - T;
    let mut message = vec![Gf2p8_11d::zero(); shard_len * k];
    let mut parity = vec![Gf2p8_11d::zero(); shard_len * T];
    let mut workspace = vec![Gf2p8_11d::zero(); shard_len * T];

    for i in 0..k {
        for j in 0..shard_len {
            message[i * shard_len + j] = Gf2p8_11d::from((T + i + j) as u8);
        }
    }

    rs.encode_systematic_sharded(&message, &mut parity, &mut workspace, shard_len);

    let mut codeword = vec![Gf2p8_11d::zero(); shard_len * N];
    codeword[..T * shard_len].clone_from_slice(&parity);
    codeword[T * shard_len..].clone_from_slice(&message);
    codeword
}

#[test]
fn decode_no_errors() {
    decode_no_errors_t::<4, 2>();
    decode_no_errors_t::<8, 4>();
    decode_no_errors_t::<16, 8>();
    decode_no_errors_t::<32, 16>();
    decode_no_errors_t::<64, 32>();
    decode_no_errors_t::<128, 64>();
    decode_no_errors_t::<256, 128>();
}

fn decode_no_errors_t<const N: usize, const T: usize>() {
    let rs = Rs::<N, T>::new();
    let original = generate_lch_codeword(&rs);
    let mut received = original.clone();
    assert!(
        rs.decode_systematic_scalar(&mut received),
        "decode failed for N={N}, T={T}"
    );
    assert_eq!(received, original, "mismatch for N={N}, T={T}");
}

#[test]
fn decode_max_corrupt_parity() {
    decode_max_corrupt_parity_t::<4, 2>();
    decode_max_corrupt_parity_t::<8, 4>();
    decode_max_corrupt_parity_t::<16, 8>();
    decode_max_corrupt_parity_t::<32, 16>();
    decode_max_corrupt_parity_t::<64, 32>();
    decode_max_corrupt_parity_t::<128, 64>();
    decode_max_corrupt_parity_t::<256, 128>();
}

fn decode_max_corrupt_parity_t<const N: usize, const T: usize>() {
    let rs = Rs::<N, T>::new();
    let original = generate_lch_codeword(&rs);
    let mut received = original.clone();

    // Corrupt parity
    for (i, r) in received.iter_mut().take(T / 2).enumerate() {
        *r = (*r).mul((i as u8 + 2).into())
    }

    assert!(
        rs.decode_systematic_scalar(&mut received),
        "Failed to decode corrupt parity at N={N}, T={T}"
    );
    assert_eq!(received, original);
}

#[test]
fn decode_max_corrupt_data() {
    decode_max_corrupt_data_t::<4, 2>();
    decode_max_corrupt_data_t::<8, 4>();
    decode_max_corrupt_data_t::<16, 8>();
    decode_max_corrupt_data_t::<32, 16>();
    decode_max_corrupt_data_t::<64, 32>();
    decode_max_corrupt_data_t::<128, 64>();
    decode_max_corrupt_data_t::<256, 128>();
}

fn decode_max_corrupt_data_t<const N: usize, const T: usize>() {
    let rs = Rs::<N, T>::new();
    let original = generate_lch_codeword(&rs);
    let mut received = original.clone();

    // Corrupt data
    for (i, r) in received.iter_mut().skip(T).take(T / 2).enumerate() {
        *r = (*r).mul((i as u8 + 2).into())
    }

    assert!(
        rs.decode_systematic_scalar(&mut received),
        "Failed to decode corrupt data at N={N}, T={T}"
    );
    assert_eq!(received, original);
}

#[test]
fn recompute_max_corrupt_data() {
    recompute_max_corrupt_data_t::<4, 2>();
    recompute_max_corrupt_data_t::<8, 4>();
    recompute_max_corrupt_data_t::<16, 8>();
    recompute_max_corrupt_data_t::<32, 16>();
    recompute_max_corrupt_data_t::<64, 32>();
    recompute_max_corrupt_data_t::<128, 64>();
    recompute_max_corrupt_data_t::<256, 128>();
}

fn recompute_max_corrupt_data_t<const N: usize, const T: usize>() {
    let rs = Rs::<N, T>::new();
    let original = generate_lch_codeword(&rs);
    let mut received = original.clone();

    // Corrupt data
    for (i, r) in received.iter_mut().skip(T).enumerate() {
        *r = (*r).mul((i as u8 + 2).into())
    }

    rs.recompute_data_from_parity(&mut received);
    assert_eq!(received, original);
}

#[test]
fn encode_sharded_writes_parity() {
    encode_sharded_writes_parity_t::<4, 2>();
    encode_sharded_writes_parity_t::<8, 4>();
    encode_sharded_writes_parity_t::<16, 8>();
    encode_sharded_writes_parity_t::<32, 16>();
    encode_sharded_writes_parity_t::<64, 32>();
    encode_sharded_writes_parity_t::<128, 64>();
    encode_sharded_writes_parity_t::<256, 128>();
}

fn encode_sharded_writes_parity_t<const N: usize, const T: usize>() {
    const SHARD_LEN: usize = 32;
    let rs = Rs::<N, T>::new();
    let codeword = generate_sharded_lch_codeword(&rs, SHARD_LEN);
    // Verify that the parity shards have been written into.
    assert_ne!(
        codeword[..T * SHARD_LEN],
        vec![Gf2p8_11d::zero(); T * SHARD_LEN],
    );
}

#[test]
fn encode_sharded_generalizes_scalar() {
    encode_sharded_generalizes_scalar_t::<4, 2>();
    encode_sharded_generalizes_scalar_t::<8, 4>();
    encode_sharded_generalizes_scalar_t::<16, 8>();
    encode_sharded_generalizes_scalar_t::<32, 16>();
    encode_sharded_generalizes_scalar_t::<64, 32>();
    encode_sharded_generalizes_scalar_t::<128, 64>();
    encode_sharded_generalizes_scalar_t::<256, 128>();
}

fn encode_sharded_generalizes_scalar_t<const N: usize, const T: usize>() {
    let rs = Rs::<N, T>::new();
    let scalar = generate_lch_codeword(&rs);
    let sharded = generate_sharded_lch_codeword(&rs, 1);
    assert_eq!(scalar, sharded);
}

#[test]
fn recompute_max_corrupt_data_sharded() {
    recompute_max_corrupt_data_sharded_t::<4, 2>();
    recompute_max_corrupt_data_sharded_t::<8, 4>();
    recompute_max_corrupt_data_sharded_t::<16, 8>();
    recompute_max_corrupt_data_sharded_t::<32, 16>();
    recompute_max_corrupt_data_sharded_t::<64, 32>();
    recompute_max_corrupt_data_sharded_t::<128, 64>();
    recompute_max_corrupt_data_sharded_t::<256, 128>();
}

fn recompute_max_corrupt_data_sharded_t<const N: usize, const T: usize>() {
    const SHARD_LEN: usize = 16;
    let rs = Rs::<N, T>::new();
    let original = generate_sharded_lch_codeword(&rs, SHARD_LEN);
    let mut received = original.clone();

    // Corrupt data
    received[T * SHARD_LEN..].fill(Gf2p8_11d::zero());

    let mut workspace = vec![Gf2p8_11d::zero(); T * SHARD_LEN];
    rs.recompute_data_from_parity_sharded(&mut received, &mut workspace, SHARD_LEN);
    assert_eq!(received, original);
}

#[test]
fn recover_erasures_sharded() {
    recover_erasures_sharded_t::<2, 1>();
    recover_erasures_sharded_t::<4, 1>();
    recover_erasures_sharded_t::<4, 2>();
    recover_erasures_sharded_t::<8, 1>();
    recover_erasures_sharded_t::<8, 2>();
    recover_erasures_sharded_t::<8, 4>();
    recover_erasures_sharded_t::<16, 1>();
    recover_erasures_sharded_t::<16, 2>();
    recover_erasures_sharded_t::<16, 4>();
    recover_erasures_sharded_t::<16, 8>();
    recover_erasures_sharded_t::<32, 1>();
    recover_erasures_sharded_t::<32, 2>();
    recover_erasures_sharded_t::<32, 4>();
    recover_erasures_sharded_t::<32, 8>();
    recover_erasures_sharded_t::<32, 16>();
    recover_erasures_sharded_t::<64, 1>();
    recover_erasures_sharded_t::<64, 2>();
    recover_erasures_sharded_t::<64, 4>();
    recover_erasures_sharded_t::<64, 8>();
    recover_erasures_sharded_t::<64, 16>();
    recover_erasures_sharded_t::<64, 32>();
    recover_erasures_sharded_t::<128, 1>();
    recover_erasures_sharded_t::<128, 2>();
    recover_erasures_sharded_t::<128, 4>();
    recover_erasures_sharded_t::<128, 8>();
    recover_erasures_sharded_t::<128, 16>();
    recover_erasures_sharded_t::<128, 32>();
    recover_erasures_sharded_t::<128, 64>();
    recover_erasures_sharded_t::<256, 1>();
    recover_erasures_sharded_t::<256, 2>();
    recover_erasures_sharded_t::<256, 4>();
    recover_erasures_sharded_t::<256, 8>();
    recover_erasures_sharded_t::<256, 16>();
    recover_erasures_sharded_t::<256, 32>();
    recover_erasures_sharded_t::<256, 64>();
    recover_erasures_sharded_t::<256, 128>();
}

fn recover_erasures_sharded_t<const N: usize, const T: usize>() {
    const SHARD_LEN: usize = 1;

    let rs = Rs::<N, T>::new();
    let zero_shard = vec![Gf2p8_11d::zero(); SHARD_LEN];

    let original = generate_sharded_lch_codeword(&rs, SHARD_LEN);
    let mut received = original.clone();
    let mut erasure_positions = Vec::new();

    for (i, shard) in received
        .chunks_mut(SHARD_LEN)
        .enumerate()
        .skip(1)
        .take(T * 2 - 1)
        .step_by(2)
    {
        shard.clone_from_slice(&zero_shard);
        erasure_positions.push(i as u8);
    }

    let mut workspace = vec![Gf2p8_11d::zero(); N * SHARD_LEN];

    assert!(rs.recover_erasures_sharded(
        &mut received,
        &mut workspace,
        SHARD_LEN,
        &erasure_positions
    ));
    assert_eq!(received, original, "N={N}, T={T}");
}

#[test]
fn recover_erasures_sharded_clobber() {
    recover_erasures_sharded_clobber_t::<2, 1>();
    recover_erasures_sharded_clobber_t::<4, 1>();
    recover_erasures_sharded_clobber_t::<4, 2>();
    recover_erasures_sharded_clobber_t::<8, 1>();
    recover_erasures_sharded_clobber_t::<8, 2>();
    recover_erasures_sharded_clobber_t::<8, 4>();
    recover_erasures_sharded_clobber_t::<16, 1>();
    recover_erasures_sharded_clobber_t::<16, 2>();
    recover_erasures_sharded_clobber_t::<16, 4>();
    recover_erasures_sharded_clobber_t::<16, 8>();
    recover_erasures_sharded_clobber_t::<32, 1>();
    recover_erasures_sharded_clobber_t::<32, 2>();
    recover_erasures_sharded_clobber_t::<32, 4>();
    recover_erasures_sharded_clobber_t::<32, 8>();
    recover_erasures_sharded_clobber_t::<32, 16>();
    recover_erasures_sharded_clobber_t::<64, 1>();
    recover_erasures_sharded_clobber_t::<64, 2>();
    recover_erasures_sharded_clobber_t::<64, 4>();
    recover_erasures_sharded_clobber_t::<64, 8>();
    recover_erasures_sharded_clobber_t::<64, 16>();
    recover_erasures_sharded_clobber_t::<64, 32>();
    recover_erasures_sharded_clobber_t::<128, 1>();
    recover_erasures_sharded_clobber_t::<128, 2>();
    recover_erasures_sharded_clobber_t::<128, 4>();
    recover_erasures_sharded_clobber_t::<128, 8>();
    recover_erasures_sharded_clobber_t::<128, 16>();
    recover_erasures_sharded_clobber_t::<128, 32>();
    recover_erasures_sharded_clobber_t::<128, 64>();
    recover_erasures_sharded_clobber_t::<256, 1>();
    recover_erasures_sharded_clobber_t::<256, 2>();
    recover_erasures_sharded_clobber_t::<256, 4>();
    recover_erasures_sharded_clobber_t::<256, 8>();
    recover_erasures_sharded_clobber_t::<256, 16>();
    recover_erasures_sharded_clobber_t::<256, 32>();
    recover_erasures_sharded_clobber_t::<256, 64>();
    recover_erasures_sharded_clobber_t::<256, 128>();
}

fn recover_erasures_sharded_clobber_t<const N: usize, const T: usize>() {
    const SHARD_LEN: usize = 1;

    let rs = Rs::<N, T>::new();
    let zero_shard = vec![Gf2p8_11d::zero(); SHARD_LEN];

    let original = generate_sharded_lch_codeword(&rs, SHARD_LEN);
    let mut received = original.clone();
    let mut erasure_positions = Vec::new();

    for (i, shard) in received
        .chunks_mut(SHARD_LEN)
        .enumerate()
        .skip(1)
        .take(T * 2 - 1)
        .step_by(2)
    {
        shard.clone_from_slice(&zero_shard);
        erasure_positions.push(i as u8);
    }

    assert!(rs.recover_erasures_sharded_clobber(&mut received, SHARD_LEN, &erasure_positions));

    for i in 0..N {
        if erasure_positions.contains(&u8::try_from(i).unwrap()) {
            assert_eq!(
                received[i * SHARD_LEN],
                original[i * SHARD_LEN],
                "N={N}, T={T}, i={i}",
            );
        }
    }
}

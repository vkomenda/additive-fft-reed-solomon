use super::*;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};

#[test]
fn fft_sharded_zero_padded_matches_fft_sharded_rec() {
    const SHARD_LEN: usize = 37;

    let mut rng = SmallRng::seed_from_u64(42);

    for k in 1..=8u8 {
        let n = 1usize << k;
        for log_support in 0..k {
            let support = 1usize << log_support;
            let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
            let support_len = support * SHARD_LEN;
            rng.fill_bytes(unsafe {
                std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, support_len)
            });
            a[support * SHARD_LEN..].fill(Gf2p8_11d::zero());
            let mut b = a.clone();
            b[support * SHARD_LEN..].fill(Gf2p8_11d::from(0xaa));

            let beta = Gf2p8_11d::zero();
            fft_sharded_recursive::<_, B>(&mut a, SHARD_LEN, k, beta);
            fft_sharded_zero_padded_recursive::<_, B>(&mut b, SHARD_LEN, k, beta, log_support);

            assert_eq!(a, b, "k={k} log_support={log_support}");
        }
    }
}

#[test]
fn unrolled_fft_sharded_zero_padded_matches_fft_sharded_rec() {
    const SHARD_LEN: usize = 37;

    let mut rng = SmallRng::seed_from_u64(42);

    for k in 1..=8u8 {
        let n = 1 << k;
        for log_support in 0..k {
            let support = 1 << log_support;

            let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
            let support_len = support * SHARD_LEN;
            rng.fill_bytes(unsafe {
                std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, support_len)
            });
            a[support * SHARD_LEN..].fill(Gf2p8_11d::zero());
            let mut b = a.clone();
            b[support * SHARD_LEN..].fill(Gf2p8_11d::from(0xaa));

            fft_sharded_recursive::<_, B>(&mut a, SHARD_LEN, k, Gf2p8_11d::zero());
            LutKernel::<_, B>::fft_sharded_zero_padded(&mut b, SHARD_LEN, k, log_support);

            assert_eq!(a, b);
        }
    }
}

#[test]
fn dit4_matches_two_dit2_levels() {
    const SHARD_LEN: usize = 37;
    let mut rng = SmallRng::seed_from_u64(1);

    for d in [1, 2, 4, 8] {
        for base in [0, 1, 3] {
            let n = base + 4 * d;

            let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
            rng.fill_bytes(unsafe {
                std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
            });
            let mut b = a.clone();

            let m02 = LutKernel::<_, B>::mul_table(Gf2p8_11d(0x53));
            let m01 = LutKernel::<_, B>::mul_table(Gf2p8_11d(0x2a));
            let m23 = LutKernel::<_, B>::mul_table(Gf2p8_11d(0xc7));

            // The wide level over the whole node, then the narrow level over each half.
            LutKernel::<_, B>::butterfly_fwd_dit2(&mut a, SHARD_LEN, base, 2 * d, m02);
            LutKernel::<_, B>::butterfly_fwd_dit2(&mut a, SHARD_LEN, base, d, m01);
            LutKernel::<_, B>::butterfly_fwd_dit2(&mut a, SHARD_LEN, base + 2 * d, d, m23);

            LutKernel::<_, B>::butterfly_fwd_dit4(&mut b, SHARD_LEN, base, d, m01, m23, m02);

            assert_eq!(a, b, "d={d} base={base}");
        }
    }
}

#[test]
fn fft_sharded_dit4_matches_fft_sharded_rec() {
    const SHARD_LEN: usize = 37;

    let mut rng = SmallRng::seed_from_u64(42);
    let beta: u8 = rng.random();
    let beta = Gf2p8_11d::from(beta);

    for k in 1..=8 {
        let n = 1 << k;

        let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
        rng.fill_bytes(unsafe {
            std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
        });
        let mut b = a.clone();

        fft_sharded_recursive::<_, B>(&mut a, SHARD_LEN, k, beta);
        LutKernel::<_, B>::fft_sharded_dit4(&mut b, SHARD_LEN, k, beta);

        assert_eq!(a, b, "k={k}");
    }
}

#[test]
fn ifft_sharded_dit4_matches_ifft_sharded_rec() {
    const SHARD_LEN: usize = 37;

    let mut rng = SmallRng::seed_from_u64(42);
    let beta: u8 = rng.random();
    let beta = Gf2p8_11d::from(beta);

    for k in 1..=8 {
        let n = 1 << k;

        let mut a = vec![Gf2p8_11d::zero(); n * SHARD_LEN];
        rng.fill_bytes(unsafe {
            std::slice::from_raw_parts_mut(a.as_mut_ptr() as *mut u8, n * SHARD_LEN)
        });
        let mut b = a.clone();

        ifft_sharded_recursive::<_, B>(&mut a, SHARD_LEN, k, beta);
        LutKernel::<_, B>::ifft_sharded_dit4(&mut b, SHARD_LEN, k, beta);

        assert_eq!(a, b, "k={k}");
    }
}

use std::hint::black_box;

use additive_fft_reed_solomon::{
    codec::{Codec, ParityFirst},
    kernel::lut_kernel::LutKernel,
    poly_11d_lut::{CantorBasisLut11d, StandardBasisLut11d},
};
use additive_fft_reed_solomon_gf2p8::Gf2p8_11d;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

/// T distinct erasure positions below N, sorted ascending as the decoder requires.
fn erasures(n: usize, t: usize, rng: &mut SmallRng) -> Vec<u8> {
    let mut all: Vec<u8> = (0..n).map(|i| i as u8).collect();
    all.shuffle(rng);
    let mut e = all[..t].to_vec();
    e.sort_unstable();
    e
}

macro_rules! bench_locator {
    ($group:expr, $rng:expr, $basis:ty, $basis_name:expr, [$(($n:expr, $t:expr)),* $(,)?]) => {
        $({
            type Rs = Codec<Gf2p8_11d, $basis, LutKernel<Gf2p8_11d, $basis>, ParityFirst, $n, $t>;
            let rs = Rs::new();
            let positions = erasures($n, $t, $rng);
            $group.bench_function(format!("{}_N{}_T{}", $basis_name, $n, $t), |b| {
                b.iter(|| rs.locator_log_evals(black_box(&positions), black_box($n)))
            });
        })*
    };
}

fn bench_locator_log_evals(c: &mut Criterion) {
    let mut group = c.benchmark_group("locator_log_evals");
    let mut rng = SmallRng::seed_from_u64(42);

    bench_locator!(
        group,
        &mut rng,
        CantorBasisLut11d,
        "cantor",
        [
            (4, 2),
            (16, 8),
            (32, 16),
            (64, 16),
            (64, 32),
            (128, 64),
            (256, 128)
        ]
    );
    bench_locator!(
        group,
        &mut rng,
        StandardBasisLut11d,
        "standard",
        [
            (4, 2),
            (16, 8),
            (32, 16),
            (64, 16),
            (64, 32),
            (128, 64),
            (256, 128)
        ]
    );

    group.finish();
}

criterion_group!(benches, bench_locator_log_evals);
criterion_main!(benches);

use crate::gf2p8lut::SubspaceBasisLut;
use crate::kernel::Kernel;
use additive_fft_reed_solomon_gf2p8::Gf2p8;

#[test]
fn fft_evaluates_x() {
    let n = 256;
    let mut a = vec![Gf2p8_11d::zero(); n];
    a[1] = Gf2p8_11d::one(); // coefficient of X̂_1 = x
    K::fft_sharded(&mut a, 1, 8, Gf2p8_11d::zero());
    for i in 0..n {
        assert_eq!(a[i], B::get_subspace_point_lut(i as u8), "i={i}");
    }
}

#[test]
fn fft_evaluates_subspace_polys() {
    for l in 0..8u8 {
        let mut a = vec![Gf2p8_11d::zero(); 256];
        a[1 << l] = Gf2p8_11d::one();
        K::fft_sharded(&mut a, 1, 8, Gf2p8_11d::zero());
        for i in 0..256 {
            let w = B::get_subspace_point_lut(i as u8);
            assert_eq!(a[i], B::eval_subspace_poly_lut(l, w), "l={l} i={i}");
        }
    }
}

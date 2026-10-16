use super::*;
use crate::poly_arith::{SubspaceBasisPolyArith, SubspaceBasisPolySliceArith, poly};
use additive_fft_reed_solomon_gf2p8::FIELD_SIZE;

#[test]
fn scalar_fft_ifft_id() {
    let basis = B {};
    let mut data = [0u8.into(); 256];

    for i in 0..128 {
        data[i] = Gf2p8_11d::from(i as u8);
    }
    let original = data;

    // k=8 for 256 points, beta=0 for the standard subspace V_8
    basis.fft_scalar(&mut data, 8, 0u8.into());

    assert_ne!(data, original, "FFT should have transformed the data");

    basis.ifft_scalar(&mut data, 8, 0u8.into());

    for i in 0..256 {
        assert_eq!(
            data[i], original[i],
            "IFFT failed to recover coefficient at index {}",
            i
        );
    }
}
#[test]
fn poly_mul_lnh_by_zero() {
    let basis = B {};
    let zero = [0u8.into(); FIELD_SIZE];

    let mut a = [0u8.into(); FIELD_SIZE];
    a[0] = 0x53.into();
    a[1] = 0xca.into();
    a[3] = 1.into();

    assert_eq!(basis.poly_mul_lnh(&a, &zero), zero);
    assert_eq!(basis.poly_mul_lnh(&zero, &a), zero);
}

#[test]
fn poly_mul_lnh_by_one() {
    let basis = B {};
    let mut one = [0u8.into(); FIELD_SIZE];
    one[0] = 1.into();

    let mut a = [0u8.into(); FIELD_SIZE];
    a[0] = 0xab.into();
    a[2] = 0x3f.into();
    a[7] = 0x11.into();

    assert_eq!(basis.poly_mul_lnh(&a, &one), a);
    assert_eq!(basis.poly_mul_lnh(&one, &a), a);
}

#[test]
fn poly_mul_lnh_const_by_const() {
    let basis = B {};
    let mut a = [0u8.into(); FIELD_SIZE];
    let a0 = 0x53.into();
    a[0] = a0;

    let mut b = [0u8.into(); FIELD_SIZE];
    let b0 = 0xca.into();
    b[0] = b0;

    let result = basis.poly_mul_lnh(&a, &b);
    assert_eq!(result[0], a0.mul_lut(b0));
    assert_eq!(result[1..], [0u8.into(); FIELD_SIZE - 1]);
}

#[test]
fn poly_mul_lnh_commutes() {
    let basis = B {};
    let mut a = [0u8.into(); FIELD_SIZE];
    a[0] = 1.into();
    a[1] = 0xff.into();

    let mut b = [0u8.into(); FIELD_SIZE];
    b[0] = 0x57.into();
    b[2] = 0x83.into();
    b[5] = 0x2a.into();

    assert_eq!(basis.poly_mul_lnh(&a, &b), basis.poly_mul_lnh(&b, &a));
}

#[test]
fn poly_mul_lnh_associative() {
    let basis = B {};

    let mut a = [0.into(); FIELD_SIZE];
    a[0] = 3.into();
    a[1] = 9.into();

    let mut b = [0.into(); FIELD_SIZE];
    b[0] = 0xe5.into();
    b[3] = 0x7c.into();

    let mut c = [0.into(); FIELD_SIZE];
    c[0] = 0xaa.into();
    c[1] = 0x55.into();

    let ab_c = basis.poly_mul_lnh(&basis.poly_mul_lnh(&a, &b), &c);
    let a_bc = basis.poly_mul_lnh(&a, &basis.poly_mul_lnh(&b, &c));
    assert_eq!(ab_c, a_bc);
}

#[test]
fn poly_mul_lnh_distributive_over_addition() {
    let basis = B {};

    let mut a = [0.into(); FIELD_SIZE];
    a[0] = 0x11.into();
    a[2] = 0x22.into();

    let mut b = [0.into(); FIELD_SIZE];
    b[1] = 0x33.into();
    b[3] = 0x44.into();

    let mut c = [0.into(); FIELD_SIZE];
    c[0] = 0x55.into();
    c[2] = 0x66.into();

    let b_plus_c = poly::add(&b, &c);

    let lhs = basis.poly_mul_lnh(&a, &b_plus_c);
    let rhs = poly::add(&basis.poly_mul_lnh(&a, &b), &basis.poly_mul_lnh(&a, &c));

    assert_eq!(lhs, rhs);
}

#[test]
fn poly_div_lnh_by_constant_scales_coefficients() {
    let basis = B {};
    let c: Gf2p8_11d = 0x53.into();
    let c_inv = c.inv_lut();

    let mut a = [0.into(); FIELD_SIZE];
    a[0] = 0xef.into();
    a[3] = 0xcd.into();
    a[7] = 0xab.into();

    let mut b = [0.into(); FIELD_SIZE];
    b[0] = c;

    let (q, r) = basis.poly_div_lnh(&a, &b).unwrap();

    assert_eq!(r, [0.into(); FIELD_SIZE]);

    for i in 0..FIELD_SIZE {
        assert_eq!(q[i], a[i].mul_lut(c_inv));
    }
}

#[test]
fn field_non_degeneracy() {
    for k in 1..=7 {
        let v_k = B::get_subspace_point_lut(1 << k);
        let v_k_minus_1 = B::get_subspace_point_lut(1 << (k - 1));
        let s_k_minus_1_v_k = B::eval_subspace_poly_lut(k - 1, v_k);
        let s_k_minus_1_v_k_minus_1 = B::eval_subspace_poly_lut(k - 1, v_k_minus_1);
        assert_ne!(s_k_minus_1_v_k_minus_1, s_k_minus_1_v_k);
    }
}

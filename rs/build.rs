use additive_fft_reed_solomon_gf2p8::{
    CantorBasis, CantorBasis11d, EXP_TABLE_SIZE, FIELD_SIZE, Gf2p8, Gf2p8_11d,
};
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn write_points<G>(f: &mut File, it: impl Iterator<Item = G>, has_subarrays: bool)
where
    u8: From<G>,
{
    for (i, point) in it.enumerate() {
        if i % 16 == 0 {
            write!(f, "\n    ").unwrap();
            if has_subarrays {
                write!(f, "    ").unwrap();
            }
        }
        write!(f, "0x{:02x}, ", u8::from(point)).unwrap();
    }
    if has_subarrays {
        writeln!(f, "\n    ],").unwrap();
    } else {
        writeln!(f, "\n];").unwrap();
    }
}

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("tables_11d.rs");
    let mut f = File::create(&dest_path).unwrap();

    let (exp_table, log_table) = Gf2p8_11d::exp_log_tables();
    let inv_table = Gf2p8_11d::inv_table(&exp_table, &log_table);
    write!(f, "\npub const EXP_TABLE: [u8; {}] = [", EXP_TABLE_SIZE).unwrap();
    write_points(&mut f, exp_table.into_iter(), false);
    write!(f, "\npub const LOG_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, log_table.into_iter(), false);
    write!(f, "\npub const INV_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, inv_table.into_iter(), false);

    let basis = CantorBasis11d::new();

    write!(f, "\npub const CANTOR_BASIS: [u8; 8] = [").unwrap();
    write_points(&mut f, basis.into_iter(), false);

    let gfni_mul_iter = Gf2p8_11d::iter_gfni_mul_matrices();

    writeln!(f, "\npub const GFNI_MUL_TABLE: [u64; {}] = [", FIELD_SIZE).unwrap();
    for mat in gfni_mul_iter {
        writeln!(f, "    0x{:016x},", mat).unwrap();
    }
    writeln!(f, "];").unwrap();

    let (num_points, points_iter) = basis.iter_subspace_points();

    write!(f, "\npub const CANTOR_SUBSPACE: [u8; {}] = [", num_points).unwrap();
    write_points(&mut f, points_iter, false);

    let sub_poly_luts = basis.gen_all_subspace_poly_luts();

    writeln!(
        f,
        "\npub const SUBSPACE_POLY_VALUES: [[u8; {}]; 9] = [",
        FIELD_SIZE,
    )
    .unwrap();
    for lut in sub_poly_luts {
        write!(f, "    [").unwrap();
        write_points(&mut f, lut.into_iter(), true);
    }
    writeln!(f, "];").unwrap();

    let sub_poly_coeffs_iter = CantorBasis11d::gen_subspace_poly_coeffs();

    write!(f, "\npub const SUBSPACE_POLY_COEFFS: [u8; 9] = [").unwrap();
    write_points(&mut f, sub_poly_coeffs_iter, false);

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/mod.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/avx512_impl.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/generic.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/bit_matrix.rs");
    println!("cargo:rerun-if-changed=src/poly_11d/field_defs.rs");

    // CPU feature detection
    #[cfg(target_arch = "x86_64")]
    {
        let has_gfni = is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("gfni");
        let has_avx2 = is_x86_feature_detected!("avx2");

        if has_gfni {
            println!("cargo:rustc-cfg=native_gfni");
        }
        if has_avx2 {
            println!("cargo:rustc-cfg=native_avx2");
        }
    }

    // Emit the lint checker tweaks on all platforms.
    println!("cargo:rustc-check-cfg=cfg(native_gfni)");
    println!("cargo:rustc-check-cfg=cfg(native_avx2)");
}

use additive_fft_reed_solomon_gf2p8::{
    CantorBasis, CantorBasis11d, EXP_TABLE_SIZE, FIELD_SIZE, Gf2p8, Gf2p8_11d,
};
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

fn write_butterfly_fwd<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    twiddle: G,
    offset: usize,
    half: usize,
) -> io::Result<()> {
    let end = offset + half * 2;

    writeln!(f, "    {{")?;
    let mul_lut = twiddle.make_mul_table(exp, log);
    writeln!(f, "        let lut: [u8; FIELD_SIZE] = [")?;
    for (i, &b) in mul_lut.iter().enumerate() {
        if i % 16 == 0 {
            write!(f, "            ")?;
        }
        write!(f, "0x{b:02X},")?;
        if i % 16 == 15 {
            writeln!(f)?;
        }
    }
    writeln!(f, "        ];")?;

    if half == 1 {
        writeln!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);"
        )?;
        writeln!(
            f,
            "        butterfly_fwd(&mut lo[..shard_len], &mut hi[..shard_len], &lut);"
        )?;
    } else {
        writeln!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];"
        )?;
        writeln!(f, "        for i in 0..{half} {{")?;
        writeln!(
            f,
            "            let (left, right) = block.split_at_mut((i + {half}) * shard_len);"
        )?;
        writeln!(
            f,
            "            let a = &mut left[i * shard_len..(i + 1) * shard_len];"
        )?;
        writeln!(f, "            let b = &mut right[..shard_len];")?;
        writeln!(f, "            butterfly_fwd(a, b, &lut);")?;
        writeln!(f, "        }}")?;
    }
    writeln!(f, "    }}")?;
    Ok(())
}

fn write_butterfly_inv<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    twiddle: G,
    offset: usize,
    half: usize,
) -> io::Result<()> {
    let end = offset + half * 2;
    writeln!(f, "    {{")?;
    let mul_lut = twiddle.make_mul_table(exp, log);
    writeln!(f, "        let lut: [u8; FIELD_SIZE] = [")?;
    for (i, &b) in mul_lut.iter().enumerate() {
        if i % 16 == 0 {
            write!(f, "            ")?;
        }
        write!(f, "0x{b:02X},")?;
        if i % 16 == 15 {
            writeln!(f)?;
        }
    }
    writeln!(f, "        ];")?;
    if half == 1 {
        writeln!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);"
        )?;
        writeln!(
            f,
            "        butterfly_inv(&mut lo[..shard_len], &mut hi[..shard_len], &lut);"
        )?;
    } else {
        writeln!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];"
        )?;
        writeln!(f, "        for i in 0..{half} {{")?;
        writeln!(
            f,
            "            let (left, right) = block.split_at_mut((i + {half}) * shard_len);"
        )?;
        writeln!(
            f,
            "            let a = &mut left[i * shard_len..(i + 1) * shard_len];"
        )?;
        writeln!(f, "            let b = &mut right[..shard_len];")?;
        writeln!(f, "            butterfly_inv(a, b, &lut);")?;
        writeln!(f, "        }}")?;
    }
    writeln!(f, "    }}")?;
    Ok(())
}

fn write_fft_lut<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; FIELD_SIZE]; 8],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    l: u8,
    beta: G,
    offset: usize,
) -> io::Result<()> {
    let half = 1 << l;
    let twiddle = if l == 0 {
        beta
    } else {
        lut[l as usize][beta.into_usize()]
    };

    write_butterfly_fwd(f, exp, log, twiddle, offset, half)?;

    if l == 0 {
        return Ok(());
    }

    let next_beta = beta.add(basis[l as usize]);
    write_fft_lut(f, basis, lut, exp, log, l - 1, beta, offset)?;
    write_fft_lut(f, basis, lut, exp, log, l - 1, next_beta, offset + half)?;
    Ok(())
}

fn write_ifft_lut<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; FIELD_SIZE]; 8],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    l: u8,
    beta: G,
    offset: usize,
) -> io::Result<()> {
    let half = 1 << l;
    if l == 0 {
        let twiddle = beta;
        write_butterfly_inv(f, exp, log, twiddle, offset, half)?;
        return Ok(());
    }

    let next_beta = beta.add(basis[l as usize]);
    write_ifft_lut(f, basis, lut, exp, log, l - 1, beta, offset)?;
    write_ifft_lut(f, basis, lut, exp, log, l - 1, next_beta, offset + (1 << l))?;

    let twiddle = lut[l as usize][beta.into_usize()];
    write_butterfly_inv(f, exp, log, twiddle, offset, 1 << l)?;
    Ok(())
}

fn write_fft_lut_case<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; 256]; 8],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    n: usize,
    k: u8,
    is_ifft: bool,
) -> io::Result<()> {
    writeln!(
        f,
        "pub fn {}fft_sharded_lut_{n}<G: Gf2p8>(shards: &mut [G], shard_len: usize) {{",
        if is_ifft { "i" } else { "" }
    )?;
    writeln!(f, "    debug_assert_eq!(shards.len(), {n} * shard_len);")?;
    if !is_ifft {
        write_fft_lut(f, basis, lut, exp, log, k - 1, G::zero(), 0)?;
    } else {
        write_ifft_lut(f, basis, lut, exp, log, k - 1, G::zero(), 0)?;
    }
    writeln!(f, "}}")?;
    writeln!(f)?;
    Ok(())
}

fn write_unrolled_kernel_lut<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    sub_poly_luts: &[[G; FIELD_SIZE]; 8],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
) -> io::Result<()>
where
    u8: From<G>,
{
    writeln!(
        f,
        "use additive_fft_reed_solomon_gf2p8::{{FIELD_SIZE, Gf2p8}};"
    )?;
    writeln!(f, "use super::{{butterfly_fwd, butterfly_inv}};")?;
    writeln!(f)?;

    let cases: Vec<(usize, u8)> = (1..9).map(|a| (1usize << a, a)).collect();

    for (n, k) in cases {
        write_fft_lut_case(f, basis, sub_poly_luts, exp, log, n, k, false)?;
        write_fft_lut_case(f, basis, sub_poly_luts, exp, log, n, k, true)?;
    }

    Ok(())
}

fn write_points<G>(f: &mut impl Write, it: impl Iterator<Item = G>, has_subarrays: bool)
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
    let mut f = BufWriter::new(File::create(&dest_path).unwrap());

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

    let sub_poly_luts8: &[[Gf2p8_11d; 256]; 8] = sub_poly_luts[..8].try_into().unwrap();

    let dest_kernel_lut = Path::new(&out_dir).join("unrolled_lut_kernel_11d.rs");
    let mut fkl = BufWriter::new(File::create(&dest_kernel_lut).unwrap());
    write_unrolled_kernel_lut(
        &mut fkl,
        basis.as_ref(),
        sub_poly_luts8,
        &exp_table,
        &log_table,
    )
    .expect("LUT kernel");

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

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/mod.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/avx512_impl.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/generic.rs");
    println!("cargo:rerun-if-changed=src/gf2p8/bit_matrix.rs");
    println!("cargo:rerun-if-changed=src/poly_11d/field_defs.rs");
}

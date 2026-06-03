use additive_fft_reed_solomon_gf2p8::{
    CantorBasis, CantorBasis11d, EXP_TABLE_SIZE, FIELD_SIZE, Gf2p8, Gf2p8_11d,
};
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

fn write_lut(f: &mut impl Write, lut: &[u8; FIELD_SIZE]) -> io::Result<()> {
    writeln!(f, "        let lut: [u8; FIELD_SIZE] = [")?;
    for (i, &b) in lut.iter().enumerate() {
        if i % 16 == 0 {
            write!(f, "            ")?;
        }
        write!(f, "0x{b:02x},")?;
        if i % 16 == 15 {
            writeln!(f)?;
        }
    }
    writeln!(f, "        ];")?;

    Ok(())
}

fn write_butterfly_fwd<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    twiddle: G,
    offset: usize,
    half: usize,
) -> io::Result<()> {
    let end = offset + half * 2;

    let fwd_op = if twiddle == G::zero() {
        "            for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = bi.add(*ai); }"
    } else {
        "            butterfly_fwd(a, b, &lut);"
    };

    let fwd_op_half1 = if twiddle == G::zero() {
        "        for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = bi.add(*ai); }"
    } else {
        "        butterfly_fwd(&mut lo[..shard_len], &mut hi[..shard_len], &lut);"
    };

    writeln!(f, "    {{")?;
    if twiddle != G::zero() {
        write_lut(f, &twiddle.make_mul_table(exp, log))?;
    }

    if half == 1 {
        write!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);
{fwd_op_half1}
    }}\n"
        )?;
    } else {
        write!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];
        for i in 0..{half} {{
            let (left, right) = block.split_at_mut((i + {half}) * shard_len);
            let a = &mut left[i * shard_len..(i + 1) * shard_len];
            let b = &mut right[..shard_len];
{fwd_op}
        }}
    }}\n"
        )?;
    }
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

    let inv_op = if twiddle == G::zero() {
        "            for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = ai.add(*bi); }"
    } else {
        "            butterfly_inv(a, b, &lut);"
    };

    let inv_op_half1 = if twiddle == G::zero() {
        "        for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = ai.add(*bi); }"
    } else {
        "        butterfly_inv(&mut lo[..shard_len], &mut hi[..shard_len], &lut);"
    };

    writeln!(f, "    {{")?;
    if twiddle != G::zero() {
        write_lut(f, &twiddle.make_mul_table(exp, log))?;
    }

    if half == 1 {
        write!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);
{inv_op_half1}
    }}\n"
        )?;
    } else {
        write!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];
        for i in 0..{half} {{
            let (left, right) = block.split_at_mut((i + {half}) * shard_len);
            let a = &mut left[i * shard_len..(i + 1) * shard_len];
            let b = &mut right[..shard_len];
{inv_op}
        }}
    }}\n"
        )?;
    }
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
    lut: &[[G; FIELD_SIZE]; 8],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
    n: usize,
    k: u8,
    beta: G,
    is_ifft: bool,
) -> io::Result<()> {
    writeln!(
        f,
        "pub fn {}fft_sharded_lut_{n}{}<G: Gf2p8>(shards: &mut [G], shard_len: usize) {{",
        if is_ifft { "i" } else { "" },
        if beta != G::zero() {
            format!("_{:02x}", beta.into())
        } else {
            "".to_string()
        }
    )?;
    writeln!(f, "    debug_assert_eq!(shards.len(), {n} * shard_len);")?;
    if !is_ifft {
        write_fft_lut(f, basis, lut, exp, log, k, beta, 0)?;
    } else {
        write_ifft_lut(f, basis, lut, exp, log, k, beta, 0)?;
    }
    writeln!(f, "}}")?;
    writeln!(f)?;
    Ok(())
}

fn _write_ifft_lut_dispatch<G: Gf2p8>(
    f: &mut impl Write,
    subspace_points: &[G; FIELD_SIZE],
    omega_cases: &[(usize, u8, usize)],
) -> io::Result<()> {
    writeln!(
        f,
        "#[inline(always)]
pub fn dispatch_ifft_lut<G: Gf2p8>(
    basis: &impl CantorBasisLut<G>,
    shards: &mut [G], shard_len: usize, k: u8, beta: G
) {{
    match (k, u8::from(beta)) {{
        (0, _) => {{}}"
    )?;
    for &(n, k, t) in omega_cases {
        let omega = subspace_points[t].into();
        writeln!(
            f,
            "        ({k}, b) if b == CANTOR_SUBSPACE[{t}] => ifft_sharded_lut_{n}_{omega:02x}(shards, shard_len),"
        )?;
    }
    writeln!(
        f,
        "        _ => super::ifft_sharded_lut(basis, shards, shard_len, k, beta),
    }}
}}"
    )?;
    Ok(())
}

fn write_unrolled_kernel_lut<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    sub_poly_luts: &[[G; FIELD_SIZE]; 8],
    subspace_points: &[G; FIELD_SIZE],
    exp: &[u8; EXP_TABLE_SIZE],
    log: &[u8; FIELD_SIZE],
) -> io::Result<()>
where
    u8: From<G>,
{
    writeln!(
        f,
        "\
use additive_fft_reed_solomon_gf2p8::{{FIELD_SIZE, Gf2p8}};
use super::{{butterfly_fwd, butterfly_inv}};
"
    )?;

    let cases: Vec<(usize, u8)> = (0..8).map(|a| (2usize << a, a)).collect();

    for (n, k) in cases {
        write_fft_lut_case(f, basis, sub_poly_luts, exp, log, n, k, G::zero(), false)?;
        write_fft_lut_case(f, basis, sub_poly_luts, exp, log, n, k, G::zero(), true)?;
    }

    let omega_cases: Vec<(usize, u8, usize)> =
        (0..8).map(|a| (2usize << a, a, 1usize << a)).collect();

    for &(n, k, t) in &omega_cases {
        let omega = subspace_points[t];
        write_fft_lut_case(f, basis, sub_poly_luts, exp, log, n, k, omega, true)?;
    }

    Ok(())
}

fn write_butterfly_fwd_gfni<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    mats: &[u64; FIELD_SIZE],
    twiddle: G,
    offset: usize,
    half: usize,
) -> io::Result<()> {
    let end = offset + half * 2;

    let fwd_op = if twiddle == G::zero() {
        "for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = bi.add(*ai); }"
    } else {
        "\
            unsafe {
                butterfly_fwd_gfni(a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, shard_len, mat);
            }"
    };

    let fwd_op_half1 = if twiddle == G::zero() {
        "for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = bi.add(*ai); }"
    } else {
        "\
        unsafe {
            butterfly_fwd_gfni(lo.as_mut_ptr() as *mut u8, hi.as_mut_ptr() as *mut u8, shard_len, mat);
        }"
    };

    writeln!(f, "    {{")?;
    if twiddle != G::zero() {
        let mat = mats[twiddle.into_usize()];
        writeln!(
            f,
            "        let mat = unsafe {{ _mm512_set1_epi64(0x{mat:016x}u64 as i64) }};"
        )?;
    }

    if half == 1 {
        writeln!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);
        {fwd_op_half1}
    }}"
        )?;
    } else {
        writeln!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];
        for i in 0..{half} {{
            let (left, right) = block.split_at_mut((i + {half}) * shard_len);
            let a = &mut left[i * shard_len..(i + 1) * shard_len];
            let b = &mut right[..shard_len];
            {fwd_op}
        }}
    }}"
        )?;
    }
    Ok(())
}

fn write_butterfly_inv_gfni<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    mats: &[u64; FIELD_SIZE],
    twiddle: G,
    offset: usize,
    half: usize,
) -> io::Result<()> {
    let end = offset + half * 2;

    let inv_op = if twiddle == G::zero() {
        "for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = ai.add(*bi); }"
    } else {
        "\
            unsafe {
                butterfly_inv_gfni(a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, shard_len, mat);
            }"
    };

    let inv_op_half1 = if twiddle == G::zero() {
        "for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = ai.add(*bi); }"
    } else {
        "\
        unsafe {
            butterfly_inv_gfni(lo.as_mut_ptr() as *mut u8, hi.as_mut_ptr() as *mut u8, shard_len, mat);
        }"
    };

    writeln!(f, "    {{")?;
    if twiddle != G::zero() {
        let mat = mats[twiddle.into_usize()];
        writeln!(
            f,
            "        let mat = unsafe {{ _mm512_set1_epi64(0x{mat:016x}u64 as i64) }};"
        )?;
    }

    if half == 1 {
        writeln!(
            f,
            "        let (lo, hi) = shards[{offset} * shard_len..].split_at_mut(shard_len);
        {inv_op_half1}
    }}"
        )?;
    } else {
        writeln!(
            f,
            "        let block = &mut shards[{offset} * shard_len..{end} * shard_len];
        for i in 0..{half} {{
            let (left, right) = block.split_at_mut((i + {half}) * shard_len);
            let a = &mut left[i * shard_len..(i + 1) * shard_len];
            let b = &mut right[..shard_len];
            {inv_op}
        }}
    }}"
        )?;
    }
    Ok(())
}

fn write_fft_gfni<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; FIELD_SIZE]; 8],
    mats: &[u64; FIELD_SIZE],
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

    write_butterfly_fwd_gfni(f, mats, twiddle, offset, half)?;

    if l == 0 {
        return Ok(());
    }

    let next_beta = beta.add(basis[l as usize]);
    write_fft_gfni(f, basis, lut, mats, l - 1, beta, offset)?;
    write_fft_gfni(f, basis, lut, mats, l - 1, next_beta, offset + half)?;
    Ok(())
}

fn write_ifft_gfni<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; FIELD_SIZE]; 8],
    mats: &[u64; FIELD_SIZE],
    l: u8,
    beta: G,
    offset: usize,
) -> io::Result<()> {
    let half = 1 << l;
    if l == 0 {
        let twiddle = beta;
        write_butterfly_inv_gfni(f, mats, twiddle, offset, half)?;
        return Ok(());
    }

    let next_beta = beta.add(basis[l as usize]);
    write_ifft_gfni(f, basis, lut, mats, l - 1, beta, offset)?;
    write_ifft_gfni(f, basis, lut, mats, l - 1, next_beta, offset + (1 << l))?;

    let twiddle = lut[l as usize][beta.into_usize()];
    write_butterfly_inv_gfni(f, mats, twiddle, offset, 1 << l)?;
    Ok(())
}

fn write_fft_gfni_case<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    lut: &[[G; FIELD_SIZE]; 8],
    mats: &[u64; FIELD_SIZE],
    n: usize,
    k: u8,
    beta: G,
    is_ifft: bool,
) -> io::Result<()> {
    writeln!(f, "#[cfg(any(native_gfni, feature = \"compile_gfni\"))]")?;
    //    writeln!(f, "#[target_feature(enable = \"avx512f,avx512bw,gfni\")]")?;
    writeln!(
        f,
        "pub fn {}fft_sharded_gfni_{n}{}<G: Gf2p8>(shards: &mut [G], shard_len: usize) {{",
        if is_ifft { "i" } else { "" },
        if beta != G::zero() {
            format!("_{:02x}", beta.into())
        } else {
            "".to_string()
        }
    )?;
    writeln!(f, "    debug_assert_eq!(shards.len(), {n} * shard_len);")?;
    if !is_ifft {
        write_fft_gfni(f, basis, lut, mats, k, beta, 0)?;
    } else {
        write_ifft_gfni(f, basis, lut, mats, k, beta, 0)?;
    }
    writeln!(f, "}}")?;
    writeln!(f)?;
    Ok(())
}

fn write_unrolled_kernel_gfni<G: Gf2p8 + fmt::Debug>(
    f: &mut impl Write,
    basis: &[G],
    sub_poly_luts: &[[G; FIELD_SIZE]; 8],
    subspace_points: &[G; FIELD_SIZE],
    gfni_mul_mats: &[u64; FIELD_SIZE],
) -> io::Result<()>
where
    u8: From<G>,
{
    writeln!(
        f,
        "\
        use additive_fft_reed_solomon_gf2p8::Gf2p8;
use super::{{butterfly_fwd_gfni, butterfly_inv_gfni}};
use std::arch::x86_64::*;
"
    )?;

    let cases: Vec<(usize, u8)> = (0..8).map(|a| (2usize << a, a)).collect();

    for (n, k) in cases {
        write_fft_gfni_case(
            f,
            basis,
            sub_poly_luts,
            gfni_mul_mats,
            n,
            k,
            G::zero(),
            false,
        )?;
        write_fft_gfni_case(
            f,
            basis,
            sub_poly_luts,
            gfni_mul_mats,
            n,
            k,
            G::zero(),
            true,
        )?;
    }

    let omega_cases: Vec<(usize, u8, usize)> =
        (0..8).map(|a| (2usize << a, a, 1usize << a)).collect();

    for (n, k, t) in omega_cases {
        let omega = subspace_points[t];
        write_fft_gfni_case(f, basis, sub_poly_luts, gfni_mul_mats, n, k, omega, true)?;
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
    let gfni_mul_mats: [u64; FIELD_SIZE] = gfni_mul_iter.collect::<Vec<_>>().try_into().unwrap();

    writeln!(f, "\npub const GFNI_MUL_TABLE: [u64; {}] = [", FIELD_SIZE).unwrap();
    for mat in gfni_mul_mats {
        writeln!(f, "    0x{:016x},", mat).unwrap();
    }
    writeln!(f, "];").unwrap();

    let (num_points, points_iter) = basis.iter_subspace_points();
    let subspace_points: [Gf2p8_11d; FIELD_SIZE] =
        points_iter.collect::<Vec<_>>().try_into().unwrap();

    write!(f, "\npub const CANTOR_SUBSPACE: [u8; {}] = [", num_points).unwrap();
    write_points(&mut f, subspace_points.into_iter(), false);

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
        &subspace_points,
        &exp_table,
        &log_table,
    )
    .expect("LUT kernel");

    let dest_kernel_gfni = Path::new(&out_dir).join("unrolled_gfni_kernel_11d.rs");
    let mut fkg = BufWriter::new(File::create(&dest_kernel_gfni).unwrap());
    write_unrolled_kernel_gfni(
        &mut fkg,
        basis.as_ref(),
        sub_poly_luts8,
        &subspace_points,
        &gfni_mul_mats,
    )
    .expect("GFNI kernel");

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

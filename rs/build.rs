use additive_fft_reed_solomon_gf2p8::{
    CantorBasis, CantorBasis11d, EXP_TABLE_SIZE, FIELD_SIZE, Gf2p8, Gf2p8_11d,
};
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::marker::PhantomData;
use std::path::Path;

fn write_copy_half(f: &mut impl Write, offset: usize, half: usize) -> io::Result<()> {
    let end = offset + half * 2;
    write!(
        f,
        "    {{
        let (lo, hi) = shards[{offset} * shard_len..{end} * shard_len]
            .split_at_mut({half} * shard_len);
        hi.copy_from_slice(lo);
    }}\n"
    )
}

struct UnrollTarget<G> {
    name: &'static str,
    cfg: &'static str,
    attr: &'static str,
    get_mul_table: fn(usize) -> String,
    mul_table_import: &'static str,
    g: &'static str,
    _phant: PhantomData<G>,
}

impl<G: Gf2p8 + fmt::Debug> UnrollTarget<G> {
    fn write_butterfly_fwd(
        &self,
        f: &mut impl Write,
        twiddle: G,
        offset: usize,
        half: usize,
    ) -> io::Result<()> {
        let end = offset + half * 2;

        let fwd_op = if twiddle == G::zero() {
            "for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = bi.add(*ai); }"
        } else {
            "butterfly_fwd(a, b, shard_len, m);"
        };

        let fwd_op_half1 = if twiddle == G::zero() {
            "for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = bi.add(*ai); }"
        } else {
            "butterfly_fwd(lo, hi, shard_len, m);"
        };

        writeln!(f, "    {{")?;
        if twiddle != G::zero() {
            writeln!(
                f,
                "        let m = {};",
                (self.get_mul_table)(twiddle.into_usize())
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

    fn write_butterfly_inv(
        &self,
        f: &mut impl Write,
        twiddle: G,
        offset: usize,
        half: usize,
    ) -> io::Result<()> {
        let end = offset + half * 2;

        let inv_op = if twiddle == G::zero() {
            "for (ai, bi) in a.iter().zip(b.iter_mut()) { *bi = ai.add(*bi); }"
        } else {
            "butterfly_inv(a, b, shard_len, m);"
        };

        let inv_op_half1 = if twiddle == G::zero() {
            "for (ai, bi) in lo.iter().zip(hi.iter_mut()) { *bi = ai.add(*bi); }"
        } else {
            "butterfly_inv(lo, hi, shard_len, m);"
        };

        writeln!(f, "    {{")?;
        if twiddle != G::zero() {
            writeln!(
                f,
                "        let m = {};",
                (self.get_mul_table)(twiddle.into_usize()),
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

    fn write_fft(
        &self,
        f: &mut impl Write,
        basis: &[G],
        lut: &[[G; FIELD_SIZE]; 8],
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

        self.write_butterfly_fwd(f, twiddle, offset, half)?;

        if l == 0 {
            return Ok(());
        }

        let next_beta = beta.add(basis[l as usize]);
        self.write_fft(f, basis, lut, l - 1, beta, offset)?;
        self.write_fft(f, basis, lut, l - 1, next_beta, offset + half)?;
        Ok(())
    }

    fn write_ifft(
        &self,
        f: &mut impl Write,
        basis: &[G],
        lut: &[[G; FIELD_SIZE]; 8],
        l: u8,
        beta: G,
        offset: usize,
    ) -> io::Result<()> {
        let half = 1 << l;
        if l == 0 {
            let twiddle = beta;
            self.write_butterfly_inv(f, twiddle, offset, half)?;
            return Ok(());
        }

        let next_beta = beta.add(basis[l as usize]);
        self.write_ifft(f, basis, lut, l - 1, beta, offset)?;
        self.write_ifft(f, basis, lut, l - 1, next_beta, offset + (1 << l))?;

        let twiddle = lut[l as usize][beta.into_usize()];
        self.write_butterfly_inv(f, twiddle, offset, 1 << l)?;
        Ok(())
    }

    fn write_fft_case(
        &self,
        f: &mut impl Write,
        basis: &[G],
        lut: &[[G; FIELD_SIZE]; 8],
        n: usize,
        k: u8,
        beta: G,
        is_ifft: bool,
    ) -> io::Result<()> {
        write!(f, "{}", self.cfg)?;
        write!(f, "{}", self.attr)?;
        writeln!(
            f,
            "pub fn {}fft_sharded_{}_{n}{}(shards: &mut [{}], shard_len: usize) {{",
            if is_ifft { "i" } else { "" },
            self.name,
            if beta != G::zero() {
                format!("_{:02x}", beta.into())
            } else {
                "".to_string()
            },
            self.g,
        )?;
        writeln!(f, "    debug_assert_eq!(shards.len(), {n} * shard_len);")?;
        if !is_ifft {
            self.write_fft(f, basis, lut, k, beta, 0)?;
        } else {
            self.write_ifft(f, basis, lut, k, beta, 0)?;
        }
        writeln!(f, "}}")?;
        writeln!(f)?;
        Ok(())
    }

    fn write_fft_zero_padded(
        &self,
        f: &mut impl Write,
        basis: &[G],
        lut: &[[G; FIELD_SIZE]; 8],
        l: u8,
        beta: G,
        offset: usize,
        log_support: u8,
    ) -> io::Result<()> {
        if log_support > l {
            return self.write_fft(f, basis, lut, l, beta, offset);
        }

        let half = 1 << l;
        write_copy_half(f, offset, half)?;

        if l == 0 {
            return Ok(());
        }

        let next_beta = beta.add(basis[l as usize]);
        self.write_fft_zero_padded(f, basis, lut, l - 1, beta, offset, log_support)?;
        let o2 = offset + half;
        self.write_fft_zero_padded(f, basis, lut, l - 1, next_beta, o2, log_support)?;
        Ok(())
    }

    fn write_fft_zero_padded_case(
        &self,
        f: &mut impl Write,
        basis: &[G],
        lut: &[[G; FIELD_SIZE]; 8],
        k: u8,
        log_support: u8,
    ) -> io::Result<()> {
        let n = 2 << k;
        let support = 1 << log_support;
        write!(f, "{}", self.cfg)?;
        write!(f, "{}", self.attr)?;
        writeln!(
            f,
            "pub fn fft_sharded_zero_padded_{}_{n}_{support}(shards: &mut [{}], shard_len: usize) {{",
            self.name, self.g
        )?;
        writeln!(f, "    debug_assert_eq!(shards.len(), {n} * shard_len);")?;
        self.write_fft_zero_padded(f, basis, lut, k, G::zero(), 0, log_support)?;
        writeln!(f, "}}")?;
        writeln!(f)?;
        Ok(())
    }

    fn write_unrolled_kernel(
        &self,
        f: &mut impl Write,
        basis: &[G],
        sub_poly_luts: &[[G; FIELD_SIZE]; 8],
        subspace_points: &[G; FIELD_SIZE],
    ) -> io::Result<()>
    where
        u8: From<G>,
    {
        writeln!(
            f,
            "\
            use additive_fft_reed_solomon_gf2p8::{{Gf2p8, {}}};
use super::{{butterfly_fwd, butterfly_inv, {}}};
",
            self.g, self.mul_table_import
        )?;

        for k in 0..8 {
            let n = 2usize << k;
            self.write_fft_case(f, basis, sub_poly_luts, n, k, G::zero(), false)?;
            self.write_fft_case(f, basis, sub_poly_luts, n, k, G::zero(), true)?;
        }

        for k in 0..8 {
            let n = 2usize << k;
            let t = 1usize << k;
            let omega = subspace_points[t];
            self.write_fft_case(f, basis, sub_poly_luts, n, k, omega, true)?;
        }

        for k in 0..8 {
            for log_support in 0..=k {
                self.write_fft_zero_padded_case(f, basis, sub_poly_luts, k, log_support)?;
            }
        }

        Ok(())
    }
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

fn write_bytes(f: &mut impl Write, it: impl Iterator<Item = u8>, has_subarrays: bool) {
    for (i, b) in it.enumerate() {
        if i % 16 == 0 {
            write!(f, "\n    ").unwrap();
            if has_subarrays {
                write!(f, "    ").unwrap();
            }
        }
        write!(f, "0x{:02x}, ", b).unwrap();
    }
    if has_subarrays {
        writeln!(f, "\n    ],").unwrap();
    } else {
        writeln!(f, "\n];").unwrap();
    }
}

const AVX2: UnrollTarget<Gf2p8_11d> = UnrollTarget {
    name: "avx2",
    cfg: "#[cfg(any(native_avx2, feature = \"compile_avx2\"))]\n",
    attr: "#[target_feature(enable = \"avx2\")]\n",
    get_mul_table: |t| format!("&NIBBLE_MUL_TABLE[{t}]"),
    mul_table_import: "NIBBLE_MUL_TABLE",
    g: "Gf2p8_11d",
    _phant: PhantomData,
};

const GFNI: UnrollTarget<Gf2p8_11d> = UnrollTarget {
    name: "gfni",
    cfg: "#[cfg(any(native_gfni, feature = \"compile_gfni\"))]\n",
    attr: "#[target_feature(enable = \"avx512f,avx512bw,gfni\")]\n",
    get_mul_table: |t| format!("_mm512_set1_epi64(GFNI_MUL_TABLE[{t}] as i64)"),
    mul_table_import: "GFNI_MUL_TABLE",
    g: "Gf2p8_11d",
    _phant: PhantomData,
};

const LUT: UnrollTarget<Gf2p8_11d> = UnrollTarget {
    name: "lut",
    cfg: "",
    attr: "",
    get_mul_table: |t| format!("&MUL_TABLE[{t}]"),
    mul_table_import: "MUL_TABLE",
    g: "Gf2p8_11d",
    _phant: PhantomData,
};

const NEON: UnrollTarget<Gf2p8_11d> = UnrollTarget {
    name: "neon",
    cfg: "#[cfg(native_neon)]\n",
    attr: "",
    get_mul_table: |t| format!("&NIBBLE_MUL_TABLE[{t}]"),
    mul_table_import: "NIBBLE_MUL_TABLE",
    g: "Gf2p8_11d",
    _phant: PhantomData,
};

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("tables_11d.rs");
    let mut f = BufWriter::new(File::create(&dest_path).unwrap());

    let (exp_table, log_table) = Gf2p8_11d::exp_log_tables();
    let inv_table = Gf2p8_11d::inv_table(&exp_table, &log_table);
    write!(f, "\npub static EXP_TABLE: [u8; {}] = [", EXP_TABLE_SIZE).unwrap();
    write_points(&mut f, exp_table.into_iter(), false);
    write!(f, "\npub static LOG_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, log_table.into_iter(), false);
    write!(f, "\npub static INV_TABLE: [u8; {}] = [", FIELD_SIZE).unwrap();
    write_points(&mut f, inv_table.into_iter(), false);

    let basis = CantorBasis11d::new();

    write!(f, "\npub const CANTOR_BASIS: [u8; 8] = [").unwrap();
    write_points(&mut f, basis.into_iter(), false);

    let mul_table_iter =
        (0..FIELD_SIZE).map(|x| Gf2p8_11d(x as u8).make_mul_table(&exp_table, &log_table));
    writeln!(
        f,
        "\npub static MUL_TABLE: [[u8; {FIELD_SIZE}]; {FIELD_SIZE}] = ["
    )
    .unwrap();
    for t in mul_table_iter {
        write!(f, "    [").unwrap();
        write_bytes(&mut f, t.into_iter(), true);
    }
    writeln!(f, "];").unwrap();

    let gfni_mul_iter = Gf2p8_11d::iter_gfni_mul_matrices();
    let gfni_mul_mats: [u64; FIELD_SIZE] = gfni_mul_iter.collect::<Vec<_>>().try_into().unwrap();

    writeln!(f, "\npub static GFNI_MUL_TABLE: [u64; {}] = [", FIELD_SIZE).unwrap();
    for mat in gfni_mul_mats {
        writeln!(f, "    0x{:016x},", mat).unwrap();
    }
    writeln!(f, "];").unwrap();

    let nibble_mul_iter = Gf2p8_11d::iter_nibble_mul_tables();
    let nibble_mul_tables: [([u8; 16], [u8; 16]); FIELD_SIZE] =
        nibble_mul_iter.collect::<Vec<_>>().try_into().unwrap();

    writeln!(
        f,
        "\npub static NIBBLE_MUL_TABLE: [([u8; 16], [u8; 16]); {}] = [",
        FIELD_SIZE
    )
    .unwrap();
    for t in nibble_mul_tables {
        write!(f, "    ([").unwrap();
        for i in 0..16 {
            write!(f, "0x{:02x}, ", t.0[i]).unwrap();
        }
        write!(f, "],\n     [").unwrap();
        for i in 0..16 {
            write!(f, "0x{:02x}, ", t.1[i]).unwrap();
        }
        writeln!(f, "]),").unwrap();
    }
    writeln!(f, "];").unwrap();

    let (num_points, points_iter) = basis.iter_subspace_points();
    let subspace_points: [Gf2p8_11d; FIELD_SIZE] =
        points_iter.collect::<Vec<_>>().try_into().unwrap();

    write!(f, "\npub static CANTOR_SUBSPACE: [u8; {}] = [", num_points).unwrap();
    write_points(&mut f, subspace_points.into_iter(), false);

    let sub_poly_luts = basis.gen_all_subspace_poly_luts();

    writeln!(
        f,
        "\npub static SUBSPACE_POLY_VALUES: [[u8; {}]; 9] = [",
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
    LUT.write_unrolled_kernel(&mut fkl, basis.as_ref(), sub_poly_luts8, &subspace_points)
        .expect("LUT kernel");

    let dest_kernel_avx2 = Path::new(&out_dir).join("unrolled_avx2_kernel_11d.rs");
    let mut fkg = BufWriter::new(File::create(&dest_kernel_avx2).unwrap());
    AVX2.write_unrolled_kernel(&mut fkg, basis.as_ref(), sub_poly_luts8, &subspace_points)
        .expect("AVX2 kernel");

    let dest_kernel_neon = Path::new(&out_dir).join("unrolled_neon_kernel_11d.rs");
    let mut fkg = BufWriter::new(File::create(&dest_kernel_neon).unwrap());
    NEON.write_unrolled_kernel(&mut fkg, basis.as_ref(), sub_poly_luts8, &subspace_points)
        .expect("NEON kernel");

    let dest_kernel_gfni = Path::new(&out_dir).join("unrolled_gfni_kernel_11d.rs");
    let mut fkg = BufWriter::new(File::create(&dest_kernel_gfni).unwrap());
    GFNI.write_unrolled_kernel(&mut fkg, basis.as_ref(), sub_poly_luts8, &subspace_points)
        .expect("GFNI kernel");

    // CPU feature detection
    let target = std::env::var("TARGET").unwrap();
    let host = std::env::var("HOST").unwrap();
    let native = target == host;

    if target.starts_with("aarch64") {
        // NEON is included as standard on Aarch64
        println!("cargo:rustc-cfg=native_neon");
    }

    // Feature detection only works when the build machine is the target machine. Cross builds must
    // use the compile_* features to opt in.
    #[cfg(target_arch = "x86_64")]
    if native {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("gfni")
        {
            println!("cargo:rustc-cfg=native_gfni");
        }
        if is_x86_feature_detected!("avx2") {
            println!("cargo:rustc-cfg=native_avx2");
        }
    }

    // Emit the lint checker tweaks on all platforms.
    println!("cargo:rustc-check-cfg=cfg(native_gfni)");
    println!("cargo:rustc-check-cfg=cfg(native_avx2)");
    println!("cargo:rustc-check-cfg=cfg(native_neon)");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/gf2p8lut.rs");
    println!("cargo:rerun-if-changed=src/poly_11d_lut.rs");
}

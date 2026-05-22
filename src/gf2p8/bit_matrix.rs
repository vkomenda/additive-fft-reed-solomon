#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitMatrix(pub [u8; 8]);

impl BitMatrix {
    /// Software emulation of vgf2p8affineqb (b=0).
    /// This performs: vector_out = self · vector_in, over GF(2).
    /// Matches vgf2p8affineqb: Each u8 in the array is one row of the 8x8 matrix.
    pub fn apply(&self, v: u8) -> u8 {
        let mut res = 0u8;
        for i in 0..8 {
            let bitmul = v & self.0[i];
            let dotprod = bitmul.count_ones() & 1;
            res |= (dotprod as u8) << i;
        }
        res
    }

    /// Transpose columns to rows for vgf2p8affineqb.
    pub fn transpose(&self) -> Self {
        let mut rows = [0u8; 8];
        for r in 0..8 {
            let mut row_val = 0u8;
            for c in 0..8 {
                let bit = (self.0[c] >> r) & 1;
                row_val |= bit << c;
            }
            rows[r] = row_val;
        }

        Self(rows)
    }

    /// Invert the bit matrix using Gaussian elimination.
    pub fn inv(&self) -> Option<Self> {
        let mut mat = self.0;
        let mut inv = [1, 2, 4, 8, 16, 32, 64, 128]; // Identity

        for i in 0..8 {
            // Find pivot
            let mut pivot = i;
            while pivot < 8 && (mat[pivot] & (1 << i)) == 0 {
                pivot += 1;
            }
            if pivot == 8 {
                return None;
            }

            mat.swap(i, pivot);
            inv.swap(i, pivot);

            for j in 0..8 {
                if i != j && (mat[j] & (1 << i)) != 0 {
                    mat[j] ^= mat[i];
                    inv[j] ^= inv[i];
                }
            }
        }
        Some(Self(inv))
    }

    #[cfg(test)]
    pub fn rank(&self) -> usize {
        let mut rows = self.0;
        let mut rank = 0;
        for i in 0..8 {
            let pivot = (rank..8).find(|&r| (rows[r] & (1 << i)) != 0);
            if let Some(p) = pivot {
                rows.swap(rank, p);
                for j in 0..8 {
                    if j != rank && (rows[j] & (1 << i)) != 0 {
                        rows[j] ^= rows[rank];
                    }
                }
                rank += 1;
            }
        }
        rank
    }

    pub fn to_gfni_u64(self) -> u64 {
        u64::from_be_bytes(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf2p8::{Gf2p8, Gf2p8_11d};

    #[test]
    fn identity_matrix_apply() {
        let id = BitMatrix([1, 2, 4, 8, 16, 32, 64, 128]);
        for i in 0..=255 {
            assert_eq!(id.apply(i), i, "Identity failed at {}", i);
        }
    }

    // M(a ^ b) == M(a) ^ M(b)
    #[test]
    fn apply_linearity() {
        let m = Gf2p8_11d::from(0x42).into_mul_matrix();
        let a = 0x57;
        let b = 0x83;

        let res_combined = m.apply(a ^ b);
        let res_separate = m.apply(a) ^ m.apply(b);

        assert_eq!(res_combined, res_separate);
    }

    #[test]
    fn double_transpose() {
        let m = BitMatrix([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]);
        let mt = m.transpose();
        let mtt = mt.transpose();

        for i in 0..8 {
            assert_eq!(m.0[i], mtt.0[i]);
        }
    }

    #[test]
    fn field_consistency() {
        for a_val in [0x02, 0x42, 0x80, 0xFF] {
            let a = Gf2p8_11d::from(a_val);
            let matrix = a.into_mul_matrix();
            for b_val in [0x01, 0x02, 0x55, 0xAA] {
                let b = Gf2p8_11d::from(b_val);
                let expected = u8::from(a.mul(b));
                let actual = matrix.apply(b_val);
                assert_eq!(
                    actual, expected,
                    "Matrix for {:#x} failed to multiply {:#x}. Expected {:#x}, got {:#x}",
                    a_val, b_val, expected, actual
                );
            }
        }
    }
}

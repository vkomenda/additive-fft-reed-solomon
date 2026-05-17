pub mod bit_matrix;
pub mod generic;

pub use generic::{CantorBasis, Gf2p8};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Gf2p8_11d(pub u8);

impl From<u8> for Gf2p8_11d {
    fn from(a: u8) -> Self {
        Self(a)
    }
}

impl From<Gf2p8_11d> for u8 {
    fn from(a: Gf2p8_11d) -> u8 {
        a.0
    }
}

impl Gf2p8 for Gf2p8_11d {
    const POLY: u16 = 0x11d;
    const PRIM_ELEM: Self = Self(2);
}

#[derive(Copy, Clone)]
pub struct CantorBasis11d(pub [Gf2p8_11d; 8]);

impl FromIterator<Gf2p8_11d> for CantorBasis11d {
    fn from_iter<I: IntoIterator<Item = Gf2p8_11d>>(iter: I) -> Self {
        let mut arr = [0u8.into(); 8];
        for (slot, item) in arr.iter_mut().zip(iter) {
            *slot = item;
        }
        Self(arr)
    }
}

impl IntoIterator for CantorBasis11d {
    type Item = Gf2p8_11d;
    type IntoIter = std::array::IntoIter<Gf2p8_11d, 8>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl CantorBasis<Gf2p8_11d> for CantorBasis11d {}

impl AsRef<[Gf2p8_11d]> for CantorBasis11d {
    fn as_ref(&self) -> &[Gf2p8_11d] {
        self.0.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_of_1() {
        assert!(!Gf2p8_11d::from(1).trace());
    }

    #[test]
    fn add_once() {
        let a = Gf2p8_11d::from(0x57);
        let b = Gf2p8_11d::from(0x83);
        assert_eq!(u8::from(a.add(b)), 0x57 ^ 0x83);
        assert_eq!(u8::from(a.add(a)), 0);
    }

    #[test]
    fn basic_mul() {
        let a = Gf2p8_11d::from(0x57);
        assert_eq!(u8::from(a.mul(0.into())), 0);
        assert_eq!(u8::from(a.mul(1.into())), 0x57);
    }

    #[test]
    fn wrap_mul() {
        let two = Gf2p8_11d::from(2);
        let high_bit = Gf2p8_11d::from(0x80);
        assert_eq!(u8::from(two.mul(high_bit)), 0x1d);
    }

    #[test]
    fn mul_inv() {
        for i in 1..=255 {
            let a = Gf2p8_11d::from(i);
            let mut inv = Gf2p8_11d::from(0);
            for j in 1..=255 {
                if u8::from(a.mul(j.into())) == 1 {
                    inv = j.into();
                    break;
                }
            }
            assert_ne!(u8::from(inv), 0, "Inverse not found for {}", i);
        }
    }

    #[test]
    fn test_cantor_basis_properties() {
        let basis = CantorBasis11d::new();

        assert_eq!(basis.0.len(), 8);

        // Check the Chain Property: v_i^2 + v_i = v_{i-1}
        for i in 1..basis.0.len() {
            let v_curr = basis.0[i];
            let v_prev = basis.0[i - 1];

            // v^2 + v
            let lhs = v_curr.mul(v_curr).add(v_curr);
            assert_eq!(lhs, v_prev, "Chain property failed at index {}", i);
        }

        // Check trace conditions
        // For the sequence to be extendable, Tr(v_i) must be 0 for i < 7.
        for i in 0..7 {
            assert!(
                !basis.0[i].trace(),
                "Trace of v_{} must be 0 to allow extension",
                i
            );
        }

        assert!(
            is_linearly_independent(&basis),
            "Basis elements are not linearly independent"
        );
    }

    /// Helper to check linear independence over GF(2) using Gaussian Elimination
    fn is_linearly_independent(basis: &CantorBasis11d) -> bool {
        let mut matrix = basis.0.to_vec();
        let mut rank = 0;

        for bit in (0..8).rev() {
            // Find a row with a 1 at the current bit position
            let mut pivot = None;
            for i in rank..matrix.len() {
                if (u8::from(matrix[i]) >> bit) & 1 != 0 {
                    pivot = Some(i);
                    break;
                }
            }

            if let Some(p) = pivot {
                matrix.swap(rank, p);
                for i in 0..matrix.len() {
                    if i != rank && (u8::from(matrix[i]) >> bit) & 1 != 0 {
                        matrix[i] = matrix[i].add(matrix[rank]);
                    }
                }
                rank += 1;
            }
        }
        rank == basis.0.len()
    }

    #[test]
    fn basis_subspace_points() {
        let basis = CantorBasis11d::new();
        let elements: Vec<Gf2p8_11d> = basis.into_iter().collect();

        assert!((0..8).all(|i| basis.get_subspace_point(2u8.pow(i)) == elements[i as usize]));
    }
}

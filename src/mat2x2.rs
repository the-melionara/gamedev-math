#[macro_export]
macro_rules! gen_mat2x2 {
    ($ident:ident, $vec2:ident, $typ:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ident {
            rows: [[$typ; 2]; 2],
        }

        impl $ident {
            pub const IDENTITY: Self = Self { rows: [[1.0, 0.0], [0.0, 1.0]] };

            pub fn new(rows: [[$typ; 2]; 2]) -> Self {
                Self { rows }
            }

            pub fn new_with_vec(row0: $vec2, row1: $vec2) -> Self {
                Self { rows: [
                    [row0.0, row0.1],
                    [row1.0, row1.1],
                ] }
            }

            pub fn row(&self, row: usize) -> $vec2 {
                assert!(row < 2);
                return $vec2(self.rows[row][0], self.rows[row][1]);
            }

            pub fn col(&self, col: usize) -> $vec2 {
                assert!(col < 2);
                return $vec2(self.rows[0][col], self.rows[1][col]);
            }

            pub fn determinant(&self) -> $typ {
                return self.rows[0][0] * self.rows[1][1] - self.rows[1][0] * self.rows[0][1];
            }

            pub fn transp(&self) -> Self {
                Self { rows: [
                    [self.rows[0][0], self.rows[1][0]],
                    [self.rows[0][1], self.rows[1][1]],
                ] }
            }

            pub fn invert(&mut self) -> Option<()> {
                let det = self.determinant();
                if det == 0.0 {
                    return None;
                }

                let inv_det = 1.0 / det;
                let src = self.rows.clone();

                self.rows = [
                    [  src[1][1] * inv_det, -src[1][0] * inv_det ],
                    [ -src[0][1] * inv_det,  src[0][0] * inv_det ],
                ];

                Some(())
            }

            pub fn inverse(&self) -> Option<Self> {
                let mut res = self.clone();
                res.invert()?;
                return Some(res);
            }

            pub fn ptr(&self) -> *const $typ {
                return self.rows.as_ptr() as *const $typ;
            }
        }

        impl std::ops::Mul for &$ident {
            type Output = $ident;

            fn mul(self, rhs: Self) -> Self::Output {
                let mut res = $ident { rows: [[0.0; 2]; 2] };
                for i in 0..2 {
                    for j in 0..2 {
                        res.rows[i][j] = self.row(i).dot(rhs.col(i));
                    }
                }
                return res;
            }
        }

        impl std::ops::Mul<$vec2> for &$ident {
            type Output = $vec2;

            fn mul(self, rhs: $vec2) -> Self::Output {
                return $vec2(
                    self.row(0).dot(rhs),
                    self.row(1).dot(rhs),
                );
            }
        }

        impl std::ops::Index<(usize, usize)> for $ident {
            type Output = $typ;

            fn index(&self, index: (usize, usize)) -> &Self::Output {
                assert!(index.0 < 2 && index.1 < 2, "Index out of bounds. (Index was {},{}; Size was 2x2)", index.0, index.1);
                
                return &self.rows[index.0][index.1];
            }
        }

        impl std::ops::IndexMut<(usize, usize)> for $ident {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                assert!(index.0 < 2 && index.1 < 2, "Index out of bounds. (Index was {},{}; Size was 2x2)", index.0, index.1);

                return &mut self.rows[index.0][index.1];
            }
        }
    };
}
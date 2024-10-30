#[macro_export]
macro_rules! gen_mat3x3 {
    ($ident:ident, $vec3:ident, $typ:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ident {
            rows: [[$typ; 3]; 3],
        }

        impl $ident {
            pub const IDENTITY: Self = Self { rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };

            pub fn new(rows: [[$typ; 3]; 3]) -> Self {
                Self { rows }
            }

            pub fn new_with_vec(row0: $vec3, row1: $vec3, row2: $vec3) -> Self {
                Self { rows: [
                    [row0.0, row0.1, row0.2],
                    [row1.0, row1.1, row1.2],
                    [row2.0, row2.1, row2.2],
                ] }
            }

            pub fn row(&self, row: usize) -> $vec3 {
                assert!(row < 3);
                return $vec3(self.rows[row][0], self.rows[row][1], self.rows[row][2]);
            }

            pub fn col(&self, col: usize) -> $vec3 {
                assert!(col < 3);
                return $vec3(self.rows[0][col], self.rows[1][col], self.rows[2][col]);
            }

            pub fn determinant(&self) -> $typ {
                return self.rows[0][0] * (self.rows[1][1] * self.rows[2][2] - self.rows[2][1] * self.rows[1][2]) -
                    self.rows[0][1] * (self.rows[1][0] * self.rows[2][2] - self.rows[2][0] * self.rows[1][2]) +
                    self.rows[0][2] * (self.rows[1][0] * self.rows[2][1] - self.rows[2][0] * self.rows[1][1])
            }

            pub fn transp(&self) -> Self {
                Self { rows: [
                    [self.rows[0][0], self.rows[1][0], self.rows[2][0]],
                    [self.rows[0][1], self.rows[1][1], self.rows[2][1]],
                    [self.rows[0][2], self.rows[1][2], self.rows[2][2]],
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
                    [
                        (src[1][2] * src[2][1] - src[1][1] * src[2][2]) * inv_det,
                        -(src[0][2] * src[2][1] - src[0][1] * src[2][2]) * inv_det,
                        (src[0][2] * src[1][1] - src[0][1] * src[1][2]) * inv_det,
                    ],
                    [
                        -(src[1][2] * src[2][0] - src[1][0] * src[2][2]) * inv_det,
                        (src[0][2] * src[2][0] - src[0][0] * src[2][2]) * inv_det,
                        -(src[0][2] * src[1][0] - src[0][0] * src[1][2]) * inv_det,
                    ],
                    [
                        (src[1][1] * src[2][0] - src[1][0] * src[2][1]) * inv_det,
                        -(src[0][1] * src[2][0] - src[0][0] * src[2][1]) * inv_det,
                        (src[0][1] * src[1][0] - src[0][0] * src[1][1]) * inv_det,
                    ],
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
                let mut res = $ident { rows: [[0.0; 3]; 3] };
                for i in 0..3 {
                    for j in 0..3 {
                        res.rows[i][j] = self.row(i).dot(rhs.col(i));
                    }
                }
                return res;
            }
        }

        impl std::ops::Mul<$vec3> for &$ident {
            type Output = $vec3;

            fn mul(self, rhs: $vec3) -> Self::Output {
                return $vec3(
                    self.row(0).dot(rhs),
                    self.row(1).dot(rhs),
                    self.row(2).dot(rhs),
                );
            }
        }

        impl std::ops::Index<(usize, usize)> for $ident {
            type Output = $typ;

            fn index(&self, index: (usize, usize)) -> &Self::Output {
                assert!(index.0 < 3 && index.1 < 3, "Index out of bounds. (Index was {},{}; Size was 3x3)", index.0, index.1);
                
                return &self.rows[index.0][index.1];
            }
        }

        impl std::ops::IndexMut<(usize, usize)> for $ident {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                assert!(index.0 < 3 && index.1 < 3, "Index out of bounds. (Index was {},{}; Size was 3x3)", index.0, index.1);

                return &mut self.rows[index.0][index.1];
            }
        }
    };
}

#[macro_export]
macro_rules! impl_tf3x3 {
    ($ident:ident, $vec2:ident, $typ:ty) => {
        impl $ident {
            pub fn tf_matrix(pos: $vec2, rot: $typ, scale: $vec2) -> Self {
                let right = scale.xvec().rotate(rot);
                let up = scale.yvec().rotate(rot);

                return Self { rows: [
                    [right.0, up.0, pos.0],
                    [right.1, up.1, pos.1],
                    [0.0, 0.0, 1.0],
                ]}
            }
        }
    };
}

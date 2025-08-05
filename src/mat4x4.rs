#[macro_export]
macro_rules! gen_mat4x4 {
    ($ident:ident, $vec4:ident, $typ:ty) => {
        #[repr(C)]
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ident {
            rows: [[$typ; 4]; 4],
        }

        impl $ident {
            pub const IDENTITY: Self = Self { rows: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] };

            pub fn new(rows: [[$typ; 4]; 4]) -> Self {
                Self { rows }
            }

            pub fn new_with_vec(row0: $vec4, row1: $vec4, row2: $vec4, row3: $vec4) -> Self {
                Self { rows: [
                    [row0.0, row0.1, row0.2, row0.3],
                    [row1.0, row1.1, row1.2, row1.3],
                    [row2.0, row2.1, row2.2, row2.3],
                    [row3.0, row3.1, row3.2, row3.3],
                ] }
            }

            pub fn row(&self, row: usize) -> $vec4 {
                assert!(row < 4);
                return $vec4(self.rows[row][0], self.rows[row][1], self.rows[row][2], self.rows[row][3]);
            }

            pub fn col(&self, col: usize) -> $vec4 {
                assert!(col < 4);
                return $vec4(self.rows[0][col], self.rows[1][col], self.rows[2][col], self.rows[3][col]);
            }

            pub fn determinant(&self) -> $typ {
                let s0 = self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)];
                let s1 = self[(0, 0)] * self[(1, 2)] - self[(1, 0)] * self[(0, 2)];
                let s2 = self[(0, 0)] * self[(1, 3)] - self[(1, 0)] * self[(0, 3)];
                let s3 = self[(0, 1)] * self[(1, 2)] - self[(1, 1)] * self[(0, 2)];
                let s4 = self[(0, 1)] * self[(1, 3)] - self[(1, 1)] * self[(0, 3)];
                let s5 = self[(0, 2)] * self[(1, 3)] - self[(1, 2)] * self[(0, 3)];
                let c5 = self[(2, 2)] * self[(3, 3)] - self[(3, 2)] * self[(2, 3)];
                let c4 = self[(2, 1)] * self[(3, 3)] - self[(3, 1)] * self[(2, 3)];
                let c3 = self[(2, 1)] * self[(3, 2)] - self[(3, 1)] * self[(2, 2)];
                let c2 = self[(2, 0)] * self[(3, 3)] - self[(3, 0)] * self[(2, 3)];
                let c1 = self[(2, 0)] * self[(3, 2)] - self[(3, 0)] * self[(2, 2)];
                let c0 = self[(2, 0)] * self[(3, 1)] - self[(3, 0)] * self[(2, 1)];

                return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
            }

            pub fn transp(&self) -> Self {
                Self { rows: [
                    [self.rows[0][0], self.rows[1][0], self.rows[2][0], self.rows[3][0]],
                    [self.rows[0][1], self.rows[1][1], self.rows[2][1], self.rows[3][1]],
                    [self.rows[0][2], self.rows[1][2], self.rows[2][2], self.rows[3][2]],
                    [self.rows[0][3], self.rows[1][3], self.rows[2][3], self.rows[3][3]],
                ] }
            }

            pub fn invert(&mut self) -> Option<()> {
                let s0 = self[(0, 0)] * self[(1, 1)] - self[(1, 0)] * self[(0, 1)];
                let s1 = self[(0, 0)] * self[(1, 2)] - self[(1, 0)] * self[(0, 2)];
                let s2 = self[(0, 0)] * self[(1, 3)] - self[(1, 0)] * self[(0, 3)];
                let s3 = self[(0, 1)] * self[(1, 2)] - self[(1, 1)] * self[(0, 2)];
                let s4 = self[(0, 1)] * self[(1, 3)] - self[(1, 1)] * self[(0, 3)];
                let s5 = self[(0, 2)] * self[(1, 3)] - self[(1, 2)] * self[(0, 3)];
                let c5 = self[(2, 2)] * self[(3, 3)] - self[(3, 2)] * self[(2, 3)];
                let c4 = self[(2, 1)] * self[(3, 3)] - self[(3, 1)] * self[(2, 3)];
                let c3 = self[(2, 1)] * self[(3, 2)] - self[(3, 1)] * self[(2, 2)];
                let c2 = self[(2, 0)] * self[(3, 3)] - self[(3, 0)] * self[(2, 3)];
                let c1 = self[(2, 0)] * self[(3, 2)] - self[(3, 0)] * self[(2, 2)];
                let c0 = self[(2, 0)] * self[(3, 1)] - self[(3, 0)] * self[(2, 1)];

                let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
                if det == 0.0 {
                    return None;
                }

                let inv_det = 1.0 / det;

                self.rows = [
                    [
                        (self[(1,1)] * c5 - self[(1,2)] * c4 + self[(1,3)] * c3) * inv_det,
                        (-self[(0,1)] * c5 + self[(0,2)] * c4 - self[(0,3)] * c3) * inv_det,
                        (self[(3,1)] * s5 - self[(3,2)] * s4 + self[(3,3)] * s3) * inv_det,
                        (-self[(2,1)] * s5 + self[(2,2)] * s4 - self[(2,3)] * s3) * inv_det,
                    ],
                    [
                        (-self[(1,0)] * c5 + self[(1,2)] * c2 - self[(1,3)] * c1) * inv_det,
                        (self[(0,0)] * c5 - self[(0,2)] * c2 + self[(0,3)] * c1) * inv_det,
                        (-self[(3,0)] * s5 + self[(3,2)] * s2 - self[(3,3)] * s1) * inv_det,
                        (self[(2,0)] * s5 - self[(2,2)] * s2 + self[(2,3)] * s1) * inv_det,
                    ],
                    [
                        (self[(1,0)] * c4 - self[(1,1)] * c2 + self[(1,3)] * c0) * inv_det,
                        (-self[(0,0)] * c4 + self[(0,1)] * c2 - self[(0,3)] * c0) * inv_det,
                        (self[(3,0)] * s4 - self[(3,1)] * s2 + self[(3,3)] * s0) * inv_det,
                        (-self[(2,0)] * s4 + self[(2,1)] * s2 - self[(2,3)] * s0) * inv_det,
                    ],
                    [
                        (-self[(1,0)] * c3 + self[(1,1)] * c1 - self[(1,2)] * c0) * inv_det,
                        (self[(0,0)] * c3 - self[(0,1)] * c1 + self[(0,2)] * c0) * inv_det,
                        (-self[(3,0)] * s3 + self[(3,1)] * s1 - self[(3,2)] * s0) * inv_det,
                        (self[(2,0)] * s3 - self[(2,1)] * s1 + self[(2,2)] * s0) * inv_det,
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
                let mut res = $ident { rows: [[0.0; 4]; 4] };
                for i in 0..4 {
                    for j in 0..4 {
                        res.rows[i][j] = self.row(i).dot(rhs.col(j));
                    }
                }
                return res;
            }
        }

        impl std::ops::Mul<$vec4> for &$ident {
            type Output = $vec4;

            fn mul(self, rhs: $vec4) -> Self::Output {
                return $vec4(
                    self.row(0).dot(rhs),
                    self.row(1).dot(rhs),
                    self.row(2).dot(rhs),
                    self.row(3).dot(rhs),
                );
            }
        }

        impl std::ops::Index<(usize, usize)> for $ident {
            type Output = $typ;

            fn index(&self, index: (usize, usize)) -> &Self::Output {
                assert!(index.0 < 4 && index.1 < 4, "Index out of bounds. (Index was {},{}; Size was 4x4)", index.0, index.1);

                return &self.rows[index.0][index.1];
            }
        }

        impl std::ops::IndexMut<(usize, usize)> for $ident {
            fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
                assert!(index.0 < 4 && index.1 < 4, "Index out of bounds. (Index was {},{}; Size was 4x4)", index.0, index.1);

                return &mut self.rows[index.0][index.1];
            }
        }
    };
}

#[macro_export]
macro_rules! impl_tf4x4 {
    ($ident:ident, $vec3:ident, $quat:ident, $typ:ty) => {
        impl $ident {
            pub fn tf_matrix(pos: $vec3, rot: $quat, scale: $vec3) -> Self {
                let right = scale.xvec().rotate(rot);
                let up = scale.yvec().rotate(rot);
                let forw = scale.zvec().rotate(rot);

                return Self { rows: [
                    [right.0, up.0, forw.0, pos.0],
                    [right.1, up.1, forw.1, pos.1],
                    [right.2, up.2, forw.2, pos.2],
                    [0.0, 0.0, 0.0, 1.0],
                ]}
            }

            pub fn proj_matrix(aspect_ratio: $typ, fov: $typ, near: $typ, far: $typ) -> Self {
                let tan = (fov * 0.5).tan();
                return Self { rows: [
                    [1.0 / (aspect_ratio * tan), 0.0,       0.0,                     0.0                       ],
                    [0.0,                        1.0 / tan, 0.0,                     0.0                       ],
                    [0.0,                        0.0,       far / (far - near),      -far * near / (far - near)],
                    [0.0,                        0.0,       1.0,                     0.0                       ],
                ]};
            }

            pub fn proj_matrix_vk(aspect_ratio: $typ, fov: $typ, near: $typ, far: $typ) -> Self {
                let tan = (fov * 0.5).tan();
                return Self { rows: [
                    [1.0 / (aspect_ratio * tan),  0.0,        0.0,                 0.0                      ],
                    [0.0,                        -1.0 / tan,  0.0,                 0.0                      ],
                    [0.0,                         0.0,        far / (far - near), -far * near / (far - near)],
                    [0.0,                         0.0,        1.0,                 0.0                      ],
                ]};
            }
        }
    };
}

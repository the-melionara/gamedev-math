use std::ops::{Add, AddAssign, Mul, MulAssign, Div, DivAssign};

use crate::vector3::{Vector3f32, Vector3f64};

macro_rules! quat_gen {
    ($ident:ident, $vec:ident, $typ:ty) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $ident(pub $typ, pub $typ, pub $typ, pub $typ);

        impl $ident {
            pub const IDENT: Self = Self(1.0, 0.0, 0.0, 0.0);
            pub const I: Self = Self(0.0, 1.0, 0.0, 0.0);
            pub const J: Self = Self(0.0, 0.0, 1.0, 0.0);
            pub const K: Self = Self(0.0, 0.0, 0.0, 1.0);

            pub fn rotator(axis_angle: $vec) -> Self {
                let theta = axis_angle.magnitude() * 0.5;
                let axis = axis_angle.normalized();

                return Self::from_split(theta.cos(), axis * theta.sin());
            }

            // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_(in_3-2-1_sequence)_to_quaternion_conversion
            // function is slightly modified because for some reason the wikipedia version mixes up some things, idk why
            pub fn rotator_euler(euler: $vec) -> Self {
                let pitch = -euler.0; // gl fuckery
                let yaw = euler.1;
                let roll = euler.2;
                
                let (sp, cp) = (pitch * 0.5).sin_cos();
                let (sy, cy) = (yaw * 0.5).sin_cos();
                let (sr, cr) = (roll * 0.5).sin_cos();

                return Self(
                    cp * cy * cr + sp * sy * sr,
                    sp * cy * cr - cp * sy * sr,
                    cp * sy * cr + sp * cy * sr,
                    cp * cy * sr - sp * sy * cr,
                );
            }

            pub const fn from_split(real: $typ, im: $vec) -> Self {
                return Self(real, im.0, im.1, im.2);
            }

            pub const fn inverted(self) -> Self {
                let inv_sqr_norm = 1.0 / (self.0 * self.0 + self.1 * self.1 + self.2 * self.2 + self.3 * self.3);
                return Self(
                    self.0 * inv_sqr_norm,
                    self.1 * inv_sqr_norm,
                    self.2 * inv_sqr_norm,
                    self.3 * inv_sqr_norm,
                );
            }

            pub fn norm(self) -> $typ {
                return (self.0 * self.0 + self.1 * self.1 + self.2 * self.2 + self.3 * self.3).sqrt();
            }

            pub fn normalize(&mut self) {
                *self = *self / self.norm();
            }
            
            pub fn normalized(self) -> Self {
                let inv_norm = 1.0 / self.norm();
                return Self(
                    self.0 * inv_norm,
                    self.1 * inv_norm,
                    self.2 * inv_norm,
                    self.3 * inv_norm,
                );
            }

            pub const fn conjugate(&self) -> Self {
                return Self(self.0, -self.1, -self.2, -self.3);
            }

            pub const fn vector(&self) -> $vec {
                return $vec(self.1, self.2, self.3);
            }

            pub fn basis(&self) -> ($vec, $vec, $vec) {
                let x = $vec::RIGHT.rotate(*self).normalized();
                let y = $vec::UP.rotate(*self).normalized();

                return (
                    x,
                    y,
                    x.cross(y),
                )
            }

            pub fn forw(&self) -> $vec {
                return self.basis().2;
            }
        }

        impl Add for $ident {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                return Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3);
            }
        }

        impl AddAssign for $ident {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Mul<$typ> for $ident {
            type Output = Self;

            fn mul(self, rhs: $typ) -> Self::Output {
                return Self(self.0 * rhs, self.1 * rhs, self.2 * rhs, self.3 * rhs);
            }
        }

        impl MulAssign<$typ> for $ident {
            fn mul_assign(&mut self, rhs: $typ) {
                *self = *self * rhs;
            }
        }

        impl Div<$typ> for $ident {
            type Output = Self;

            fn div(self, rhs: $typ) -> Self::Output {
                let rhs = 1.0 / rhs;
                return Self(self.0 * rhs, self.1 * rhs, self.2 * rhs, self.3 * rhs);
            }
        }

        impl DivAssign<$typ> for $ident {
            fn div_assign(&mut self, rhs: $typ) {
                *self = *self / rhs;
            }
        }

        impl Mul for $ident {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                return Self(
                    self.0 * rhs.0 - self.1 * rhs.1 - self.2 * rhs.2 - self.3 * rhs.3,
                    self.0 * rhs.1 + self.1 * rhs.0 + self.2 * rhs.3 - self.3 * rhs.2,
                    self.0 * rhs.2 - self.1 * rhs.3 + self.2 * rhs.0 + self.3 * rhs.1,
                    self.0 * rhs.3 + self.1 * rhs.2 - self.2 * rhs.1 + self.3 * rhs.0,
                );
            }
        }

        impl MulAssign for $ident {
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl From<$vec> for $ident {
            fn from(value: $vec) -> Self {
                return Self(0.0, value.0, value.1, value.2)
            }
        }
    };
}

quat_gen!(Quaternionf32, Vector3f32, f32);
quat_gen!(Quaternionf64, Vector3f64, f64);
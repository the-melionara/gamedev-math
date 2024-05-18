use std::{fmt::Display, ops::{Add, AddAssign, MulAssign, SubAssign, DivAssign, Div, Mul, Neg, Sub}, simd::{cmp::SimdOrd, f32x4, f64x4, i32x4, num::{SimdFloat, SimdInt, SimdUint}, u32x4, StdFloat}};

#[cfg(feature = "vec2")]
use crate::vector2::{Vector2bool, Vector2f32, Vector2f64, Vector2i32, Vector2u32};

#[cfg(not(feature = "vec2"))] type Vector2bool = ();
#[cfg(not(feature = "vec2"))] type Vector2f32 = ();
#[cfg(not(feature = "vec2"))] type Vector2f64 = ();
#[cfg(not(feature = "vec2"))] type Vector2i32 = ();
#[cfg(not(feature = "vec2"))] type Vector2u32 = ();

#[cfg(feature = "quaternion")]
use crate::quaternion::{Quaternionf32, Quaternionf64};

#[cfg(not(feature = "quaternion"))] type Quaternionf32 = ();
#[cfg(not(feature = "quaternion"))] type Quaternionf64 = ();

macro_rules! vec_type_gen {
    ($ident:ident, $typ:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $ident(pub $typ, pub $typ, pub $typ);
    };
}

/// Implements unsigned constants
macro_rules! uvec_const_gen {
    ($ident:ident, $typ:ty, $zero:literal, $one:literal) => {
        impl $ident {
            pub const ZERO: Self = Self($zero, $zero, $zero);
            pub const UP: Self = Self::up($one);
            pub const RIGHT: Self = Self::right($one);
            pub const FORW: Self = Self::forw($one);
            pub const ONE: Self = Self::one($one);
        
            pub const fn up(fact: $typ) -> Self {
                return Self($zero, fact, $zero);
            }
        
            pub const fn right(fact: $typ) -> Self {
                return Self(fact, $zero, $zero);
            }

            pub const fn forw(fact: $typ) -> Self {
                return Self($zero, $zero, fact);
            }
        
            pub const fn one(fact: $typ) -> Self {
                return Self(fact, fact, fact);
            }
        }
    };
}

/// Implements signed contants
macro_rules! svec_const_gen {
    ($ident:ident, $typ:ty, $zero:literal, $one:literal) => {
        impl $ident {
            pub const DOWN: Self = Self::down($one);
            pub const LEFT: Self = Self::left($one);
            pub const BACK: Self = Self::back($one);
            
            pub const fn down(fact: $typ) -> Self {
                return Self($zero, -fact, $zero);
            }
        
            pub const fn left(fact: $typ) -> Self {
                return Self(-fact, $zero, $zero);
            }

            pub const fn back(fact: $typ) -> Self {
                return Self($zero, $zero, -fact);
            }
        }
    };
}

macro_rules! vec_base_impl_gen {
    ($ident:ident, $vec2:tt, $typ:ty, $zero:literal) => {
        impl $ident {
            pub fn x(self) -> $typ {
                return self.0;
            }
        
            pub fn y(self) -> $typ {
                return self.1;
            }

            pub fn z(self) -> $typ {
                return self.2;
            }

            #[cfg(feature = "vec2")]
            pub fn xy(self) -> $vec2 {
                return $vec2(self.0, self.1);
            }
        
            #[cfg(feature = "vec2")]
            pub fn yz(self) -> $vec2 {
                return $vec2(self.1, self.2);
            }
        
            #[cfg(feature = "vec2")]
            pub fn xz(self) -> $vec2 {
                return $vec2(self.0, self.2);
            }

        
            pub fn set_x(&mut self, x: $typ) {
                self.0 = x;
            }
        
            pub fn set_y(&mut self, y: $typ) {
                self.1 = y;
            }

            pub fn set_z(&mut self, z: $typ) {
                self.2 = z;
            }
        
            pub fn xvec(self) -> Self {
                return Self(self.0, $zero, $zero);
            }
        
            pub fn yvec(self) -> Self {
                return Self($zero, self.1, $zero);
            }

            pub fn zvec(self) -> Self {
                return Self($zero, $zero, self.2);
            }
        }

        impl Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({}, {}, {})", self.0, self.1, self.2)
            }
        }
    };
}

macro_rules! vec_scalar_impl_gen {
    ($ident:ident, $typ:ty, $styp:ty, $zero:literal, $min:expr, $max:expr) => {
        impl $ident {
            /// Performs the dot product between two vectors.
            pub fn dot(self, other: Self) -> $typ {
                return (self.to_simd() * other.to_simd()).reduce_sum();
            }

            pub fn sqr_magnitude(self) -> $typ {
                return self.dot(self);
            }


            /// Multiplies the vectors component-wise
            pub fn scale(self, other: Self) -> Self {
                return Self::from_simd(self.to_simd() * other.to_simd());
            }

            /// Divides the vectors component-wise
            pub fn inv_scale(self, other: Self) -> Self {
                return Self::from_simd(self.to_simd() / other.to_simd());
            }


            pub fn min(self, other: Self) -> Self {
                Self::from_simd(self.to_simd().simd_min(other.to_simd()))
            }
        
            pub fn max(self, other: Self) -> Self {
                Self::from_simd(self.to_simd().simd_max(other.to_simd()))
            }
        
            pub fn clamp(self, min: Self, max: Self) -> Self {
                self.max(min).min(max)
            }

            
            pub fn max_axis(self) -> $typ {
                return self.to_simd_ext($min).reduce_max();
            }
        
            pub fn min_axis(self) -> $typ {
                return self.to_simd_ext($max).reduce_min();
            }


            pub fn to_simd(self) -> $styp {
                <$styp>::from_array([self.0, self.1, self.2, $zero])
            }

            pub fn to_simd_ext(self, w: $typ) -> $styp {
                <$styp>::from_array([self.0, self.1, self.2, w])
            }
        
            pub fn from_simd(simd: $styp) -> Self {
                Self(simd[0], simd[1], simd[2])
            }
        }

        impl Add for $ident {
            type Output = Self;
        
            fn add(self, rhs: Self) -> Self::Output {
                return Self::from_simd(self.to_simd() + rhs.to_simd());
            }
        }
        
        impl AddAssign for $ident {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs
            }
        }
        
        impl Sub for $ident {
            type Output = Self;
        
            fn sub(self, rhs: Self) -> Self::Output {
                return Self::from_simd(self.to_simd() - rhs.to_simd());
            }
        }

        impl SubAssign for $ident {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs
            }
        }

        impl Mul<$typ> for $ident {
            type Output = Self;
        
            fn mul(self, rhs: $typ) -> Self::Output {
                return Self::from_simd(self.to_simd() * <$styp>::splat(rhs));
            }
        }

        impl MulAssign<$typ> for $ident {
            fn mul_assign(&mut self, rhs: $typ) {
                *self = *self * rhs
            }
        }
        
        impl Div<$typ> for $ident {
            type Output = Self;
        
            fn div(self, rhs: $typ) -> Self::Output {
                return Self::from_simd(self.to_simd() / <$styp>::splat(rhs));
            }
        }

        impl DivAssign<$typ> for $ident {
            fn div_assign(&mut self, rhs: $typ) {
                *self = *self * rhs
            }
        }
        
        impl From<($typ, $typ, $typ)> for $ident {
            fn from(value: ($typ, $typ, $typ)) -> Self {
                Self(value.0, value.1, value.2)
            }
        }
        
        impl From<$ident> for ($typ, $typ, $typ) {
            fn from(value: $ident) -> Self {
                (value.0, value.1, value.2)
            }
        }
    };
}

macro_rules! vec_float_impl_gen {
    ($ident:ident, $typ:ty, $styp:ty, $quat:tt) => {
        impl $ident {
            pub fn cross(self, rhs: Self) -> Self {
                return Self(self.1 * rhs.2 - rhs.1 * self.2, -self.0 * rhs.2 + rhs.0 * self.2, self.0 * rhs.1 - rhs.0 * self.1);
            }
        
            #[cfg(feature = "quaternion")]
            pub fn rotate(self, q: $quat) -> Self {
                // optimized version by https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
                let u = q.vector();
                let s = q.0;

                return (u * u.dot(self) * 2.0) + self * (s * s - u.sqr_magnitude()) + (u.cross(self) * 2.0 * s);
            }
        
        
            /// Returns the magnitude of the vector.
            pub fn magnitude(self) -> $typ {
                if self == Self::ZERO {
                    return 0.0;
                }
        
                return (self.to_simd() * self.to_simd()).reduce_sum().sqrt();
            }
        
            /// Returns the vector with a magnitude of 1.
            pub fn normalized(self) -> Self {
                let mag = self.magnitude();
                if mag == 0.0 {
                    return self;
                }
        
                return self / mag;
            }
        
            /// Returns the vector divided by the squared magnitude.
            pub fn inverted(self) -> Self {
                let sqr_mag = self.dot(self);
                if sqr_mag == 0.0 {
                    return self;
                }
        
                return self / sqr_mag;
            }
        
            /// Inverts each component
            pub fn inv_dims(self) -> Self {
                return Self::from_simd(<$styp>::splat(1.0) / self.to_simd());
            }
        
            pub fn average(slice: &[Self]) -> Self {
                let mut res = Self::ZERO;
                for v in slice {
                    res += *v;
                }
                return res / slice.len() as $typ;
            }
        
            pub fn dist_to(self, other: Self) -> $typ {
                (self - other).magnitude()
            }
        
            pub fn sqr_dist_to(self, other: Self) -> $typ {
                (self - other).sqr_magnitude()
            }
        
        
            
            pub fn min_mag(self, other: $typ) -> Self {
                let mag = self.magnitude();
                return self / mag * mag.min(other);
            }
        
            pub fn max_mag(self, other: $typ) -> Self {
                let mag = self.magnitude();
                return self / mag * mag.max(other);
            }
        
            pub fn clamp_mag(self, min: $typ, max: $typ) -> Self {
                let mag = self.magnitude();
                return self / mag * mag.clamp(min, max);
            }
        
        
            pub fn floor(self) -> Self {
                return Self::from_simd(self.to_simd().floor());
            }
        
            pub fn round(self) -> Self {
                return Self::from_simd(self.to_simd().round());
            }
        
            pub fn ceil(self) -> Self {
                return Self::from_simd(self.to_simd().ceil());
            }
        }

        impl Neg for $ident {
            type Output = Self;
        
            fn neg(self) -> Self::Output {
                return Self::from_simd(-self.to_simd());
            }
        }
    };
}

macro_rules! vec_conv_impl_gen {
    ($ident:ident, $typ:ty, $vec_a:ty, $vec_b:ty, $vec_c:ty) => {
        impl From<$vec_a> for $ident {
            fn from(value: $vec_a) -> Self {
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ)
            }
        }

        impl From<$vec_b> for $ident {
            fn from(value: $vec_b) -> Self {
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ)
            }
        }

        impl From<$vec_c> for $ident {
            fn from(value: $vec_c) -> Self {
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ)
            }
        }
    };
}

// |>    Generate Vectors    <| //
vec_type_gen!(Vector3bool, bool);
vec_type_gen!(Vector3i32, i32);
vec_type_gen!(Vector3u32, u32);
vec_type_gen!(Vector3f32, f32);
vec_type_gen!(Vector3f64, f64);


// |>    Generate Constants    <| //
uvec_const_gen!(Vector3bool, bool, false, true);

uvec_const_gen!(Vector3i32, i32, 0, 1);
svec_const_gen!(Vector3i32, i32, 0, 1);

uvec_const_gen!(Vector3u32, u32, 0, 1);

uvec_const_gen!(Vector3f32, f32, 0.0, 1.0);
svec_const_gen!(Vector3f32, f32, 0.0, 1.0);

uvec_const_gen!(Vector3f64, f64, 0.0, 1.0);
svec_const_gen!(Vector3f64, f64, 0.0, 1.0);


// |>    Generate Impls    <| //
vec_base_impl_gen!(Vector3bool, Vector2bool, bool, false);

vec_base_impl_gen!(Vector3i32, Vector2i32, i32, 0);
vec_scalar_impl_gen!(Vector3i32, i32, i32x4, 0, i32::MIN, i32::MAX);

vec_base_impl_gen!(Vector3u32, Vector2u32, u32, 0);
vec_scalar_impl_gen!(Vector3u32, u32, u32x4, 0, u32::MIN, u32::MAX);

vec_base_impl_gen!(Vector3f32, Vector2f32, f32, 0.0);
vec_scalar_impl_gen!(Vector3f32, f32, f32x4, 0.0, f32::NEG_INFINITY, f32::INFINITY);
vec_float_impl_gen!(Vector3f32, f32, f32x4, Quaternionf32);

vec_base_impl_gen!(Vector3f64, Vector2f64, f64, 0.0);
vec_scalar_impl_gen!(Vector3f64, f64, f64x4, 0.0, f64::NEG_INFINITY, f64::INFINITY);
vec_float_impl_gen!(Vector3f64, f64, f64x4, Quaternionf64);

// |>    Generate Casts    <| //
vec_conv_impl_gen!(Vector3i32, i32, Vector3u32, Vector3f32, Vector3f64);
vec_conv_impl_gen!(Vector3u32, u32, Vector3i32, Vector3f32, Vector3f64);
vec_conv_impl_gen!(Vector3f32, f32, Vector3i32, Vector3u32, Vector3f64);
vec_conv_impl_gen!(Vector3f64, f64, Vector3i32, Vector3u32, Vector3f32);

impl From<Vector3bool> for Vector3i32 {
    fn from(value: Vector3bool) -> Self {
        Self(value.0 as i32, value.1 as i32, value.2 as i32)
    }
}

impl From<Vector3bool> for Vector3u32 {
    fn from(value: Vector3bool) -> Self {
        Self(value.0 as u32, value.1 as u32, value.2 as u32)
    }
}

impl From<Vector3bool> for Vector3f32 {
    fn from(value: Vector3bool) -> Self {
        Self(value.0 as i32 as f32, value.1 as i32 as f32, value.2 as i32 as f32)
    }
}

impl From<Vector3bool> for Vector3f64 {
    fn from(value: Vector3bool) -> Self {
        Self(value.0 as i32 as f64, value.1 as i32 as f64, value.2 as i32 as f64)
    }
}


impl From<Vector3i32> for Vector3bool {
    fn from(value: Vector3i32) -> Self {
        Self(value.0 != 0, value.1 != 0, value.2 != 0)
    }
}

impl From<Vector3u32> for Vector3bool {
    fn from(value: Vector3u32) -> Self {
        Self(value.0 != 0, value.1 != 0, value.2 != 0)
    }
}

impl From<Vector3f32> for Vector3bool {
    fn from(value: Vector3f32) -> Self {
        Self(value.0 != 0.0, value.1 != 0.0, value.2 != 0.0)
    }
}

impl From<Vector3f64> for Vector3bool {
    fn from(value: Vector3f64) -> Self {
        Self(value.0 != 0.0, value.1 != 0.0, value.2 != 0.0)
    }
}
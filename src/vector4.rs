use std::{fmt::Display, ops::{Add, AddAssign, MulAssign, SubAssign, DivAssign, Div, Mul, Neg, Sub}, simd::{cmp::SimdOrd, f32x4, f64x4, i32x4, num::{SimdFloat, SimdInt, SimdUint}, u32x4, StdFloat}};

#[cfg(feature = "vec2")]
use crate::vector2::{Vector2bool, Vector2f32, Vector2f64, Vector2i32, Vector2u32};

#[cfg(not(feature = "vec2"))] type Vector2bool = ();
#[cfg(not(feature = "vec2"))] type Vector2f32 = ();
#[cfg(not(feature = "vec2"))] type Vector2f64 = ();
#[cfg(not(feature = "vec2"))] type Vector2i32 = ();
#[cfg(not(feature = "vec2"))] type Vector2u32 = ();

#[cfg(feature = "vec3")]
use crate::vector3::{Vector3bool, Vector3f32, Vector3f64, Vector3i32, Vector3u32};

#[cfg(not(feature = "vec3"))] type Vector3bool = ();
#[cfg(not(feature = "vec3"))] type Vector3f32 = ();
#[cfg(not(feature = "vec3"))] type Vector3f64 = ();
#[cfg(not(feature = "vec3"))] type Vector3i32 = ();
#[cfg(not(feature = "vec3"))] type Vector3u32 = ();

macro_rules! vec_type_gen {
    ($ident:ident, $typ:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $ident(pub $typ, pub $typ, pub $typ, pub $typ);
    };
}

/// Implements unsigned constants
macro_rules! uvec_const_gen {
    ($ident:ident, $typ:ty, $zero:literal, $one:literal) => {
        impl $ident {
            pub const ZERO: Self = Self($zero, $zero, $zero, $zero);
            pub const UP: Self = Self::up($one);
            pub const RIGHT: Self = Self::right($one);
            pub const FORW: Self = Self::forw($one);
            pub const POSW: Self = Self::posw($one);
            pub const ONE: Self = Self::one($one);
        
            pub const fn up(fact: $typ) -> Self {
                return Self($zero, fact, $zero, $zero);
            }
        
            pub const fn right(fact: $typ) -> Self {
                return Self(fact, $zero, $zero, $zero);
            }

            pub const fn forw(fact: $typ) -> Self {
                return Self($zero, $zero, fact, $zero);
            }

            pub const fn posw(fact: $typ) -> Self {
                return Self($zero, $zero, $zero, fact);
            }
        
            pub const fn one(fact: $typ) -> Self {
                return Self(fact, fact, fact, fact);
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
            pub const NEGW: Self = Self::negw($one);
            
            pub const fn down(fact: $typ) -> Self {
                return Self($zero, -fact, $zero, $zero);
            }
        
            pub const fn left(fact: $typ) -> Self {
                return Self(-fact, $zero, $zero, $zero);
            }

            pub const fn back(fact: $typ) -> Self {
                return Self($zero, $zero, -fact, $zero);
            }

            pub const fn negw(fact: $typ) -> Self {
                return Self($zero, $zero, $zero, -fact);
            }
        }
    };
}

macro_rules! vec_base_impl_gen {
    ($ident:ident, $vec3:tt, $vec2:tt, $typ:ty, $zero:literal) => {
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

            pub fn w(self) -> $typ {
                return self.3;
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
            pub fn zw(self) -> $vec2 {
                return $vec2(self.2, self.3);
            }
        
            #[cfg(feature = "vec2")]
            pub fn xz(self) -> $vec2 {
                return $vec2(self.0, self.2);
            }

            #[cfg(feature = "vec2")]
            pub fn yw(self) -> $vec2 {
                return $vec2(self.1, self.3);
            }


            #[cfg(feature = "vec3")]
            pub fn xyz(self) -> $vec3 {
                return $vec3(self.0, self.1, self.2);
            }
        
            #[cfg(feature = "vec3")]
            pub fn xyw(self) -> $vec3 {
                return $vec3(self.0, self.1, self.3);
            }

            #[cfg(feature = "vec3")]
            pub fn xzw(self) -> $vec3 {
                return $vec3(self.0, self.2, self.3);
            }
        
            #[cfg(feature = "vec3")]
            pub fn yzw(self) -> $vec3 {
                return $vec3(self.1, self.2, self.3);
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

            pub fn set_w(&mut self, w: $typ) {
                self.3 = w;
            }
        

            pub fn xvec(self) -> Self {
                return Self(self.0, $zero, $zero, $zero);
            }
        
            pub fn yvec(self) -> Self {
                return Self($zero, self.1, $zero, $zero);
            }

            pub fn zvec(self) -> Self {
                return Self($zero, $zero, self.2, $zero);
            }

            pub fn wvec(self) -> Self {
                return Self($zero, $zero, $zero, self.2);
            }
        }

        impl Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({}, {}, {}, {})", self.0, self.1, self.2, self.3)
            }
        }
    };
}

macro_rules! vec_scalar_impl_gen {
    ($ident:ident, $typ:ty, $styp:ty) => {
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
                return self.to_simd().reduce_max();
            }
        
            pub fn min_axis(self) -> $typ {
                return self.to_simd().reduce_min();
            }


            pub fn to_simd(self) -> $styp {
                <$styp>::from_array([self.0, self.1, self.2, self.3])
            }
        
            pub fn from_simd(simd: $styp) -> Self {
                Self(simd[0], simd[1], simd[2], simd[3])
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
        
        impl From<($typ, $typ, $typ, $typ)> for $ident {
            fn from(value: ($typ, $typ, $typ, $typ)) -> Self {
                Self(value.0, value.1, value.2, value.3)
            }
        }
        
        impl From<$ident> for ($typ, $typ, $typ, $typ) {
            fn from(value: $ident) -> Self {
                (value.0, value.1, value.2, value.3)
            }
        }
    };
}

macro_rules! vec_float_impl_gen {
    ($ident:ident, $typ:ty, $styp:ty) => {
        impl $ident {
            //pub fn cross(self) -> Self {
                //return Self(self.1, -self.0);
            //}
        
            /// Rotates the vector by the rotation provided in radians.<br>
            /// For rotations that need to match visuals, use `rotate_cw`
            /*pub fn rotate(self, rot: $typ) -> Self {
                return Self(self.0 * rot.cos() + self.1 * rot.sin(), self.0 * rot.sin() - self.1 * rot.cos());
            }
        
            /// Rotates the vector by the specified rotation, but in a way that matches up with visual rotations.
            pub fn rotate_cw(self, rot: $typ) -> Self {
                let rotated = self.rotate(rot);
                return Self(rotated.0, -rotated.1);
            }*/
        
        
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
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ, value.3 as $typ)
            }
        }

        impl From<$vec_b> for $ident {
            fn from(value: $vec_b) -> Self {
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ, value.3 as $typ)
            }
        }

        impl From<$vec_c> for $ident {
            fn from(value: $vec_c) -> Self {
                Self(value.0 as $typ, value.1 as $typ, value.2 as $typ, value.3 as $typ)
            }
        }
    };
}

// |>    Generate Vectors    <| //
vec_type_gen!(Vector4bool, bool);
vec_type_gen!(Vector4i32, i32);
vec_type_gen!(Vector4u32, u32);
vec_type_gen!(Vector4f32, f32);
vec_type_gen!(Vector4f64, f64);


// |>    Generate Constants    <| //
uvec_const_gen!(Vector4bool, bool, false, true);

uvec_const_gen!(Vector4i32, i32, 0, 1);
svec_const_gen!(Vector4i32, i32, 0, 1);

uvec_const_gen!(Vector4u32, u32, 0, 1);

uvec_const_gen!(Vector4f32, f32, 0.0, 1.0);
svec_const_gen!(Vector4f32, f32, 0.0, 1.0);

uvec_const_gen!(Vector4f64, f64, 0.0, 1.0);
svec_const_gen!(Vector4f64, f64, 0.0, 1.0);


// |>    Generate Impls    <| //
vec_base_impl_gen!(Vector4bool, Vector3bool, Vector2bool, bool, false);

vec_base_impl_gen!(Vector4i32, Vector3i32, Vector2i32, i32, 0);
vec_scalar_impl_gen!(Vector4i32, i32, i32x4);

vec_base_impl_gen!(Vector4u32, Vector3u32, Vector2u32, u32, 0);
vec_scalar_impl_gen!(Vector4u32, u32, u32x4);

vec_base_impl_gen!(Vector4f32, Vector3f32, Vector2f32, f32, 0.0);
vec_scalar_impl_gen!(Vector4f32, f32, f32x4);
vec_float_impl_gen!(Vector4f32, f32, f32x4);

vec_base_impl_gen!(Vector4f64, Vector3f64, Vector2f64, f64, 0.0);
vec_scalar_impl_gen!(Vector4f64, f64, f64x4);
vec_float_impl_gen!(Vector4f64, f64, f64x4);

// |>    Generate Casts    <| //
vec_conv_impl_gen!(Vector4i32, i32, Vector4u32, Vector4f32, Vector4f64);
vec_conv_impl_gen!(Vector4u32, u32, Vector4i32, Vector4f32, Vector4f64);
vec_conv_impl_gen!(Vector4f32, f32, Vector4i32, Vector4u32, Vector4f64);
vec_conv_impl_gen!(Vector4f64, f64, Vector4i32, Vector4u32, Vector4f32);

impl From<Vector4bool> for Vector4i32 {
    fn from(value: Vector4bool) -> Self {
        Self(value.0 as i32, value.1 as i32, value.2 as i32, value.3 as i32)
    }
}

impl From<Vector4bool> for Vector4u32 {
    fn from(value: Vector4bool) -> Self {
        Self(value.0 as u32, value.1 as u32, value.2 as u32, value.3 as u32)
    }
}

impl From<Vector4bool> for Vector4f32 {
    fn from(value: Vector4bool) -> Self {
        Self(value.0 as i32 as f32, value.1 as i32 as f32, value.2 as i32 as f32, value.3 as i32 as f32)
    }
}

impl From<Vector4bool> for Vector4f64 {
    fn from(value: Vector4bool) -> Self {
        Self(value.0 as i32 as f64, value.1 as i32 as f64, value.2 as i32 as f64, value.3 as i32 as f64)
    }
}


impl From<Vector4i32> for Vector4bool {
    fn from(value: Vector4i32) -> Self {
        Self(value.0 != 0, value.1 != 0, value.2 != 0, value.3 != 0)
    }
}

impl From<Vector4u32> for Vector4bool {
    fn from(value: Vector4u32) -> Self {
        Self(value.0 != 0, value.1 != 0, value.2 != 0, value.3 != 0)
    }
}

impl From<Vector4f32> for Vector4bool {
    fn from(value: Vector4f32) -> Self {
        Self(value.0 != 0.0, value.1 != 0.0, value.2 != 0.0, value.3 != 0.0)
    }
}

impl From<Vector4f64> for Vector4bool {
    fn from(value: Vector4f64) -> Self {
        Self(value.0 != 0.0, value.1 != 0.0, value.2 != 0.0, value.3 != 0.0)
    }
}
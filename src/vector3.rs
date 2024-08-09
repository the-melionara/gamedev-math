#[macro_export]
macro_rules! gen_vec3 {
    ($ident:ident, $vec2:tt, $typ:ty, $zero:literal) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $ident(pub $typ, pub $typ, pub $typ);

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

            pub fn xy(self) -> $vec2 {
                return $vec2(self.0, self.1);
            }
        
            pub fn yz(self) -> $vec2 {
                return $vec2(self.1, self.2);
            }
        
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

        impl std::fmt::Display for $ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "({}, {}, {})", self.0, self.1, self.2)
            }
        }
    };
}

#[macro_export]
macro_rules! unsigned_vec3_impl {
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

#[macro_export]
macro_rules! signed_vec3_impl {
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

#[macro_export]
macro_rules! scalar_vec3_impl {
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

        impl std::ops::Add for $ident {
            type Output = Self;
        
            fn add(self, rhs: Self) -> Self::Output {
                return Self::from_simd(self.to_simd() + rhs.to_simd());
            }
        }
        
        impl std::ops::AddAssign for $ident {
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs
            }
        }
        
        impl std::ops::Sub for $ident {
            type Output = Self;
        
            fn sub(self, rhs: Self) -> Self::Output {
                return Self::from_simd(self.to_simd() - rhs.to_simd());
            }
        }

        impl std::ops::SubAssign for $ident {
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs
            }
        }

        impl std::ops::Mul<$typ> for $ident {
            type Output = Self;
        
            fn mul(self, rhs: $typ) -> Self::Output {
                return Self::from_simd(self.to_simd() * <$styp>::splat(rhs));
            }
        }

        impl std::ops::MulAssign<$typ> for $ident {
            fn mul_assign(&mut self, rhs: $typ) {
                *self = *self * rhs
            }
        }
        
        impl std::ops::Div<$typ> for $ident {
            type Output = Self;
        
            fn div(self, rhs: $typ) -> Self::Output {
                return Self::from_simd(self.to_simd() / <$styp>::splat(rhs));
            }
        }

        impl std::ops::DivAssign<$typ> for $ident {
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

#[macro_export]
macro_rules! float_vec3_impl {
    ($ident:ident, $typ:ty, $styp:ty) => {
        impl $ident {
            pub fn cross(self, rhs: Self) -> Self {
                return Self(self.1 * rhs.2 - rhs.1 * self.2, -self.0 * rhs.2 + rhs.0 * self.2, self.0 * rhs.1 - rhs.0 * self.1);
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

        impl std::ops::Neg for $ident {
            type Output = Self;
        
            fn neg(self) -> Self::Output {
                return Self::from_simd(-self.to_simd());
            }
        }
    };
}

#[macro_export]
macro_rules! rot_vec3_impl {
    ($ident:ident, $quat:ident) => {
        impl $ident {
            pub fn rotate(self, q: $quat) -> Self {
                // optimized version by https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
                let u = q.vector();
                let s = q.0;
    
                return (u * u.dot(self) * 2.0) + self * (s * s - u.sqr_magnitude()) + (u.cross(self) * 2.0 * s);
            }
        }
    };
}

#[macro_export]
macro_rules! cast_vec3_impl {
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

#[macro_export]
macro_rules! updim_vec3_impl {
    (2, $ident:ident, $typ:ty, $vec:ty) => {
        impl $ident {
            pub fn from_xy(xy: $vec, z: $typ) -> Self {
                Self(xy.0, xy.1, z)
            }

            pub fn from_xz(xz: $vec, y: $typ) -> Self {
                Self(xz.0, y, xz.1)
            }

            pub fn from_yz(x: $typ, yz: $vec) -> Self {
                Self(x, yz.0, yz.1)
            }
        }
    };
}
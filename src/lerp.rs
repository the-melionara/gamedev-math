use std::simd::{LaneCount, Simd, SupportedLaneCount};

#[cfg(feature = "vec2")] use crate::vector2::{Vector2f32, Vector2f64};
#[cfg(feature = "vec3")] use crate::vector3::{Vector3f32, Vector3f64};
#[cfg(feature = "vec4")] use crate::vector4::{Vector4f32, Vector4f64};


pub trait Lerp<T> {
    /// Linearly interpolates between `self` and `other` using the factor `t`.
    fn lerp(self, other: Self, t: T) -> Self;

    /// Linearly interpolates between `self` and `other` using the factor `t`. The interpolation is done component-wise.
    fn lerp_c(self, other: Self, t: Self) -> Self;

    /// Inverse of the lerp function. Returns the factor needed to get `res` from interpolation between `self` and `other`
    fn inv_lerp(self, other: Self, res: Self) -> Self;
}

impl Lerp<f32> for f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.lerp(other, t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self) / (other - self)
    }
}


impl Lerp<f64> for f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.lerp(other, t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self) / (other - self)
    }
}

impl<const N: usize> Lerp<f32> for Simd<f32, N> where LaneCount<N>: SupportedLaneCount {
    fn lerp(self, other: Self, t: f32) -> Self {
        self * Self::splat(1.0 - t) + other * Self::splat(t)
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self * (Self::splat(1.0) - t) + other * t
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self) / (other - self)
    }
}

impl<const N: usize> Lerp<f64> for Simd<f64, N> where LaneCount<N>: SupportedLaneCount {
    fn lerp(self, other: Self, t: f64) -> Self {
        self * Self::splat(1.0 - t) + other * Self::splat(t)
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self * (Self::splat(1.0) - t) + other * t
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self) / (other - self)
    }
}


#[cfg(feature = "vec2")]
impl Lerp<f32> for Vector2f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}

#[cfg(feature = "vec2")]
impl Lerp<f64> for Vector2f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}


#[cfg(feature = "vec3")]
impl Lerp<f32> for Vector3f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}

#[cfg(feature = "vec3")]
impl Lerp<f64> for Vector3f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}


#[cfg(feature = "vec4")]
impl Lerp<f32> for Vector4f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}

#[cfg(feature = "vec4")]
impl Lerp<f64> for Vector4f64 {
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    fn lerp_c(self, other: Self, t: Self) -> Self {
        self.scale(Self::ONE - t) + other.scale(t)
    }

    fn inv_lerp(self, other: Self, res: Self) -> Self {
        (res - self).inv_scale(other - self)
    }
}
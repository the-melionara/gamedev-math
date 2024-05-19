#![feature(portable_simd)]
#![feature(const_fn_floating_point_arithmetic)]

// Vectors
pub mod vector2;
pub mod vector3;
pub mod vector4;

pub mod quaternion;

// Rects
pub mod rect;

// Matrices
pub mod mat2x2;
pub mod mat3x3;
pub mod mat4x4;

// Utils
pub mod lerp;
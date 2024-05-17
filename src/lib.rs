#![feature(portable_simd)]
#![feature(const_fn_floating_point_arithmetic)]

// Vectors
#[cfg(feature = "vec2")] pub mod vector2;
#[cfg(feature = "vec3")] pub mod vector3;
#[cfg(feature = "vec4")] pub mod vector4;

// Rects
#[cfg(feature = "rect")] pub mod rect;

// Matrices
pub mod mat3x3;
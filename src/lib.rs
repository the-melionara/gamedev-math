#![feature(portable_simd)]
#![feature(const_fn_floating_point_arithmetic)]

// Vectors
#[cfg(feature = "vec2")] pub mod vector2;
#[cfg(feature = "vec3")] pub mod vector3;
#[cfg(feature = "vec4")] pub mod vector4;

#[cfg(feature = "quaternion")] pub mod quaternion;

// Rects
#[cfg(feature = "rect")] pub mod rect;

// Matrices
#[cfg(feature = "mat2x2")] pub mod mat2x2;
#[cfg(feature = "mat3x3")] pub mod mat3x3;
#[cfg(feature = "mat4x4")] pub mod mat4x4;

// Utils
#[cfg(feature = "lerp")] pub mod lerp;
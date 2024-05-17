use crate::vector2::{Vector2f32, Vector2f64, Vector2i32, Vector2u32};

macro_rules! rect_type_gen {
    ($ident:ident, $vec:ty) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $ident {
            pub start: $vec,
            pub end: $vec,
        }
    };
}

macro_rules! rect_base_impl_gen {
    ($ident:ident, $vec:tt, $typ:ty) => {
        impl $ident {
            pub const IDENT: Self = Self { start: <$vec>::ZERO, end: <$vec>::ONE };
    
            pub fn up(&self) -> $typ {
                return self.end.1;
            }
    
            pub fn down(&self) -> $typ {
                return self.start.1;
            }
    
            pub fn right(&self) -> $typ {
                return self.end.0;
            }
    
            pub fn left(&self) -> $typ {
                return self.start.0;
            }
    
            pub fn lu(&self) -> $vec {
                return $vec(self.left(), self.up());
            }
    
            pub fn ru(&self) -> $vec {
                return $vec(self.right(), self.up());
            }
    
            pub fn ld(&self) -> $vec {
                return $vec(self.left(), self.down());
            }
    
            pub fn rd(&self) -> $vec {
                return $vec(self.right(), self.down());
            }

            pub fn size(&self) -> $vec {
                return self.end - self.start;
            }

            pub fn contains(&self, pos: $vec) -> bool {
                return
                    pos.0 >= self.left() && pos.0 <= self.right() &&
                    pos.1 >= self.down() && pos.1 <= self.up();
            }
        
            pub fn expand(self, fact: $typ) -> Self {
                return Self { start: self.start - $vec::one(fact), end: self.end + $vec::one(fact) };
            }
        }
    };
}

// |>    Generate Rects    <| //
rect_type_gen!(Recti32, Vector2i32);
rect_type_gen!(Rectu32, Vector2u32);
rect_type_gen!(Rectf32, Vector2f32);
rect_type_gen!(Rectf64, Vector2f64);

// |>    Generate Impls    <| //
rect_base_impl_gen!(Recti32, Vector2i32, i32);
rect_base_impl_gen!(Rectu32, Vector2u32, u32);
rect_base_impl_gen!(Rectf32, Vector2f32, f32);
rect_base_impl_gen!(Rectf64, Vector2f64, f64);

// |>    Manual Impls    <| //
impl Recti32 {
    pub fn center(&self) -> Vector2i32 {
        return (self.start + self.end) / 2;
    }
}

impl Rectu32 {
    pub fn center(&self) -> Vector2u32 {
        return (self.start + self.end) / 2;
    }
}

impl Rectf32 {
    pub fn center(&self) -> Vector2f32 {
        return (self.start + self.end) / 2.0;
    }

    pub fn sample(&self, pos: Vector2f32) -> Vector2f32 {
        return self.start.scale(Vector2f32::ONE - pos) + self.end.scale(pos);
    }
}

impl Rectf64 {
    pub fn center(&self) -> Vector2f64 {
        return (self.start + self.end) / 2.0;
    }

    pub fn sample(&self, pos: Vector2f64) -> Vector2f64 {
        return self.start.scale(Vector2f64::ONE - pos) + self.end.scale(pos);
    }
}
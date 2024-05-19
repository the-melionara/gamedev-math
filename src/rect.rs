#[macro_export]
macro_rules! gen_rect {
    ($ident:ident, $vec:tt, $typ:ty, $two:literal) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $ident {
            pub start: $vec,
            pub end: $vec,
        }
    
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

            pub fn center(&self) -> $vec {
                return (self.start + self.end) / $two;
            }
        }
    };
}

#[macro_export]
macro_rules! float_rect_impl {
    ($ident:ident, $vec:ident) => {
        impl $ident {
            pub fn sample(&self, pos: $vec) -> $vec {
                return self.start.scale($vec::ONE - pos) + self.end.scale(pos);
            }
        }
    };
}
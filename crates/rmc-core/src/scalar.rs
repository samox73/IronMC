use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Sub};

use num_complex::Complex64;

pub trait Scalar:
    Copy
    + Clone
    + Debug
    + Default
    + PartialEq
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn conj(self) -> Self;
    fn abs_sqr(self) -> f64;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn conj(self) -> Self {
        self
    }

    fn abs_sqr(self) -> f64 {
        self * self
    }
}

impl Scalar for Complex64 {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn abs_sqr(self) -> f64 {
        self.norm_sqr()
    }
}

pub trait SampleType: Clone + Debug + 'static {
    type Value: Scalar;

    fn size(&self) -> usize;
}

impl SampleType for f64 {
    type Value = f64;

    fn size(&self) -> usize {
        1
    }
}

impl SampleType for Complex64 {
    type Value = Complex64;

    fn size(&self) -> usize {
        1
    }
}

use num_complex::Complex64;
use rmc_core::{SampleType, Scalar};

fn accepts_scalar<T: Scalar>(value: T) -> T {
    value.conj()
}

fn accepts_sample<T: SampleType>(sample: &T) -> usize {
    sample.size()
}

#[test]
fn f64_is_scalar_and_scalar_sample() {
    assert_eq!(f64::zero(), 0.0);
    assert_eq!(f64::one(), 1.0);
    assert_eq!(accepts_scalar(2.5_f64), 2.5);
    assert_eq!(2.5_f64.abs_sqr(), 6.25);
    assert_eq!(accepts_sample(&2.5_f64), 1);
}

#[test]
fn complex64_is_scalar_and_scalar_sample() {
    let value = Complex64::new(1.5, -2.0);

    assert_eq!(Complex64::zero(), Complex64::new(0.0, 0.0));
    assert_eq!(Complex64::one(), Complex64::new(1.0, 0.0));
    assert_eq!(accepts_scalar(value), Complex64::new(1.5, 2.0));
    assert_eq!(value.abs_sqr(), 6.25);
    assert_eq!(accepts_sample(&value), 1);
}

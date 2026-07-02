use std::f64::consts::PI;

use nalgebra::Vector3;
use rmc_core::random::safe_exponential_sample;
use rmc_frohlich::diagram::{phi_from_cartesian, spherical_to_cartesian, theta_from_cartesian};

fn cpp_draw_from_exp(begin: f64, end: f64, lambda: f64, random_0_1: f64) -> f64 {
    begin - (1.0 - random_0_1 * (1.0 - (-(lambda * (end - begin))).exp())).ln() / lambda
}

#[test]
fn safe_exponential_sample_matches_cpp_draw_for_positive_lambda() {
    for (lambda, a, b, r) in [
        (0.2, 0.0, 3.0, 0.1),
        (1.7, -2.0, 0.5, 0.6),
        (8.0, 1.0, 2.0, 0.95),
    ] {
        let rust = safe_exponential_sample(r, lambda, a, b);
        let cpp = cpp_draw_from_exp(a, b, lambda, r);
        assert!((rust - cpp).abs() < 1.0e-12);
    }
}

#[test]
fn spherical_cartesian_round_trip() {
    let r = 1.7;
    let theta = 0.37 * PI;
    let phi = -0.42 * PI;
    let v = spherical_to_cartesian(r, theta, phi);
    assert!((v.norm() - r).abs() < 1.0e-12);
    assert!((theta_from_cartesian(&v) - theta).abs() < 1.0e-12);
    assert!((phi_from_cartesian(&v) - phi).abs() < 1.0e-12);

    let axis = Vector3::new(0.0, 0.0, r);
    assert!(theta_from_cartesian(&axis).abs() < 1.0e-12);
}

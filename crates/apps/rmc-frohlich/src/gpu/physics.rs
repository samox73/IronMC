//! GPU-adjacent scalar physics leaves.
//!
//! CubeCL kernels use scalar arguments instead of nalgebra types, so these mirrors stay next to
//! the CPU leaves and are covered by parity tests.

use crate::physics::{self, Vec3};

pub fn bare_dispersion(p: Vec3) -> f64 {
    physics::bare_dispersion(p)
}

pub fn dispersion(p: Vec3, mu: f64) -> f64 {
    physics::dispersion(p, mu)
}

pub fn bare_propagator(p: Vec3, mu: f64, tau: f64) -> f64 {
    physics::bare_propagator(p, mu, tau)
}

pub fn phonon_lambda(q: Vec3) -> f64 {
    physics::phonon_lambda(q)
}

pub fn segment_exponent(p: Vec3, mu: f64, dtau: f64) -> f64 {
    physics::segment_exponent(p, mu, dtau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_physics_mirrors_cpu_leaves() {
        let p = [0.25, -0.5, 1.25];
        let q = [0.75, 0.0, -0.25];
        assert_eq!(bare_dispersion(p), physics::bare_dispersion(p));
        assert_eq!(dispersion(p, -1.1), physics::dispersion(p, -1.1));
        assert_eq!(
            bare_propagator(p, -1.1, 0.75),
            physics::bare_propagator(p, -1.1, 0.75)
        );
        assert_eq!(phonon_lambda(q), physics::phonon_lambda(q));
        assert_eq!(
            segment_exponent(p, -1.1, 0.5),
            physics::segment_exponent(p, -1.1, 0.5)
        );
    }
}

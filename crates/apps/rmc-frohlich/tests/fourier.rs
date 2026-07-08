use std::f64::consts::PI;

use rmc_frohlich::diagram::Diagram;
use rmc_frohlich::fourier::{analyze_self_energy, get_omega, FourierConfig};
use rmc_grids::{Grid1d, LinearGrid};

#[test]
fn omega_index_folds_like_fft_convention() {
    let dt = 0.5;
    let n = 8;
    let unit = 2.0 * PI / (dt * n as f64);
    assert!((get_omega(0, dt, n) - 0.0).abs() < 1.0e-12);
    assert!((get_omega(3, dt, n) - 3.0 * unit).abs() < 1.0e-12);
    assert!((get_omega(4, dt, n) + 4.0 * unit).abs() < 1.0e-12);
    assert!((get_omega(7, dt, n) + unit).abs() < 1.0e-12);
}

#[test]
fn fourier_analysis_returns_finite_series() {
    let bins = 32;
    let grid = LinearGrid::new(0.0, 8.0, bins + 1).unwrap();
    let cfg = FourierConfig {
        alpha: 1.0,
        mu: -1.1,
        momentum: 0.0,
        beta: 20.0,
        num_frequencies: 256,
    };
    let self_energy = grid
        .bin_centers()
        .map(|tau| {
            cfg.alpha * Diagram::MASS.sqrt() / (PI.sqrt() * tau.sqrt())
                * (-(Diagram::OMEGA - cfg.mu) * tau).exp()
        })
        .collect::<Vec<_>>();

    let output = analyze_self_energy(&cfg, &grid, &self_energy).unwrap();
    assert_eq!(output.tau.len(), bins);
    assert_eq!(output.g0_t.len(), bins);
    assert_eq!(output.s_higher_t.len(), bins);
    assert_eq!(output.g_w.real.len(), bins);
    assert_eq!(output.g_w.imag.len(), bins);
    assert_eq!(output.g_t.len(), bins);
    assert!(output.g_t.iter().all(|value| value.is_finite()));
    assert!(output.g_w.real.iter().all(|value| value.is_finite()));
    assert!(output.g_w.imag.iter().all(|value| value.is_finite()));
    assert!(output.s_higher_t.iter().all(|value| value.abs() < 1.0e-10));
}

use std::f64::consts::PI;

use num_complex::Complex;
use rmc_grids::{Grid1d, LinearGrid};
use rmc_numeric::CubicSplineInterpolation;
use rustfft::FftPlanner;

use crate::diagram::Diagram;
use crate::measurement::PolaronStats;

pub type FourierResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FourierConfig {
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub beta: f64,
    pub num_frequencies: usize,
}

impl FourierConfig {
    pub fn from_stats(stats: &PolaronStats, beta: f64) -> Self {
        Self {
            alpha: stats.alpha,
            mu: stats.mu,
            momentum: stats.momentum,
            beta,
            num_frequencies: 1 << 16,
        }
    }

    fn dispersion(&self) -> f64 {
        self.momentum * self.momentum / (2.0 * Diagram::MASS) - self.mu
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComplexSeries {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FourierOutput {
    pub tau: Vec<f64>,
    pub g0_t: Vec<f64>,
    pub s_higher_t: Vec<f64>,
    pub g_w: ComplexSeries,
    pub g_t: Vec<f64>,
}

/// Fourier-transforms the measured self-energy Σ(τ) into the Matsubara-frequency Green's
/// function G(ω) and back into G(τ), via the FFT convolution trick in [`analyze_self_energy`].
pub fn analyze_stats(stats: &PolaronStats, beta: f64) -> FourierResult<FourierOutput> {
    let grid = stats.grid.grid();
    let self_energy = stats.get_exact();
    analyze_self_energy(&FourierConfig::from_stats(stats, beta), &grid, &self_energy)
}

pub fn analyze_self_energy(
    cfg: &FourierConfig,
    grid: &LinearGrid,
    self_energy: &[f64],
) -> FourierResult<FourierOutput> {
    if self_energy.len() < 2 {
        return Err("fourier analysis needs at least two self-energy points".into());
    }
    if cfg.num_frequencies < 2 {
        return Err("num_frequencies must be at least two".into());
    }

    let min_time = grid.bin_center(0).ok_or("missing first bin center")?;
    let max_time = grid
        .bin_center(grid.bin_count() - 1)
        .ok_or("missing last bin center")?;
    let coarse_grid = LinearGrid::new(min_time, max_time, self_energy.len())?;
    let dense_grid = LinearGrid::new(min_time, cfg.beta, cfg.num_frequencies)?;
    let delta_t = (cfg.beta - min_time) / (cfg.num_frequencies - 1) as f64;

    let taus = coarse_grid.points().collect::<Vec<_>>();
    let dense_taus = dense_grid.points().collect::<Vec<_>>();

    let s1_t = initialize_s1_t(cfg, &taus);
    let s_prime_t = self_energy
        .iter()
        .zip(&s1_t)
        .map(|(s, s1)| s - s1)
        .collect::<Vec<_>>();
    let s_prime_spline = CubicSplineInterpolation::natural(coarse_grid, s_prime_t.clone())?;
    let s_prime_t_dense = dense_taus
        .iter()
        .map(|&tau| {
            if tau <= max_time {
                s_prime_spline.evaluate(tau).unwrap_or(0.0)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let s1_t_dense = initialize_s1_t(cfg, &dense_taus);
    let s_t_dense = s_prime_t_dense
        .iter()
        .zip(&s1_t_dense)
        .map(|(prime, s1)| prime + s1)
        .collect::<Vec<_>>();

    let s_w_dense = fft_self_energy(cfg, delta_t, &s_t_dense, &s1_t_dense);
    let g0_w_dense = initialize_g0_w(cfg, delta_t);
    let g_w_dense = g0_w_dense
        .iter()
        .zip(&s_w_dense)
        .map(|(g0, s)| Complex::new(1.0, 0.0) / (Complex::new(1.0, 0.0) / *g0 - *s))
        .collect::<Vec<_>>();

    let g0_t_dense = initialize_g0_t(cfg, &dense_taus);
    let g_t_dense = ifft_green(cfg, &g_w_dense, &g0_w_dense, &g0_t_dense);

    let g_t_spline = CubicSplineInterpolation::natural(dense_grid, g_t_dense)?;
    let g_w_real_spline = CubicSplineInterpolation::natural(
        dense_grid,
        g_w_dense.iter().map(|value| value.re).collect::<Vec<_>>(),
    )?;
    let g_w_imag_spline = CubicSplineInterpolation::natural(
        dense_grid,
        g_w_dense.iter().map(|value| value.im).collect::<Vec<_>>(),
    )?;

    let tau = coarse_grid.points().collect::<Vec<_>>();
    let g_t = tau
        .iter()
        .map(|&t| g_t_spline.evaluate(t))
        .collect::<Result<Vec<_>, _>>()?;
    let g_w_real = tau
        .iter()
        .map(|&t| g_w_real_spline.evaluate(t))
        .collect::<Result<Vec<_>, _>>()?;
    let g_w_imag = tau
        .iter()
        .map(|&t| g_w_imag_spline.evaluate(t))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(FourierOutput {
        tau,
        g0_t: initialize_g0_t(cfg, &taus),
        s_higher_t: s_prime_t,
        g_w: ComplexSeries {
            real: g_w_real,
            imag: g_w_imag,
        },
        g_t,
    })
}

pub fn get_omega(index: usize, delta_t: f64, num_frequencies: usize) -> f64 {
    let folded = if index < num_frequencies.div_ceil(2) {
        index as isize
    } else {
        -((num_frequencies - index) as isize)
    };
    2.0 * PI * folded as f64 / (delta_t * num_frequencies as f64)
}

fn initialize_g0_w(cfg: &FourierConfig, delta_t: f64) -> Vec<Complex<f64>> {
    (0..cfg.num_frequencies)
        .map(|j| {
            Complex::new(1.0, 0.0)
                / Complex::new(cfg.dispersion(), get_omega(j, delta_t, cfg.num_frequencies))
        })
        .collect()
}

fn initialize_s1_w(cfg: &FourierConfig, delta_t: f64) -> Vec<Complex<f64>> {
    (0..cfg.num_frequencies)
        .map(|j| {
            Complex::new(cfg.alpha, 0.0)
                / Complex::new(1.0 - cfg.mu, get_omega(j, delta_t, cfg.num_frequencies)).sqrt()
        })
        .collect()
}

fn initialize_s1_t(cfg: &FourierConfig, taus: &[f64]) -> Vec<f64> {
    taus.iter()
        .map(|&tau| {
            cfg.alpha * Diagram::MASS.sqrt() / (PI.sqrt() * tau.sqrt())
                * (-(Diagram::OMEGA - cfg.mu) * tau).exp()
        })
        .collect()
}

fn initialize_g0_t(cfg: &FourierConfig, taus: &[f64]) -> Vec<f64> {
    taus.iter()
        .map(|&tau| (-(cfg.dispersion() * tau)).exp())
        .collect()
}

fn fft_self_energy(
    cfg: &FourierConfig,
    delta_t: f64,
    s_t: &[f64],
    s1_t: &[f64],
) -> Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cfg.num_frequencies);
    let mut buffer = s_t
        .iter()
        .zip(s1_t)
        .map(|(s, s1)| Complex::new((s - s1) * delta_t, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut buffer);
    let s1_w = initialize_s1_w(cfg, delta_t);
    buffer
        .into_iter()
        .zip(s1_w)
        .map(|(fft_value, s1)| fft_value + s1)
        .collect()
}

fn ifft_green(
    cfg: &FourierConfig,
    g_w: &[Complex<f64>],
    g0_w: &[Complex<f64>],
    g0_t: &[f64],
) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(cfg.num_frequencies);
    let mut buffer = g_w
        .iter()
        .zip(g0_w)
        .map(|(g, g0)| (*g - *g0) / cfg.beta)
        .collect::<Vec<_>>();
    fft.process(&mut buffer);
    buffer
        .into_iter()
        .zip(g0_t)
        .map(|(value, g0)| value.re + g0)
        .collect()
}

use rand::Rng;
use rand_distr::{Distribution, Normal};
use rmc_core::dispatch_update;
use rmc_core::mc::{WeightedUpdate, WeightedUpdateSet};
use rmc_core::Result;
use rmc_stats::BinnedScalar;

pub const DEFAULT_BATCH_SIZE: usize = 1_000;

#[derive(Clone, Debug)]
pub struct MinimalState {
    pub x: f64,
}

impl Default for MinimalState {
    fn default() -> Self {
        Self { x: 0.0 }
    }
}

impl MinimalState {
    pub fn weight_ratio(x_new: f64, x_old: f64) -> f64 {
        (-(x_new * x_new - x_old * x_old) / 2.0).exp()
    }
}

dispatch_update! {
    #[derive(Clone, Debug)]
    pub enum MinimalUpdate<MinimalState> {
        Gaussian(GaussianShift),
        Uniform(UniformShift),
        Mirror(Mirror),
    }
    ; reject
}

#[derive(Clone, Debug)]
pub struct GaussianShift {
    sigma: f64,
    normal: Normal<f64>,
    x_prime: f64,
}

impl GaussianShift {
    pub fn new(sigma: f64) -> Self {
        Self {
            sigma,
            normal: Normal::new(0.0, 1.0).expect("unit normal parameters are valid"),
            x_prime: 0.0,
        }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, state: &MinimalState, rng: &mut R) -> f64 {
        self.x_prime = state.x + self.sigma * self.normal.sample(rng);
        MinimalState::weight_ratio(self.x_prime, state.x)
    }

    pub fn accept(&mut self, state: &mut MinimalState) {
        state.x = self.x_prime;
    }

    pub fn reject(&mut self, _state: &mut MinimalState) {}
}

#[derive(Clone, Debug)]
pub struct UniformShift {
    a: f64,
    x_prime: f64,
}

impl UniformShift {
    pub fn new(a: f64) -> Self {
        Self { a, x_prime: 0.0 }
    }

    pub fn attempt<R: Rng + ?Sized>(&mut self, state: &MinimalState, rng: &mut R) -> f64 {
        self.x_prime = state.x + (2.0 * rng.gen::<f64>() - 1.0) * self.a;
        MinimalState::weight_ratio(self.x_prime, state.x)
    }

    pub fn accept(&mut self, state: &mut MinimalState) {
        state.x = self.x_prime;
    }

    pub fn reject(&mut self, _state: &mut MinimalState) {}
}

#[derive(Clone, Debug, Default)]
pub struct Mirror {
    x_prime: f64,
}

impl Mirror {
    pub fn attempt<R: Rng + ?Sized>(&mut self, state: &MinimalState, _rng: &mut R) -> f64 {
        self.x_prime = -state.x;
        1.0
    }

    pub fn accept(&mut self, state: &mut MinimalState) {
        state.x = self.x_prime;
    }

    pub fn reject(&mut self, _state: &mut MinimalState) {}
}

pub fn minimal_measurement(
    block_size: usize,
) -> Result<(
    BinnedScalar<impl Fn(&MinimalState) -> f64>,
    BinnedScalar<impl Fn(&MinimalState) -> f64>,
)> {
    Ok((
        BinnedScalar::new(block_size, |s: &MinimalState| s.x)?,
        BinnedScalar::new(block_size, |s: &MinimalState| s.x * s.x)?,
    ))
}

pub fn build_full() -> Result<WeightedUpdateSet<MinimalUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(MinimalUpdate::Gaussian(GaussianShift::new(1.0)), 1.0),
        WeightedUpdate::new(MinimalUpdate::Uniform(UniformShift::new(2.0)), 1.0),
        WeightedUpdate::new(MinimalUpdate::Mirror(Mirror::default()), 1.0),
    ])
}

pub fn build_bare() -> Result<WeightedUpdateSet<MinimalUpdate>> {
    WeightedUpdateSet::new(vec![WeightedUpdate::new(
        MinimalUpdate::Gaussian(GaussianShift::new(1.0)),
        1.0,
    )])
}

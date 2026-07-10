use std::fs;
use std::path::Path;

use rmc_core::mc::SimulationParams;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RunConfig {
    /// Fröhlich coupling constant.
    pub alpha: f64,
    /// Chemical potential.
    pub mu: f64,
    /// External momentum of the polaron.
    pub momentum: f64,
    /// Imaginary-time cutoff for the diagram.
    pub max_tau: f64,
    pub start_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
    #[serde(default = "default_max_order_gpu")]
    pub max_order_gpu: usize,
    /// Number of bins in the self-energy histogram over `[0, max_tau]`.
    pub num_bins: usize,
    /// Number of jackknife batches.
    pub n_batches: usize,
    /// Initial guess for the ground-state energy, used to reweight `AddPhonon`/`RemovePhonon`.
    pub energy_estimate: f64,
    /// Steps between successive re-estimates of `energy_estimate`.
    pub initial_self_consistent_period: usize,
    /// Growth factor applied to the self-consistent period after each reweighting.
    pub period_multiplier: f64,
    /// Inverse temperature used for the Fourier analysis of the self-energy.
    pub fft_beta: f64,
    pub seed: u64,
    pub chains: u64,
    pub max_steps: u64,
    pub warmup_steps: u64,
    pub steps_per_cycle: u64,
    pub cycles_per_check: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            mu: -1.1,
            momentum: 0.0,
            max_tau: 30.0,
            start_tau: 1.0,
            min_order: 0,
            max_order: 10_000,
            max_order_gpu: default_max_order_gpu(),
            num_bins: 2_000,
            n_batches: 256,
            energy_estimate: -1.0168,
            initial_self_consistent_period: 1_000,
            period_multiplier: 1.5,
            fft_beta: 100.0,
            seed: 8_267_165_747_609_980_501,
            chains: 1,
            max_steps: 100_000,
            warmup_steps: 0,
            steps_per_cycle: 5,
            cycles_per_check: 1_000_000,
        }
    }
}

pub fn default_max_order_gpu() -> usize {
    256
}

impl RunConfig {
    pub fn simulation_params(&self) -> SimulationParams {
        SimulationParams {
            max_steps: self.max_steps,
            steps_per_cycle: self.steps_per_cycle,
            cycles_per_check: self.cycles_per_check,
        }
    }

    pub fn warmup_params(&self) -> SimulationParams {
        SimulationParams {
            max_steps: self.warmup_steps,
            steps_per_cycle: self.steps_per_cycle,
            cycles_per_check: self.cycles_per_check,
        }
    }

    pub fn from_json_str(json: &str) -> serde_json::Result<Self> {
        serde_json::from_str(json)
    }

    pub fn to_json_string(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let json = fs::read_to_string(path)
            .map_err(|e| format!("failed to read config file '{}': {e}", path.display()))?;
        Ok(Self::from_json_str(&json)
            .map_err(|e| format!("failed to parse config file '{}': {e}", path.display()))?)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(path, self.to_json_string()?)?;
        Ok(())
    }
}

use std::fs;
use std::path::Path;

use rmc_core::mc::SimulationParams;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RunConfig {
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub max_tau: f64,
    pub start_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
    pub num_bins: usize,
    pub n_batches: usize,
    pub energy_estimate: f64,
    pub initial_self_consistent_period: usize,
    pub period_multiplier: f64,
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
        Ok(Self::from_json_str(&fs::read_to_string(path)?)?)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        fs::write(path, self.to_json_string()?)?;
        Ok(())
    }
}

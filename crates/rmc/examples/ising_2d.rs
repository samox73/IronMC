//! Minimal 2D Ising-model simulation using the current `rmc-core` prototype API.
//!
//! The example models a square `16 x 16` lattice with periodic boundary conditions. Each lattice
//! site stores a spin `s_i = +/-1`, and the energy is the standard nearest-neighbor Ising energy
//! with coupling `J = 1`. A Monte Carlo step proposes flipping one randomly selected spin. The
//! update computes the local energy difference for that flip,
//!
//! `Delta E = 2 s_i sum_neighbors s_j`,
//!
//! and accepts it with the Metropolis probability `min(1, exp(-beta Delta E))`. The spin index and
//! acceptance draw both come from the framework-managed per-chain RNG, so runs are reproducible for
//! a fixed `SeedSource`.
//!
//! The lattice is owned as the chain state. The update receives `&mut IsingLattice` when it
//! proposes and accepts a flip, while the measurement receives `&IsingLattice` once per simulation
//! cycle. The measurement records the magnetization per spin and absolute magnetization per spin,
//! then returns a typed `IsingSummary` from `Measurement::finish`. `IsingSummary` implements
//! `Merge`, which lets `run_parallel` combine independent chains with the same
//! reduction mechanism used by the rest of the prototype.
//!
//! The example runs the same model in two modes:
//!
//! 1. A single chain via `run_typed`.
//! 2. Eight independent chains via `run_parallel`.
//!
//! This is still a prototype example, but unlike the first version it does not use `Arc<Mutex<_>>`
//! in the hot path. Each rayon worker owns a complete independent chain state.

use rmc::mc::{
    run_parallel, run_typed, Measurement, MetropolisKernel, ParallelConfig, SimulationParams,
    SingleUpdateSet, Update,
};
use rmc::random::{uniform_index, ChainId, Rng, SeedSource};
use rmc::Merge;

#[derive(Clone)]
struct IsingLattice {
    width: usize,
    height: usize,
    spins: Vec<i8>,
}

impl IsingLattice {
    fn ordered(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            spins: vec![1; width * height],
        }
    }

    fn len(&self) -> usize {
        self.spins.len()
    }

    fn magnetization(&self) -> i64 {
        self.spins.iter().map(|spin| i64::from(*spin)).sum()
    }

    fn delta_energy_for_flip(&self, idx: usize) -> i32 {
        let x = idx % self.width;
        let y = idx / self.width;
        let left = self.spin_at(wrap_sub(x, self.width), y);
        let right = self.spin_at((x + 1) % self.width, y);
        let up = self.spin_at(x, wrap_sub(y, self.height));
        let down = self.spin_at(x, (y + 1) % self.height);
        2 * i32::from(self.spins[idx]) * i32::from(left + right + up + down)
    }

    fn flip(&mut self, idx: usize) {
        self.spins[idx] *= -1;
    }

    fn spin_at(&self, x: usize, y: usize) -> i8 {
        self.spins[y * self.width + x]
    }
}

fn wrap_sub(value: usize, modulus: usize) -> usize {
    if value == 0 {
        modulus - 1
    } else {
        value - 1
    }
}

#[derive(Clone)]
struct SpinFlipUpdate {
    beta: f64,
    proposed_idx: usize,
}

impl SpinFlipUpdate {
    fn new(beta: f64) -> Self {
        Self {
            beta,
            proposed_idx: 0,
        }
    }
}

impl Update<IsingLattice> for SpinFlipUpdate {
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut IsingLattice, rng: &mut R) -> f64 {
        self.proposed_idx = uniform_index(rng, state.len());
        let delta_energy = state.delta_energy_for_flip(self.proposed_idx);
        if delta_energy <= 0 {
            1.0
        } else {
            (-self.beta * f64::from(delta_energy)).exp()
        }
    }

    fn accept(&mut self, state: &mut IsingLattice) {
        state.flip(self.proposed_idx);
    }
}

struct MagnetizationMeasurement {
    samples: u64,
    magnetization_sum: f64,
    abs_magnetization_sum: f64,
}

impl MagnetizationMeasurement {
    fn new() -> Self {
        Self {
            samples: 0,
            magnetization_sum: 0.0,
            abs_magnetization_sum: 0.0,
        }
    }
}

impl Measurement<IsingLattice> for MagnetizationMeasurement {
    type Output = IsingSummary;

    fn measure(&mut self, state: &IsingLattice) {
        let spins = state.len() as f64;
        let magnetization = state.magnetization() as f64 / spins;
        self.samples += 1;
        self.magnetization_sum += magnetization;
        self.abs_magnetization_sum += magnetization.abs();
    }

    fn finish(self) -> Self::Output {
        IsingSummary {
            samples: self.samples,
            magnetization_sum: self.magnetization_sum,
            abs_magnetization_sum: self.abs_magnetization_sum,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct IsingSummary {
    samples: u64,
    magnetization_sum: f64,
    abs_magnetization_sum: f64,
}

impl IsingSummary {
    fn mean_magnetization(&self) -> f64 {
        self.magnetization_sum / self.samples as f64
    }

    fn mean_abs_magnetization(&self) -> f64 {
        self.abs_magnetization_sum / self.samples as f64
    }
}

impl Merge for IsingSummary {
    fn merge(self, other: Self) -> Self {
        Self {
            samples: self.samples + other.samples,
            magnetization_sum: self.magnetization_sum + other.magnetization_sum,
            abs_magnetization_sum: self.abs_magnetization_sum + other.abs_magnetization_sum,
        }
    }
}

fn build_chain(
    _chain: ChainId,
) -> (
    IsingLattice,
    MetropolisKernel<SingleUpdateSet<SpinFlipUpdate>>,
    MagnetizationMeasurement,
) {
    let state = IsingLattice::ordered(16, 16);
    let update = SpinFlipUpdate::new(0.44); // critical point is at beta=0.440686
    let measurement = MagnetizationMeasurement::new();
    let kernel = MetropolisKernel::new(SingleUpdateSet::new(update));
    (state, kernel, measurement)
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 200_000,
        steps_per_cycle: 256,
        cycles_per_check: 1,
    }
}

fn main() -> rmc::Result<()> {
    let seed = SeedSource::new(0x15_1eaf);
    let mut rng = seed.rng_for(ChainId(0));
    let (state, mut kernel, measurement) = build_chain(ChainId(0));
    let (_state, single_stats, single_summary) =
        run_typed(state, &mut rng, &mut kernel, measurement, params())?;

    println!(
        "single chain: steps={}, samples={}, mean_m={:.3}, mean_abs_m={:.3}",
        single_stats.steps_done,
        single_summary.samples,
        single_summary.mean_magnetization(),
        single_summary.mean_abs_magnetization()
    );

    let chains = 8;
    let (parallel_stats, parallel_summary) = run_parallel(
        ParallelConfig {
            chains,
            seed,
            params: params(),
        },
        build_chain,
    )?;

    println!(
        "parallel: chains={}, steps={}, samples={}, mean_m={:.3}, mean_abs_m={:.3}",
        chains,
        parallel_stats.steps_done,
        parallel_summary.samples,
        parallel_summary.mean_magnetization(),
        parallel_summary.mean_abs_magnetization()
    );

    Ok(())
}

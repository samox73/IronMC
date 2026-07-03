use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rmc_core::mc::{
    run_parallel, run_parallel_full, run_parallel_in_pool, Measurement, MetropolisKernel,
    ParallelConfig, SimulationParams, SingleUpdateSet, Update, UpdateSet,
};
use rmc_core::random::{ChainId, SeedSource};

struct AtomicIncrementUpdate {
    value: Arc<AtomicU64>,
}

impl Update<()> for AtomicIncrementUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, _state: &mut ()) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
}

struct ProbabilisticAtomicIncrementUpdate {
    value: Arc<AtomicU64>,
}

impl Update<()> for ProbabilisticAtomicIncrementUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut (), _rng: &mut R) -> f64 {
        0.5
    }

    fn accept(&mut self, _state: &mut ()) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }
}

struct FinalValueMeasurement {
    value: Arc<AtomicU64>,
}

impl Measurement<()> for FinalValueMeasurement {
    type Output = u64;

    fn measure(&mut self, _state: &()) {}

    fn finish(self) -> Self::Output {
        self.value.load(Ordering::Relaxed)
    }
}

#[test]
fn run_parallel_merges_independent_chain_outputs() {
    let (stats, total_final_value) = run_parallel(
        ParallelConfig {
            chains: 4,
            seed: SeedSource::new(123),
            params: SimulationParams {
                max_steps: 5,
                steps_per_cycle: 1,
                cycles_per_check: 1,
            },
        },
        |_chain: ChainId| {
            let value = Arc::new(AtomicU64::new(0));
            let update = AtomicIncrementUpdate {
                value: Arc::clone(&value),
            };
            let measurement = FinalValueMeasurement { value };
            (
                (),
                MetropolisKernel::new(SingleUpdateSet::new(update)),
                measurement,
            )
        },
    )
    .unwrap();

    assert_eq!(stats.steps_done, 20);
    assert_eq!(stats.cycles_done, 20);
    assert_eq!(total_final_value, 20);
}

#[test]
fn run_parallel_rejects_zero_chains() {
    let err = run_parallel::<
        (),
        MetropolisKernel<SingleUpdateSet<AtomicIncrementUpdate>>,
        FinalValueMeasurement,
        _,
    >(
        ParallelConfig {
            chains: 0,
            seed: SeedSource::new(123),
            params: SimulationParams::default(),
        },
        |_chain| {
            let value = Arc::new(AtomicU64::new(0));
            (
                (),
                MetropolisKernel::new(SingleUpdateSet::new(AtomicIncrementUpdate {
                    value: Arc::clone(&value),
                })),
                FinalValueMeasurement { value },
            )
        },
    )
    .unwrap_err();

    assert_eq!(err.to_string(), "invalid argument: chains must be > 0");
}

#[test]
fn run_parallel_full_returns_final_kernels() {
    let (stats, total_final_value, kernels) = run_parallel_full(
        ParallelConfig {
            chains: 3,
            seed: SeedSource::new(123),
            params: SimulationParams {
                max_steps: 5,
                steps_per_cycle: 1,
                cycles_per_check: 1,
            },
        },
        |_chain: ChainId| {
            let value = Arc::new(AtomicU64::new(0));
            let update = AtomicIncrementUpdate {
                value: Arc::clone(&value),
            };
            let measurement = FinalValueMeasurement { value };
            (
                (),
                MetropolisKernel::new(SingleUpdateSet::new(update)),
                measurement,
            )
        },
    )
    .unwrap();

    assert_eq!(stats.steps_done, 15);
    assert_eq!(total_final_value, 15);
    assert_eq!(kernels.len(), 3);
    assert_eq!(
        kernels
            .iter()
            .map(|kernel| kernel.updates().stats()[0].nprops)
            .sum::<u64>(),
        15
    );
}

#[test]
fn run_parallel_is_reproducible_across_thread_counts() {
    let config = ParallelConfig {
        chains: 16,
        seed: SeedSource::new(0x5eed),
        params: SimulationParams {
            max_steps: 64,
            steps_per_cycle: 8,
            cycles_per_check: 1,
        },
    };

    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let one = run_parallel_in_pool(&one_thread, config, probabilistic_chain).unwrap();
    let four = run_parallel_in_pool(&four_threads, config, probabilistic_chain).unwrap();

    assert_eq!(one, four);
}

fn probabilistic_chain(
    _chain: ChainId,
) -> (
    (),
    MetropolisKernel<SingleUpdateSet<ProbabilisticAtomicIncrementUpdate>>,
    FinalValueMeasurement,
) {
    let value = Arc::new(AtomicU64::new(0));
    let update = ProbabilisticAtomicIncrementUpdate {
        value: Arc::clone(&value),
    };
    let measurement = FinalValueMeasurement { value };
    (
        (),
        MetropolisKernel::new(SingleUpdateSet::new(update)),
        measurement,
    )
}

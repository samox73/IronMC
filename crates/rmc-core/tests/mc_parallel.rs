use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rmc_core::mc::{
    Measurement, MetropolisKernel, Runner, SimulationParams, SingleUpdateSet, Update, UpdateSet,
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
fn runner_merges_independent_chain_outputs() {
    let report = Runner::new(SeedSource::new(123), |_chain: ChainId| {
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
    })
    .chains(4)
    .run(SimulationParams {
        max_steps: 5,
        steps_per_cycle: 1,
        cycles_per_check: 1,
    })
    .unwrap();

    assert_eq!(report.stats.steps_done, 20);
    assert_eq!(report.stats.cycles_done, 20);
    assert_eq!(report.output, 20);
}

#[test]
fn runner_rejects_zero_chains() {
    let err = match Runner::new(SeedSource::new(123), |_chain| {
        let value = Arc::new(AtomicU64::new(0));
        (
            (),
            MetropolisKernel::new(SingleUpdateSet::new(AtomicIncrementUpdate {
                value: Arc::clone(&value),
            })),
            FinalValueMeasurement { value },
        )
    })
    .chains(0)
    .run(SimulationParams::default())
    {
        Ok(_) => panic!("zero chains should fail"),
        Err(err) => err,
    };

    assert_eq!(err.to_string(), "invalid argument: chains must be > 0");
}

#[test]
fn runner_returns_final_kernels() {
    let report = Runner::new(SeedSource::new(123), |_chain: ChainId| {
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
    })
    .chains(3)
    .run(SimulationParams {
        max_steps: 5,
        steps_per_cycle: 1,
        cycles_per_check: 1,
    })
    .unwrap();

    assert_eq!(report.stats.steps_done, 15);
    assert_eq!(report.output, 15);
    assert_eq!(report.kernels.len(), 3);
    assert_eq!(
        report
            .kernels
            .iter()
            .map(|kernel| kernel.updates().stats()[0].nprops)
            .sum::<u64>(),
        15
    );
}

#[test]
fn runner_runs_warmup_before_measurement() {
    let report = Runner::new(SeedSource::new(123), |_chain| {
        (
            0_u64,
            MetropolisKernel::new(SingleUpdateSet::new(StateIncrementUpdate)),
            StateMeasurement,
        )
    })
    .chains(2)
    .warmup(SimulationParams {
        max_steps: 3,
        steps_per_cycle: 1,
        cycles_per_check: 1,
    })
    .run(SimulationParams {
        max_steps: 5,
        steps_per_cycle: 1,
        cycles_per_check: 1,
    })
    .unwrap();

    assert_eq!(report.stats.steps_done, 10);
    assert_eq!(report.output, 16);
    assert_eq!(report.states, vec![8, 8]);
    assert_eq!(
        report
            .kernels
            .iter()
            .map(|kernel| kernel.updates().stats()[0].nprops)
            .sum::<u64>(),
        10
    );
}

#[test]
fn runner_is_reproducible_across_thread_counts() {
    let params = SimulationParams {
        max_steps: 64,
        steps_per_cycle: 8,
        cycles_per_check: 1,
    };

    let one_thread = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let four_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    let one = Runner::new(SeedSource::new(0x5eed), probabilistic_chain)
        .chains(16)
        .pool(&one_thread)
        .run(params)
        .unwrap();
    let four = Runner::new(SeedSource::new(0x5eed), probabilistic_chain)
        .chains(16)
        .pool(&four_threads)
        .run(params)
        .unwrap();

    assert_eq!(one.output, four.output);
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

struct StateIncrementUpdate;

impl Update<u64> for StateIncrementUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut u64, _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut u64) {
        *state += 1;
    }
}

struct StateMeasurement;

impl Measurement<u64> for StateMeasurement {
    type Output = u64;

    fn measure(&mut self, state: &u64) {
        assert!(*state >= 4);
    }

    fn finish(self) -> Self::Output {
        8
    }
}

use rmc_core::dispatch_update;
use rmc_core::mc::{
    run_parallel, run_typed, Measurement, MetropolisKernel, ParallelConfig, SimulationParams,
    SingleUpdateSet, Update, UpdateSet,
};
use rmc_core::random::{ChainId, SeedSource};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CounterState {
    value: i64,
}

#[derive(Clone, Copy, Debug)]
struct AddUpdate {
    delta: i64,
}

impl Update<CounterState> for AddUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut CounterState, _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut CounterState) {
        state.value += self.delta;
    }
}

#[derive(Clone, Copy, Debug)]
struct RejectUpdate;

impl Update<CounterState> for RejectUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut CounterState, _rng: &mut R) -> f64 {
        0.0
    }

    fn accept(&mut self, _state: &mut CounterState) {}

    fn reject(&mut self, state: &mut CounterState) {
        state.value -= 1;
    }
}

dispatch_update! {
    #[derive(Clone, Copy, Debug)]
    enum CounterUpdate<CounterState> {
        Add(AddUpdate),
        Reject(RejectUpdate),
    }
    ; reject
}

#[derive(Clone, Copy, Debug, Default)]
struct CounterMeasurement {
    samples: u64,
    sum: i64,
}

impl Measurement<CounterState> for CounterMeasurement {
    type Output = i64;

    fn measure(&mut self, state: &CounterState) {
        self.samples += 1;
        self.sum += state.value;
    }

    fn finish(self) -> Self::Output {
        self.sum
    }
}

#[test]
fn dispatch_update_macro_forwards_update_methods() {
    let mut update = CounterUpdate::Reject(RejectUpdate);
    let mut state = CounterState { value: 3 };

    update.reject(&mut state);

    assert_eq!(state, CounterState { value: 2 });
    assert_eq!(CounterUpdate::Add(AddUpdate { delta: 1 }).name(), "Add");
}

#[test]
fn run_typed_owns_and_returns_state() {
    let mut rng = SeedSource::new(123).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(SingleUpdateSet::new(AddUpdate { delta: 2 }));

    let (state, stats, measured_sum) = run_typed(
        CounterState { value: 0 },
        &mut rng,
        &mut kernel,
        CounterMeasurement::default(),
        SimulationParams {
            max_steps: 6,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(state, CounterState { value: 12 });
    assert_eq!(stats.steps_done, 6);
    assert_eq!(stats.cycles_done, 3);
    assert_eq!(measured_sum, 4 + 8 + 12);
    assert_eq!(kernel.updates().stats()[0].nprops, 6);
    assert_eq!(kernel.updates().stats()[0].naccs, 6);
}

#[test]
fn run_parallel_merges_outputs_from_independent_states() {
    let (stats, measured_sum) = run_parallel(
        ParallelConfig {
            chains: 4,
            seed: SeedSource::new(123),
            params: SimulationParams {
                max_steps: 3,
                steps_per_cycle: 1,
                cycles_per_check: 1,
            },
        },
        |chain| {
            let state = CounterState {
                value: chain.0 as i64,
            };
            let kernel = MetropolisKernel::new(SingleUpdateSet::new(AddUpdate { delta: 1 }));
            (state, kernel, CounterMeasurement::default())
        },
    )
    .unwrap();

    assert_eq!(stats.steps_done, 12);
    assert_eq!(stats.cycles_done, 12);
    assert_eq!(measured_sum, 42);
}

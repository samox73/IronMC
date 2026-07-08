use rmc_core::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, Runner, SimulationParams,
    SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;

#[derive(Clone)]
struct Increment;

impl Update<i64> for Increment {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut i64, _rng: &mut R) -> f64 {
        *state += 1;
        1.0
    }

    fn accept(&mut self, _state: &mut i64) {}
}

#[derive(Default)]
struct SampleCount(u64);

impl Measurement<i64> for SampleCount {
    type Output = u64;

    fn measure(&mut self, _state: &i64) {
        self.0 += 1;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}

#[derive(Default)]
struct LastValue(i64);

impl Measurement<i64> for LastValue {
    type Output = i64;

    fn measure(&mut self, state: &i64) {
        self.0 = *state;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 5,
        steps_per_cycle: 2,
        cycles_per_check: 1,
    }
}

fn build_chain(
    _chain: ChainId,
) -> (
    i64,
    MetropolisKernel<SingleUpdateSet<Increment>>,
    (SampleCount, LastValue),
) {
    (
        0,
        MetropolisKernel::new(SingleUpdateSet::new(Increment)),
        (SampleCount::default(), LastValue::default()),
    )
}

#[test]
fn tuple_measurements_see_every_cycle() {
    let mut rng = SeedSource::new(7).rng_for(ChainId(0));
    let (state, mut kernel, measurement) = build_chain(ChainId(0));

    let (state, stats, output) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        measurement,
        params(),
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(state, 5);
    assert_eq!(stats.cycles_done, 3);
    assert_eq!(output, (3_u64, 5_i64));
}

#[test]
fn tuple_outputs_merge_for_runner() {
    assert_eq!((2_u64, 3.0_f64).merge((4, 1.5)), (6, 4.5));

    let report = Runner::new(SeedSource::new(7), build_chain)
        .chains(2)
        .run(params())
        .unwrap();

    assert_eq!(report.output, (6_u64, 10_i64));
}

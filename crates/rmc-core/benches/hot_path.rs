use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rmc_core::mc::{
    run_typed, Measurement, MetropolisKernel, SimulationParams, SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};

#[derive(Clone)]
struct IncrementStateUpdate;

impl Update<u64> for IncrementStateUpdate {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut u64, rng: &mut R) -> f64 {
        *state = state.wrapping_add(rng.next_u64() & 1);
        1.0
    }

    fn accept(&mut self, state: &mut u64) {
        *state = state.wrapping_add(1);
    }
}

struct FinalStateMeasurement;

impl Measurement<u64> for FinalStateMeasurement {
    type Output = u64;

    fn measure(&mut self, _state: &u64) {}

    fn finish(self) -> Self::Output {
        0
    }
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 10_000,
        steps_per_cycle: 10_000,
        cycles_per_check: 1,
    }
}

fn bench_static_single_update(c: &mut Criterion) {
    c.bench_function("static_single_update_10k_steps", |b| {
        b.iter_batched(
            || {
                let rng = SeedSource::new(0x5eed).rng_for(ChainId(0));
                let kernel = MetropolisKernel::new(SingleUpdateSet::new(IncrementStateUpdate));
                (rng, kernel)
            },
            |(mut rng, mut kernel)| {
                let (state, stats, output) = run_typed(
                    0_u64,
                    &mut rng,
                    &mut kernel,
                    FinalStateMeasurement,
                    params(),
                )
                .unwrap();
                black_box((state, stats, output));
            },
            BatchSize::SmallInput,
        )
    });
}

fn hot_path(c: &mut Criterion) {
    bench_static_single_update(c);
}

criterion_group!(benches, hot_path);
criterion_main!(benches);

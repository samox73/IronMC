use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use rmc_core::dispatch_update;
use rmc_core::mc::{
    run_typed, Measurement, MetropolisKernel, SimulationParams, SingleUpdateSet, Update,
    WeightedUpdate, WeightedUpdateSet,
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

struct CycleMeanMeasurement {
    count: u64,
    sum: u64,
    sum_sq: u64,
}

impl CycleMeanMeasurement {
    fn new() -> Self {
        Self {
            count: 0,
            sum: 0,
            sum_sq: 0,
        }
    }
}

impl Measurement<u64> for CycleMeanMeasurement {
    type Output = (u64, u64, u64);

    fn measure(&mut self, state: &u64) {
        self.count += 1;
        self.sum = self.sum.wrapping_add(*state);
        self.sum_sq = self.sum_sq.wrapping_add(state.wrapping_mul(*state));
    }

    fn finish(self) -> Self::Output {
        (self.count, self.sum, self.sum_sq)
    }
}

#[derive(Clone, Copy)]
struct TunedUpdate {
    delta: u64,
    reject_every: u64,
    proposal: u64,
}

impl TunedUpdate {
    fn new(delta: u64, reject_every: u64) -> Self {
        Self {
            delta,
            reject_every,
            proposal: 0,
        }
    }

    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut u64, rng: &mut R) -> f64 {
        self.proposal = state
            .wrapping_add(self.delta)
            .wrapping_add(rng.gen_range(0..8));
        if self.proposal % self.reject_every == 0 {
            0.15
        } else {
            0.85
        }
    }

    fn accept(&mut self, state: &mut u64) {
        *state = self.proposal;
    }

    fn reject(&mut self, _state: &mut u64) {}
}

dispatch_update! {
    enum BenchUpdate<u64> {
        A(TunedUpdate),
        B(TunedUpdate),
        C(TunedUpdate),
        D(TunedUpdate),
        E(TunedUpdate),
        F(TunedUpdate),
        G(TunedUpdate),
        H(TunedUpdate),
    }
    ; reject
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 10_000,
        steps_per_cycle: 10_000,
        cycles_per_check: 1,
    }
}

fn cycle_params() -> SimulationParams {
    SimulationParams {
        max_steps: 10_000,
        steps_per_cycle: 5,
        cycles_per_check: 1,
    }
}

fn weighted_updates() -> WeightedUpdateSet<BenchUpdate> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(BenchUpdate::A(TunedUpdate::new(1, 5)), 4.0),
        WeightedUpdate::new(BenchUpdate::B(TunedUpdate::new(2, 7)), 2.0),
        WeightedUpdate::new(BenchUpdate::C(TunedUpdate::new(3, 11)), 1.0),
        WeightedUpdate::new(BenchUpdate::D(TunedUpdate::new(5, 13)), 3.0),
        WeightedUpdate::new(BenchUpdate::E(TunedUpdate::new(8, 17)), 1.0),
        WeightedUpdate::new(BenchUpdate::F(TunedUpdate::new(13, 19)), 2.0),
        WeightedUpdate::new(BenchUpdate::G(TunedUpdate::new(21, 23)), 1.0),
        WeightedUpdate::new(BenchUpdate::H(TunedUpdate::new(34, 29)), 1.0),
    ])
    .unwrap()
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

fn bench_weighted_enum_update(c: &mut Criterion) {
    c.bench_function("weighted_enum_update_10k_steps", |b| {
        b.iter_batched(
            || {
                let rng = SeedSource::new(0x5eed).rng_for(ChainId(0));
                let kernel = MetropolisKernel::new(weighted_updates());
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

fn bench_weighted_enum_update_with_measurement(c: &mut Criterion) {
    c.bench_function("weighted_enum_update_measure_every_5_steps", |b| {
        b.iter_batched(
            || {
                let rng = SeedSource::new(0x5eed).rng_for(ChainId(0));
                let kernel = MetropolisKernel::new(weighted_updates());
                (rng, kernel)
            },
            |(mut rng, mut kernel)| {
                let (state, stats, output) = run_typed(
                    0_u64,
                    &mut rng,
                    &mut kernel,
                    CycleMeanMeasurement::new(),
                    cycle_params(),
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
    bench_weighted_enum_update(c);
    bench_weighted_enum_update_with_measurement(c);
}

criterion_group!(benches, hot_path);
criterion_main!(benches);

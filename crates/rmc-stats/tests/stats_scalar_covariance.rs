use rmc_core::mc::{
    Measurement, MetropolisKernel, Runner, SimulationParams, SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;
use rmc_stats::{Accumulator, MeanAccumulator, ScalarCovariance, VarianceAccumulator};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_pair_close(actual: (f64, f64), expected: (f64, f64)) {
    assert_close(actual.0, expected.0);
    assert_close(actual.1, expected.1);
}

#[test]
fn scalar_covariance_reports_empty_state_explicitly() {
    let covariance = ScalarCovariance::new();

    assert!(covariance.is_empty());
    assert_eq!(covariance.count(), 0);
    assert_eq!(covariance.mean(), None);
    assert_eq!(covariance.mean_x(), None);
    assert_eq!(covariance.mean_y(), None);
    assert_eq!(covariance.population_variance(), None);
    assert_eq!(covariance.sample_variance(), None);
    assert_eq!(covariance.population_covariance(), None);
    assert_eq!(covariance.sample_covariance(), None);
    assert_eq!(covariance.correlation(), None);
}

#[test]
fn scalar_covariance_matches_closed_form_pair_statistics() {
    let covariance = ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 3.0)]);

    assert_eq!(covariance.count(), 4);
    assert_pair_close(covariance.mean().unwrap(), (2.5, 2.5));
    assert_close(covariance.mean_x().unwrap(), 2.5);
    assert_close(covariance.mean_y().unwrap(), 2.5);
    assert_close(covariance.sum_squared_deviations_x(), 5.0);
    assert_close(covariance.sum_squared_deviations_y(), 5.0);
    assert_close(covariance.sum_cross_deviations(), 3.0);
    assert_pair_close(covariance.population_variance().unwrap(), (1.25, 1.25));
    assert_pair_close(
        covariance.sample_variance().unwrap(),
        (5.0 / 3.0, 5.0 / 3.0),
    );
    assert_close(covariance.population_covariance().unwrap(), 0.75);
    assert_close(covariance.sample_covariance().unwrap(), 1.0);
    assert_close(covariance.correlation().unwrap(), 0.6);
}

#[test]
fn scalar_covariance_reports_perfect_positive_and_negative_correlation() {
    let positive = ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]);
    let negative = ScalarCovariance::from_pairs([(1.0, 6.0), (2.0, 4.0), (3.0, 2.0)]);

    assert_close(positive.correlation().unwrap(), 1.0);
    assert_close(negative.correlation().unwrap(), -1.0);
}

#[test]
fn scalar_covariance_keeps_constant_axis_correlation_undefined() {
    let covariance = ScalarCovariance::from_pairs([(2.0, 1.0), (2.0, 2.0), (2.0, 3.0)]);

    assert_close(covariance.population_variance_x().unwrap(), 0.0);
    assert_close(covariance.sample_variance_x().unwrap(), 0.0);
    assert_close(covariance.population_variance_y().unwrap(), 2.0 / 3.0);
    assert_close(covariance.sample_variance_y().unwrap(), 1.0);
    assert_eq!(covariance.correlation(), None);
}

#[test]
fn scalar_covariance_merge_matches_single_pass_accumulation() {
    let first = ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 1.0)]);
    let second = ScalarCovariance::from_pairs([(3.0, 4.0), (4.0, 3.0), (5.0, 7.0)]);

    let merged = first.merge(second);
    let direct =
        ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 1.0), (3.0, 4.0), (4.0, 3.0), (5.0, 7.0)]);

    assert_eq!(merged.count(), direct.count());
    assert_pair_close(merged.mean().unwrap(), direct.mean().unwrap());
    assert_pair_close(
        merged.population_variance().unwrap(),
        direct.population_variance().unwrap(),
    );
    assert_pair_close(
        merged.sample_variance().unwrap(),
        direct.sample_variance().unwrap(),
    );
    assert_close(
        merged.population_covariance().unwrap(),
        direct.population_covariance().unwrap(),
    );
    assert_close(
        merged.sample_covariance().unwrap(),
        direct.sample_covariance().unwrap(),
    );
    assert_close(merged.correlation().unwrap(), direct.correlation().unwrap());
}

#[test]
fn scalar_covariance_merge_treats_empty_accumulators_as_identity() {
    let covariance = ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 1.0)]);

    assert_eq!(ScalarCovariance::new().merge(covariance), covariance);
    assert_eq!(covariance.merge(ScalarCovariance::new()), covariance);
}

struct IncrementState;

impl Update<u64> for IncrementState {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, _state: &mut u64, _rng: &mut R) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut u64) {
        *state += 1;
    }
}

#[derive(Default)]
struct StateCovariance {
    covariance: ScalarCovariance,
}

impl Measurement<u64> for StateCovariance {
    type Output = ScalarCovariance;

    fn measure(&mut self, state: &u64) {
        let x = *state as f64;
        self.covariance.accumulate((x, 2.0 * x));
    }

    fn finish(self) -> Self::Output {
        self.covariance
    }
}

#[test]
fn scalar_covariance_merges_parallel_measurement_outputs() {
    let covariance = Runner::new(SeedSource::new(43), |_chain: ChainId| {
        (
            0_u64,
            MetropolisKernel::new(SingleUpdateSet::new(IncrementState)),
            StateCovariance::default(),
        )
    })
    .chains(4)
    .run(SimulationParams {
        max_steps: 5,
        steps_per_cycle: 1,
        cycles_per_check: 1,
    })
    .unwrap()
    .output;

    assert_eq!(covariance.count(), 20);
    assert_pair_close(covariance.mean().unwrap(), (3.0, 6.0));
    assert_pair_close(covariance.population_variance().unwrap(), (2.0, 8.0));
    assert_close(covariance.population_covariance().unwrap(), 4.0);
    assert_close(covariance.correlation().unwrap(), 1.0);
}

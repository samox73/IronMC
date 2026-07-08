use rmc_core::mc::{
    Measurement, MetropolisKernel, Runner, SimulationParams, SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;
use rmc_stats::{
    Accumulator, MeanAccumulator, ScalarMoments, VarianceAccumulator, WeightedScalarMoments,
};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn scalar_moments_reports_empty_state_explicitly() {
    let moments = ScalarMoments::new();

    assert!(moments.is_empty());
    assert_eq!(moments.count(), 0);
    assert_eq!(moments.mean(), None);
    assert_eq!(moments.population_variance(), None);
    assert_eq!(moments.sample_variance(), None);
    assert_eq!(moments.standard_error(), None);
}

#[test]
fn scalar_moments_matches_closed_form_sequence_statistics() {
    let moments = ScalarMoments::from_samples((1..=10).map(f64::from));

    assert_eq!(moments.count(), 10);
    assert_close(moments.sum(), 55.0);
    assert_close(moments.mean().unwrap(), 5.5);
    assert_close(moments.population_variance().unwrap(), 8.25);
    assert_close(moments.sample_variance().unwrap(), 55.0 / 6.0);
    assert_close(moments.sum_squared_deviations(), 82.5);
    assert_close(moments.standard_error().unwrap(), (55.0_f64 / 60.0).sqrt());
}

#[test]
fn scalar_moments_keeps_constant_variance_zero() {
    let moments = ScalarMoments::from_samples([3.25; 64]);

    assert_eq!(moments.count(), 64);
    assert_close(moments.mean().unwrap(), 3.25);
    assert_close(moments.population_variance().unwrap(), 0.0);
    assert_close(moments.sample_variance().unwrap(), 0.0);
}

#[test]
fn scalar_moments_merge_matches_single_pass_accumulation() {
    let first = ScalarMoments::from_samples([1.0, 2.0, 3.0]);
    let second = ScalarMoments::from_samples([4.0, 5.0, 6.0, 7.0]);
    let third = ScalarMoments::from_samples([8.0, 9.0, 10.0]);

    let merged = first.merge(second).merge(third);
    let direct = ScalarMoments::from_samples((1..=10).map(f64::from));

    assert_eq!(merged.count(), direct.count());
    assert_close(merged.mean().unwrap(), direct.mean().unwrap());
    assert_close(
        merged.population_variance().unwrap(),
        direct.population_variance().unwrap(),
    );
    assert_close(
        merged.sample_variance().unwrap(),
        direct.sample_variance().unwrap(),
    );
    assert_close(
        merged.sum_squared_deviations(),
        direct.sum_squared_deviations(),
    );
}

#[test]
fn scalar_moments_merge_treats_empty_accumulators_as_identity() {
    let moments = ScalarMoments::from_samples([1.0, 2.0, 3.0]);

    assert_eq!(ScalarMoments::new().merge(moments), moments);
    assert_eq!(moments.merge(ScalarMoments::new()), moments);
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
struct StateMoments {
    moments: ScalarMoments,
}

impl Measurement<u64> for StateMoments {
    type Output = ScalarMoments;

    fn measure(&mut self, state: &u64) {
        self.moments.accumulate(*state as f64);
    }

    fn finish(self) -> Self::Output {
        self.moments
    }
}

#[test]
fn scalar_moments_merges_parallel_measurement_outputs() {
    let moments = Runner::new(SeedSource::new(42), |_chain: ChainId| {
        (
            0_u64,
            MetropolisKernel::new(SingleUpdateSet::new(IncrementState)),
            StateMoments::default(),
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

    assert_eq!(moments.count(), 20);
    assert_close(moments.mean().unwrap(), 3.0);
    assert_close(moments.population_variance().unwrap(), 2.0);
}

#[test]
fn weighted_scalar_moments_reports_empty_state_explicitly() {
    let moments = WeightedScalarMoments::new();

    assert!(moments.is_empty());
    assert_eq!(moments.count(), 0);
    assert_eq!(moments.total_weight(), 0.0);
    assert_eq!(moments.squared_weight_sum(), 0.0);
    assert_eq!(moments.effective_sample_size(), None);
    assert_eq!(moments.mean(), None);
    assert_eq!(moments.population_variance(), None);
    assert_eq!(moments.sample_variance(), None);
    assert_eq!(moments.standard_error(), None);
}

#[test]
fn weighted_scalar_moments_matches_closed_form_weighted_statistics() {
    let moments =
        WeightedScalarMoments::from_weighted_samples([(2.0, 1.0), (4.0, 2.0), (8.0, 1.0)]).unwrap();

    assert_eq!(moments.count(), 3);
    assert_close(moments.total_weight(), 4.0);
    assert_close(moments.squared_weight_sum(), 6.0);
    assert_close(moments.effective_sample_size().unwrap(), 16.0 / 6.0);
    assert_close(moments.weighted_sum(), 18.0);
    assert_close(moments.mean().unwrap(), 4.5);
    assert_close(moments.weighted_sum_squared_deviations(), 19.0);
    assert_close(moments.population_variance().unwrap(), 19.0 / 4.0);
    assert_close(moments.sample_variance().unwrap(), 19.0 / 2.5);
    assert_close(
        moments.standard_error().unwrap(),
        ((19.0_f64 / 2.5) / (16.0 / 6.0)).sqrt(),
    );
}

#[test]
fn weighted_scalar_moments_ignores_zero_weights() {
    let mut moments = WeightedScalarMoments::new();

    moments.try_accumulate_weighted(1000.0, 0.0).unwrap();
    moments.try_accumulate_weighted(3.0, 2.0).unwrap();

    assert_eq!(moments.count(), 1);
    assert_close(moments.total_weight(), 2.0);
    assert_close(moments.mean().unwrap(), 3.0);
    assert_close(moments.population_variance().unwrap(), 0.0);
    assert_eq!(moments.sample_variance(), None);
}

#[test]
fn weighted_scalar_moments_rejects_invalid_weights() {
    let negative = WeightedScalarMoments::from_weighted_samples([(1.0, -1.0)]).unwrap_err();
    let infinite =
        WeightedScalarMoments::from_weighted_samples([(1.0, f64::INFINITY)]).unwrap_err();
    let nan = WeightedScalarMoments::from_weighted_samples([(1.0, f64::NAN)]).unwrap_err();

    assert_eq!(
        negative.to_string(),
        "invalid argument: weight must be non-negative"
    );
    assert_eq!(
        infinite.to_string(),
        "invalid argument: weight must be finite"
    );
    assert_eq!(nan.to_string(), "invalid argument: weight must be finite");
}

#[test]
fn weighted_scalar_moments_unit_weights_match_unweighted_moments() {
    let weighted =
        WeightedScalarMoments::from_weighted_samples((1..=10).map(|value| (f64::from(value), 1.0)))
            .unwrap();
    let unweighted = ScalarMoments::from_samples((1..=10).map(f64::from));

    assert_eq!(weighted.count(), unweighted.count());
    assert_close(weighted.total_weight(), unweighted.count() as f64);
    assert_close(weighted.mean().unwrap(), unweighted.mean().unwrap());
    assert_close(
        weighted.population_variance().unwrap(),
        unweighted.population_variance().unwrap(),
    );
    assert_close(
        weighted.sample_variance().unwrap(),
        unweighted.sample_variance().unwrap(),
    );
}

#[test]
fn weighted_scalar_moments_merge_matches_single_pass_accumulation() {
    let first = WeightedScalarMoments::from_weighted_samples([(1.0, 0.5), (2.0, 1.5)]).unwrap();
    let second = WeightedScalarMoments::from_weighted_samples([(4.0, 2.0), (8.0, 1.0)]).unwrap();

    let merged = first.merge(second);
    let direct = WeightedScalarMoments::from_weighted_samples([
        (1.0, 0.5),
        (2.0, 1.5),
        (4.0, 2.0),
        (8.0, 1.0),
    ])
    .unwrap();

    assert_eq!(merged.count(), direct.count());
    assert_close(merged.total_weight(), direct.total_weight());
    assert_close(merged.squared_weight_sum(), direct.squared_weight_sum());
    assert_close(merged.mean().unwrap(), direct.mean().unwrap());
    assert_close(
        merged.population_variance().unwrap(),
        direct.population_variance().unwrap(),
    );
    assert_close(
        merged.sample_variance().unwrap(),
        direct.sample_variance().unwrap(),
    );
}

#[test]
fn weighted_scalar_moments_merge_treats_empty_accumulators_as_identity() {
    let moments = WeightedScalarMoments::from_weighted_samples([(1.0, 0.5), (2.0, 1.5)]).unwrap();

    assert_eq!(WeightedScalarMoments::new().merge(moments), moments);
    assert_eq!(moments.merge(WeightedScalarMoments::new()), moments);
}

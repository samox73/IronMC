use proptest::prelude::*;
use rmc_core::Merge;
use rmc_stats::{
    Accumulator, MeanAccumulator, ScalarCovariance, ScalarMoments, WeightedScalarMoments,
};

const ABS_TOL: f64 = 1.0e-8;
const REL_TOL: f64 = 1.0e-10;

fn finite_samples() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-1.0e6_f64..1.0e6, 0..32)
}

fn finite_pairs() -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((-1.0e6_f64..1.0e6, -1.0e6_f64..1.0e6), 0..32)
}

fn finite_weighted_samples() -> impl Strategy<Value = Vec<(f64, f64)>> {
    prop::collection::vec((-1.0e6_f64..1.0e6, 0.0_f64..1.0e3), 0..32)
}

fn assert_close(actual: f64, expected: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    let tolerance = ABS_TOL.max(REL_TOL * scale);
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_moments_close(actual: ScalarMoments, expected: ScalarMoments) {
    assert_eq!(actual.count(), expected.count());
    assert_close(actual.sum(), expected.sum());
    assert_close(
        actual.sum_squared_deviations(),
        expected.sum_squared_deviations(),
    );
    match (actual.mean(), expected.mean()) {
        (Some(actual), Some(expected)) => assert_close(actual, expected),
        (None, None) => {}
        pair => panic!("mean presence differs: {pair:?}"),
    }
}

fn assert_weighted_moments_close(actual: WeightedScalarMoments, expected: WeightedScalarMoments) {
    assert_eq!(actual.count(), expected.count());
    assert_close(actual.total_weight(), expected.total_weight());
    assert_close(actual.squared_weight_sum(), expected.squared_weight_sum());
    assert_close(actual.weighted_sum(), expected.weighted_sum());
    assert_close(
        actual.weighted_sum_squared_deviations(),
        expected.weighted_sum_squared_deviations(),
    );
    match (actual.mean(), expected.mean()) {
        (Some(actual), Some(expected)) => assert_close(actual, expected),
        (None, None) => {}
        pair => panic!("mean presence differs: {pair:?}"),
    }
}

fn assert_covariance_close(actual: ScalarCovariance, expected: ScalarCovariance) {
    assert_eq!(actual.count(), expected.count());
    assert_close(
        actual.sum_squared_deviations_x(),
        expected.sum_squared_deviations_x(),
    );
    assert_close(
        actual.sum_squared_deviations_y(),
        expected.sum_squared_deviations_y(),
    );
    assert_close(
        actual.sum_cross_deviations(),
        expected.sum_cross_deviations(),
    );
    match (actual.mean(), expected.mean()) {
        (Some(actual), Some(expected)) => {
            assert_close(actual.0, expected.0);
            assert_close(actual.1, expected.1);
        }
        (None, None) => {}
        pair => panic!("mean presence differs: {pair:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn scalar_moments_merge_is_commutative_and_associative(a in finite_samples(), b in finite_samples(), c in finite_samples()) {
        let a_acc = ScalarMoments::from_samples(a);
        let b_acc = ScalarMoments::from_samples(b);
        let c_acc = ScalarMoments::from_samples(c);

        assert_moments_close(a_acc.merge(b_acc), b_acc.merge(a_acc));
        assert_moments_close(a_acc.merge(b_acc).merge(c_acc), a_acc.merge(b_acc.merge(c_acc)));
    }

    #[test]
    fn scalar_moments_merge_matches_single_pass(a in finite_samples(), b in finite_samples()) {
        let mut all = a.clone();
        all.extend(b.iter().copied());

        let merged = ScalarMoments::from_samples(a).merge(ScalarMoments::from_samples(b));
        let direct = ScalarMoments::from_samples(all);

        assert_moments_close(merged, direct);
    }

    #[test]
    fn weighted_scalar_moments_merge_is_commutative_and_associative(a in finite_weighted_samples(), b in finite_weighted_samples(), c in finite_weighted_samples()) {
        let a_acc = WeightedScalarMoments::from_weighted_samples(a).unwrap();
        let b_acc = WeightedScalarMoments::from_weighted_samples(b).unwrap();
        let c_acc = WeightedScalarMoments::from_weighted_samples(c).unwrap();

        assert_weighted_moments_close(a_acc.merge(b_acc), b_acc.merge(a_acc));
        assert_weighted_moments_close(a_acc.merge(b_acc).merge(c_acc), a_acc.merge(b_acc.merge(c_acc)));
    }

    #[test]
    fn weighted_scalar_moments_merge_matches_single_pass(a in finite_weighted_samples(), b in finite_weighted_samples()) {
        let mut all = a.clone();
        all.extend(b.iter().copied());

        let merged = WeightedScalarMoments::from_weighted_samples(a).unwrap()
            .merge(WeightedScalarMoments::from_weighted_samples(b).unwrap());
        let direct = WeightedScalarMoments::from_weighted_samples(all).unwrap();

        assert_weighted_moments_close(merged, direct);
    }

    #[test]
    fn scalar_covariance_merge_is_commutative_and_associative(a in finite_pairs(), b in finite_pairs(), c in finite_pairs()) {
        let a_acc = ScalarCovariance::from_pairs(a);
        let b_acc = ScalarCovariance::from_pairs(b);
        let c_acc = ScalarCovariance::from_pairs(c);

        assert_covariance_close(a_acc.merge(b_acc), b_acc.merge(a_acc));
        assert_covariance_close(a_acc.merge(b_acc).merge(c_acc), a_acc.merge(b_acc.merge(c_acc)));
    }

    #[test]
    fn scalar_covariance_merge_matches_single_pass(a in finite_pairs(), b in finite_pairs()) {
        let mut all = a.clone();
        all.extend(b.iter().copied());

        let merged = ScalarCovariance::from_pairs(a).merge(ScalarCovariance::from_pairs(b));
        let direct = ScalarCovariance::from_pairs(all);

        assert_covariance_close(merged, direct);
    }
}

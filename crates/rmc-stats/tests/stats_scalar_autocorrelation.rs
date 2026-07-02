use rmc_core::Merge;
use rmc_stats::{Accumulator, ScalarAutocorrelation};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn scalar_autocorrelation_reports_empty_state_explicitly() {
    let autocorr = ScalarAutocorrelation::new(2);

    assert!(autocorr.is_empty());
    assert_eq!(autocorr.count(), 0);
    assert_eq!(autocorr.max_lag(), 2);
    assert_eq!(autocorr.mean(), None);
    assert_eq!(autocorr.population_variance(), None);
    assert_eq!(autocorr.pair_count(0), Some(0));
    assert_eq!(autocorr.pair_count(1), Some(0));
    assert_eq!(autocorr.pair_count(2), Some(0));
    assert_eq!(autocorr.pair_count(3), None);
    assert_eq!(autocorr.autocovariance(0), None);
    assert_eq!(autocorr.autocorrelation(0), None);
    assert_eq!(autocorr.integrated_autocorrelation_time(2), None);
}

#[test]
fn scalar_autocorrelation_matches_closed_form_sequence_statistics() {
    let autocorr = ScalarAutocorrelation::from_samples(2, [1.0, 2.0, 3.0, 4.0]);

    assert_eq!(autocorr.count(), 4);
    assert_eq!(autocorr.pair_count(0), Some(4));
    assert_eq!(autocorr.pair_count(1), Some(3));
    assert_eq!(autocorr.pair_count(2), Some(2));
    assert_close(autocorr.sum(), 10.0);
    assert_close(autocorr.sum_squares(), 30.0);
    assert_close(autocorr.mean().unwrap(), 2.5);
    assert_close(autocorr.population_variance().unwrap(), 1.25);
    assert_close(autocorr.autocovariance(0).unwrap(), 1.25);
    assert_close(autocorr.autocovariance(1).unwrap(), 5.0 / 12.0);
    assert_close(autocorr.autocovariance(2).unwrap(), -0.75);
    assert_close(autocorr.autocorrelation(0).unwrap(), 1.0);
    assert_close(autocorr.autocorrelation(1).unwrap(), 1.0 / 3.0);
    assert_close(autocorr.autocorrelation(2).unwrap(), -0.6);
    assert_close(
        autocorr.integrated_autocorrelation_time(2).unwrap(),
        0.5 + 1.0 / 3.0 - 0.6,
    );
}

#[test]
fn scalar_autocorrelation_matches_alternating_process() {
    let autocorr = ScalarAutocorrelation::from_samples(2, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

    assert_close(autocorr.mean().unwrap(), 0.0);
    assert_close(autocorr.population_variance().unwrap(), 1.0);
    assert_close(autocorr.autocorrelation(1).unwrap(), -1.0);
    assert_close(autocorr.autocorrelation(2).unwrap(), 1.0);
    assert_close(autocorr.integrated_autocorrelation_time(2).unwrap(), 0.5);
}

#[test]
fn scalar_autocorrelation_merge_keeps_independent_chains_separate() {
    let first = ScalarAutocorrelation::from_samples(1, [1.0, 1.0]);
    let second = ScalarAutocorrelation::from_samples(1, [-1.0, -1.0]);

    let merged = first.merge(second);
    let direct_concatenation = ScalarAutocorrelation::from_samples(1, [1.0, 1.0, -1.0, -1.0]);

    assert_eq!(merged.count(), 4);
    assert_eq!(merged.pair_count(1), Some(2));
    assert_eq!(direct_concatenation.pair_count(1), Some(3));
    assert_close(merged.mean().unwrap(), 0.0);
    assert_close(merged.population_variance().unwrap(), 1.0);
    assert_close(merged.autocorrelation(1).unwrap(), 1.0);
    assert_close(direct_concatenation.autocorrelation(1).unwrap(), 1.0 / 3.0);
}

#[test]
fn scalar_autocorrelation_merge_treats_empty_accumulators_as_identity() {
    let autocorr = ScalarAutocorrelation::from_samples(2, [1.0, 2.0, 3.0]);

    assert_eq!(
        ScalarAutocorrelation::new(2).merge(autocorr.clone()),
        autocorr
    );
    assert_eq!(
        autocorr.clone().merge(ScalarAutocorrelation::new(2)),
        autocorr
    );
}

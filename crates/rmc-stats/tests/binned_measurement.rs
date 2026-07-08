use rmc_core::mc::Measurement;
use rmc_core::Merge;
use rmc_stats::{BinnedScalar, ScalarBlockMeans};

#[test]
fn binned_scalar_matches_manual_block_jackknife() {
    let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).sin()).collect();
    let mut measurement = BinnedScalar::new(10, |s: &f64| *s).unwrap();

    for sample in &samples {
        measurement.measure(sample);
    }

    let output = measurement.finish();
    let reference = ScalarBlockMeans::from_samples(10, samples.iter().copied())
        .unwrap()
        .jackknife();

    assert_eq!(output.estimate(), reference.estimate());
    assert_eq!(output.standard_error(), reference.standard_error());
}

#[test]
fn binned_scalar_merge_matches_merged_block_means() {
    let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 0.37).sin()).collect();
    let (left, right) = samples.split_at(50);
    let mut left_measurement = BinnedScalar::new(10, |s: &f64| *s).unwrap();
    let mut right_measurement = BinnedScalar::new(10, |s: &f64| *s).unwrap();

    for sample in left {
        left_measurement.measure(sample);
    }
    for sample in right {
        right_measurement.measure(sample);
    }

    let output = left_measurement.finish().merge(right_measurement.finish());
    let reference = ScalarBlockMeans::from_samples(10, left.iter().copied())
        .unwrap()
        .merge(ScalarBlockMeans::from_samples(10, right.iter().copied()).unwrap())
        .jackknife();

    assert_eq!(output.estimate(), reference.estimate());
    assert_eq!(output.standard_error(), reference.standard_error());
}

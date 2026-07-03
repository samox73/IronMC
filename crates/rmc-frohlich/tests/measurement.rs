use nalgebra::Vector3;
use rmc_core::mc::Measurement;
use rmc_core::Merge;
use rmc_frohlich::measurement::{jackknife_ratio, BatchedSum, PolaronMeasurement};
use rmc_frohlich::{sanity, Diagram};

#[test]
fn jackknife_ratio_constant_series_has_zero_error() {
    let mut num = BatchedSum::new(8);
    let mut den = BatchedSum::new(8);
    for _ in 0..80 {
        num.push(6.0);
        den.push(3.0);
    }

    let estimate = jackknife_ratio(&num, &den, |num, den| num / den);
    assert!((estimate.mean - 2.0).abs() < 1.0e-12);
    assert!(estimate.stderr < 1.0e-12);
}

#[test]
fn measurement_collects_order_zero_and_linked_diagrams() {
    let d0 = Diagram::default();
    let d2 = Diagram::from_arcs(
        1.0,
        -1.1,
        0.0,
        30.0,
        1.0,
        &[
            (0.0, 1.0, Vector3::new(0.2, 0.1, 0.0)),
            (0.2, 0.8, Vector3::new(0.1, -0.2, 0.05)),
        ],
    );
    sanity::check_sanity(&d2).unwrap();

    let mut measurement = PolaronMeasurement::new(20, 30.0, 8, 80, -1.0168, 1_000, 1.5, &d0);
    for _ in 0..40 {
        measurement.measure(&d0);
        measurement.measure(&d2);
    }
    let stats = measurement.finish();

    assert_eq!(stats.sample_count, 80);
    assert_eq!(stats.zeroth.total_count(), 80);
    assert!((stats.zeroth.mean().unwrap() - 0.5).abs() < 1.0e-12);
    let energy = stats.jackknife_energy();
    let z = stats.jackknife_quasiparticle_weight();
    assert!(energy.mean.is_finite());
    assert!(z.mean.is_finite());
    assert!(stats.get_exact().iter().all(|value| value.is_finite()));
}

#[test]
fn polaron_stats_merge_adds_batch_accumulators() {
    let d = Diagram::default();
    let mut lhs = PolaronMeasurement::new(10, 30.0, 4, 12, -1.0168, 1_000, 1.5, &d);
    let mut rhs = PolaronMeasurement::new(10, 30.0, 4, 12, -1.0168, 1_000, 1.5, &d);
    for _ in 0..7 {
        lhs.measure(&d);
    }
    for _ in 0..11 {
        rhs.measure(&d);
    }

    let merged = lhs.finish().merge(rhs.finish());
    assert_eq!(merged.sample_count, 18);
    assert_eq!(merged.zeroth.total_count(), 18);
    assert_eq!(merged.exact.total_count(), 18);
}

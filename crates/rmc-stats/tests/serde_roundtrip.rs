#![cfg(feature = "serde")]

use nalgebra::DVector;
use rmc_stats::{
    Accumulator, ScalarAutocorrelation, ScalarBatchMeans, ScalarBlockMeans, ScalarCovariance,
    ScalarJackknife, ScalarMoments, VectorCovariance, VectorMoments, WeightedScalarMoments,
};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn round_trip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_string(value).unwrap();
    serde_json::from_str(&encoded).unwrap()
}

#[test]
fn serde_round_trips_scalar_moments() {
    let moments = ScalarMoments::from_samples((1..=10).map(f64::from));
    let restored: ScalarMoments = round_trip(&moments);

    assert_eq!(restored, moments);
    assert_eq!(restored.count(), 10);
}

#[test]
fn serde_round_trips_weighted_scalar_moments() {
    let moments =
        WeightedScalarMoments::from_weighted_samples([(1.0, 0.5), (2.0, 1.5), (4.0, 2.0)]).unwrap();
    let restored: WeightedScalarMoments = round_trip(&moments);

    assert_eq!(restored, moments);
    assert_eq!(restored.count(), 3);
}

#[test]
fn serde_round_trips_scalar_covariance() {
    let covariance = ScalarCovariance::from_pairs([(1.0, 2.0), (2.0, 1.0), (3.0, 4.0)]);
    let restored: ScalarCovariance = round_trip(&covariance);

    assert_eq!(restored, covariance);
    assert_eq!(restored.count(), 3);
}

#[test]
fn serde_round_trips_scalar_autocorrelation() {
    let autocorr = ScalarAutocorrelation::from_samples(3, [1.0, 2.0, 3.0, 4.0]);
    let restored: ScalarAutocorrelation = round_trip(&autocorr);

    assert_eq!(restored, autocorr);
    assert_eq!(restored.count(), 4);
    assert_eq!(restored.max_lag(), 3);
}

#[test]
fn serde_round_trips_scalar_batch_means() {
    let batches = ScalarBatchMeans::from_samples(2, [1.0, 3.0, 5.0, 7.0, 11.0]).unwrap();
    let restored: ScalarBatchMeans = round_trip(&batches);

    assert_eq!(restored, batches);
    assert_eq!(restored.count(), 5);
    assert_eq!(restored.completed_batch_count(), 2);
}

#[test]
fn serde_round_trips_scalar_block_means() {
    let blocks = ScalarBlockMeans::from_samples(2, [1.0, 3.0, 5.0, 7.0, 11.0]).unwrap();
    let restored: ScalarBlockMeans = round_trip(&blocks);

    assert_eq!(restored, blocks);
    assert_eq!(restored.count(), 5);
    assert_eq!(restored.completed_block_means(), &[2.0, 6.0]);
}

#[test]
fn serde_round_trips_scalar_jackknife() {
    let jackknife = ScalarJackknife::from_values([1.0, 2.0, 4.0, 8.0]);
    let restored: ScalarJackknife = round_trip(&jackknife);

    assert_eq!(restored, jackknife);
    assert_eq!(restored.count(), 4);
}

#[test]
fn serde_round_trips_vector_moments() {
    let moments = VectorMoments::from_samples(
        2,
        [
            DVector::from_column_slice(&[1.0, 2.0]),
            DVector::from_column_slice(&[3.0, 4.0]),
        ],
    )
    .unwrap();
    let restored: VectorMoments = round_trip(&moments);

    assert_eq!(restored, moments);
    assert_eq!(restored.count(), 2);
    assert_eq!(restored.dimension(), 2);
}

#[test]
fn serde_round_trips_vector_covariance() {
    let covariance = VectorCovariance::from_samples(
        2,
        [
            DVector::from_column_slice(&[1.0, 2.0]),
            DVector::from_column_slice(&[3.0, 4.0]),
        ],
    )
    .unwrap();
    let restored: VectorCovariance = round_trip(&covariance);

    assert_eq!(restored, covariance);
    assert_eq!(restored.count(), 2);
    assert_eq!(restored.dimension(), 2);
}

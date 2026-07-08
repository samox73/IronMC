use nalgebra::{DMatrix, DVector};
use rmc_core::mc::{
    Measurement, MetropolisKernel, Runner, SimulationParams, SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;
use rmc_stats::{
    Accumulator, MeanAccumulator, VarianceAccumulator, VectorCovariance, VectorMoments,
};

fn vector(values: &[f64]) -> DVector<f64> {
    DVector::from_column_slice(values)
}

fn matrix(rows: usize, cols: usize, values: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows, cols, values)
}

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_vector_close(actual: &DVector<f64>, expected: &DVector<f64>) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

fn assert_matrix_close(actual: &DMatrix<f64>, expected: &DMatrix<f64>) {
    assert_eq!(actual.shape(), expected.shape());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn vector_moments_reports_empty_state_and_dimension_errors() {
    let moments = VectorMoments::new(2).unwrap();

    assert!(moments.is_empty());
    assert_eq!(moments.count(), 0);
    assert_eq!(moments.dimension(), 2);
    assert_eq!(moments.mean(), None);
    assert_eq!(moments.population_variance(), None);
    assert_eq!(moments.sample_variance(), None);
    assert_eq!(moments.standard_error(), None);
    assert_eq!(
        VectorMoments::new(0).unwrap_err().to_string(),
        "invalid argument: dimension must be > 0"
    );

    let mut moments = VectorMoments::new(2).unwrap();
    let err = moments.try_accumulate(vector(&[1.0])).unwrap_err();
    assert_eq!(
        err.to_string(),
        "invalid argument: sample dimension 1 does not match accumulator dimension 2"
    );
}

#[test]
fn vector_moments_matches_closed_form_component_statistics() {
    let moments = VectorMoments::from_samples(
        2,
        [
            vector(&[1.0, 2.0]),
            vector(&[3.0, 4.0]),
            vector(&[5.0, 8.0]),
        ],
    )
    .unwrap();

    assert_eq!(moments.count(), 3);
    assert_vector_close(&moments.sum(), &vector(&[9.0, 14.0]));
    assert_vector_close(&moments.mean().unwrap(), &vector(&[3.0, 14.0 / 3.0]));
    assert_vector_close(
        &moments.sum_squared_deviations(),
        &vector(&[8.0, 56.0 / 3.0]),
    );
    assert_vector_close(
        &moments.population_variance().unwrap(),
        &vector(&[8.0 / 3.0, 56.0 / 9.0]),
    );
    assert_vector_close(
        &moments.sample_variance().unwrap(),
        &vector(&[4.0, 28.0 / 3.0]),
    );
    assert_vector_close(
        &moments.standard_error().unwrap(),
        &vector(&[(4.0_f64 / 3.0).sqrt(), (28.0_f64 / 9.0).sqrt()]),
    );
}

#[test]
fn vector_moments_merge_matches_single_pass_accumulation() {
    let first = VectorMoments::from_samples(2, [vector(&[1.0, 2.0])]).unwrap();
    let second =
        VectorMoments::from_samples(2, [vector(&[3.0, 4.0]), vector(&[5.0, 8.0])]).unwrap();

    let merged = first.merge(second);
    let direct = VectorMoments::from_samples(
        2,
        [
            vector(&[1.0, 2.0]),
            vector(&[3.0, 4.0]),
            vector(&[5.0, 8.0]),
        ],
    )
    .unwrap();

    assert_eq!(merged.count(), direct.count());
    assert_vector_close(&merged.mean().unwrap(), &direct.mean().unwrap());
    assert_vector_close(
        &merged.population_variance().unwrap(),
        &direct.population_variance().unwrap(),
    );
    assert_vector_close(
        &merged.sample_variance().unwrap(),
        &direct.sample_variance().unwrap(),
    );
}

#[test]
fn vector_covariance_matches_closed_form_matrix_statistics() {
    let moments = VectorCovariance::from_samples(
        2,
        [
            vector(&[1.0, 2.0]),
            vector(&[3.0, 4.0]),
            vector(&[5.0, 8.0]),
        ],
    )
    .unwrap();

    assert_eq!(moments.count(), 3);
    assert_vector_close(&moments.sum(), &vector(&[9.0, 14.0]));
    assert_vector_close(&moments.mean().unwrap(), &vector(&[3.0, 14.0 / 3.0]));
    assert_matrix_close(
        &moments.sum_cross_deviations(),
        &matrix(2, 2, &[8.0, 12.0, 12.0, 56.0 / 3.0]),
    );
    assert_matrix_close(
        &moments.population_covariance().unwrap(),
        &matrix(2, 2, &[8.0 / 3.0, 4.0, 4.0, 56.0 / 9.0]),
    );
    assert_matrix_close(
        &moments.sample_covariance().unwrap(),
        &matrix(2, 2, &[4.0, 6.0, 6.0, 28.0 / 3.0]),
    );
    assert_vector_close(
        &moments.population_variance().unwrap(),
        &vector(&[8.0 / 3.0, 56.0 / 9.0]),
    );
    assert_vector_close(
        &moments.sample_variance().unwrap(),
        &vector(&[4.0, 28.0 / 3.0]),
    );
}

#[test]
fn vector_covariance_merge_matches_single_pass_accumulation() {
    let first = VectorCovariance::from_samples(2, [vector(&[1.0, 2.0])]).unwrap();
    let second =
        VectorCovariance::from_samples(2, [vector(&[3.0, 4.0]), vector(&[5.0, 8.0])]).unwrap();

    let merged = first.merge(second);
    let direct = VectorCovariance::from_samples(
        2,
        [
            vector(&[1.0, 2.0]),
            vector(&[3.0, 4.0]),
            vector(&[5.0, 8.0]),
        ],
    )
    .unwrap();

    assert_eq!(merged.count(), direct.count());
    assert_vector_close(&merged.mean().unwrap(), &direct.mean().unwrap());
    assert_matrix_close(
        &merged.population_covariance().unwrap(),
        &direct.population_covariance().unwrap(),
    );
    assert_matrix_close(
        &merged.sample_covariance().unwrap(),
        &direct.sample_covariance().unwrap(),
    );
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

struct StateVectorMoments {
    moments: VectorMoments,
}

impl StateVectorMoments {
    fn new() -> Self {
        Self {
            moments: VectorMoments::new(2).unwrap(),
        }
    }
}

impl Measurement<u64> for StateVectorMoments {
    type Output = VectorMoments;

    fn measure(&mut self, state: &u64) {
        let x = *state as f64;
        self.moments.accumulate(vector(&[x, 2.0 * x]));
    }

    fn finish(self) -> Self::Output {
        self.moments
    }
}

#[test]
fn vector_moments_merges_parallel_measurement_outputs() {
    let moments = Runner::new(SeedSource::new(45), |_chain: ChainId| {
        (
            0_u64,
            MetropolisKernel::new(SingleUpdateSet::new(IncrementState)),
            StateVectorMoments::new(),
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
    assert_vector_close(&moments.mean().unwrap(), &vector(&[3.0, 6.0]));
    assert_vector_close(
        &moments.population_variance().unwrap(),
        &vector(&[2.0, 8.0]),
    );
}

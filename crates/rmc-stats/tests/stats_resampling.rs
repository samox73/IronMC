use rmc_core::mc::{
    run_parallel, Measurement, MetropolisKernel, ParallelConfig, SimulationParams, SingleUpdateSet,
    Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;
use rmc_stats::{
    Accumulator, MeanAccumulator, ScalarBatchMeans, ScalarBlockMeans, ScalarJackknife,
    VarianceAccumulator,
};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn scalar_batch_means_reports_empty_state_explicitly() {
    let batches = ScalarBatchMeans::new(3).unwrap();

    assert!(batches.is_empty());
    assert_eq!(batches.count(), 0);
    assert_eq!(batches.batch_size(), 3);
    assert_eq!(batches.completed_batch_count(), 0);
    assert_eq!(batches.partial_batch_len(), 0);
    assert_eq!(batches.mean(), None);
    assert_eq!(batches.population_variance(), None);
    assert_eq!(batches.sample_variance(), None);
    assert_eq!(batches.mean_of_completed_batches(), None);
    assert_eq!(batches.completed_batch_sample_variance(), None);
    assert_eq!(batches.batch_standard_error(), None);
}

#[test]
fn scalar_batch_means_rejects_zero_batch_size() {
    let error = ScalarBatchMeans::new(0).unwrap_err();

    assert_eq!(
        error.to_string(),
        "invalid argument: batch_size must be > 0"
    );
}

#[test]
fn scalar_batch_means_matches_closed_form_batches() {
    let batches = ScalarBatchMeans::from_samples(2, [1.0, 3.0, 5.0, 7.0, 11.0]).unwrap();

    assert_eq!(batches.count(), 5);
    assert_eq!(batches.completed_batch_count(), 2);
    assert_eq!(batches.partial_batch_len(), 1);
    assert_close(batches.sum(), 27.0);
    assert_close(batches.sum_squares(), 205.0);
    assert_close(batches.mean().unwrap(), 27.0 / 5.0);
    assert_close(
        batches.population_variance().unwrap(),
        205.0 / 5.0 - (27.0_f64 / 5.0).powi(2),
    );
    assert_close(batches.sample_variance().unwrap(), 59.2 / 4.0);
    assert_close(batches.mean_of_completed_batches().unwrap(), 4.0);
    assert_close(batches.completed_batch_sample_variance().unwrap(), 8.0);
    assert_close(batches.batch_standard_error().unwrap(), 2.0);
}

#[test]
fn scalar_batch_means_merge_keeps_partial_batches_separate() {
    let first = ScalarBatchMeans::from_samples(3, [1.0, 2.0]).unwrap();
    let second = ScalarBatchMeans::from_samples(3, [10.0, 20.0]).unwrap();
    let direct_concatenation = ScalarBatchMeans::from_samples(3, [1.0, 2.0, 10.0, 20.0]).unwrap();

    let merged = first.merge(second);

    assert_eq!(merged.count(), 4);
    assert_eq!(merged.completed_batch_count(), 0);
    assert_eq!(merged.partial_batch_len(), 0);
    assert_eq!(direct_concatenation.completed_batch_count(), 1);
    assert_eq!(direct_concatenation.partial_batch_len(), 1);
    assert_close(merged.mean().unwrap(), 8.25);
    assert_eq!(merged.batch_standard_error(), None);
}

#[test]
fn scalar_batch_means_merge_treats_empty_accumulators_as_identity() {
    let batches = ScalarBatchMeans::from_samples(2, [1.0, 2.0, 3.0]).unwrap();

    assert_eq!(
        ScalarBatchMeans::new(2).unwrap().merge(batches.clone()),
        batches
    );
    assert_eq!(
        batches.clone().merge(ScalarBatchMeans::new(2).unwrap()),
        batches
    );
}

#[test]
fn scalar_block_means_retains_completed_blocks_for_jackknife() {
    let blocks = ScalarBlockMeans::from_samples(2, [1.0, 3.0, 5.0, 7.0, 11.0]).unwrap();

    assert_eq!(blocks.count(), 5);
    assert_eq!(blocks.block_size(), 2);
    assert_eq!(blocks.completed_block_count(), 2);
    assert_eq!(blocks.partial_block_len(), 1);
    assert_eq!(blocks.completed_block_means(), &[2.0, 6.0]);
    assert_close(blocks.sum(), 27.0);
    assert_close(blocks.sum_squares(), 205.0);
    assert_close(blocks.mean().unwrap(), 27.0 / 5.0);
    assert_close(blocks.mean_of_completed_blocks().unwrap(), 4.0);
    assert_close(blocks.completed_block_sample_variance().unwrap(), 8.0);
    assert_close(blocks.block_standard_error().unwrap(), 2.0);

    let jackknife = blocks.jackknife();
    assert_eq!(jackknife.values(), &[2.0, 6.0]);
    assert_close(jackknife.estimate().unwrap(), 4.0);
    assert_close(jackknife.standard_error().unwrap(), 2.0);
}

#[test]
fn scalar_block_means_rejects_zero_block_size() {
    let error = ScalarBlockMeans::new(0).unwrap_err();

    assert_eq!(
        error.to_string(),
        "invalid argument: block_size must be > 0"
    );
}

#[test]
fn scalar_block_means_merge_keeps_partial_blocks_separate() {
    let first = ScalarBlockMeans::from_samples(3, [1.0, 2.0, 3.0, 4.0]).unwrap();
    let second = ScalarBlockMeans::from_samples(3, [10.0, 20.0]).unwrap();
    let direct_concatenation =
        ScalarBlockMeans::from_samples(3, [1.0, 2.0, 3.0, 4.0, 10.0, 20.0]).unwrap();

    let merged = first.merge(second);

    assert_eq!(merged.count(), 6);
    assert_eq!(merged.completed_block_means(), &[2.0]);
    assert_eq!(merged.partial_block_len(), 0);
    assert_eq!(
        direct_concatenation.completed_block_means(),
        &[2.0, 34.0 / 3.0]
    );
    assert_close(merged.mean().unwrap(), 40.0 / 6.0);
    assert_eq!(merged.block_standard_error(), None);
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

struct StateBatchMeans {
    batches: ScalarBatchMeans,
}

impl StateBatchMeans {
    fn new(batch_size: usize) -> Self {
        Self {
            batches: ScalarBatchMeans::new(batch_size).unwrap(),
        }
    }
}

impl Measurement<u64> for StateBatchMeans {
    type Output = ScalarBatchMeans;

    fn measure(&mut self, state: &u64) {
        self.batches.accumulate(*state as f64);
    }

    fn finish(self) -> Self::Output {
        self.batches
    }
}

#[test]
fn scalar_batch_means_merges_parallel_measurement_outputs() {
    let (_stats, batches) = run_parallel(
        ParallelConfig {
            chains: 4,
            seed: SeedSource::new(44),
            params: SimulationParams {
                max_steps: 6,
                steps_per_cycle: 1,
                cycles_per_check: 1,
            },
        },
        |_chain: ChainId| {
            (
                0_u64,
                MetropolisKernel::new(SingleUpdateSet::new(IncrementState)),
                StateBatchMeans::new(2),
            )
        },
    )
    .unwrap();

    assert_eq!(batches.count(), 24);
    assert_eq!(batches.completed_batch_count(), 12);
    assert_eq!(batches.partial_batch_len(), 0);
    assert_close(batches.mean().unwrap(), 3.5);
    assert_close(batches.mean_of_completed_batches().unwrap(), 3.5);
    assert_close(
        batches.batch_standard_error().unwrap(),
        (8.0_f64 / 33.0).sqrt(),
    );
}

#[test]
fn scalar_jackknife_reports_empty_and_singleton_state_explicitly() {
    let empty = ScalarJackknife::new();
    let singleton = ScalarJackknife::from_values([2.0]);

    assert!(empty.is_empty());
    assert_eq!(empty.count(), 0);
    assert_eq!(empty.estimate(), None);
    assert_eq!(empty.delete_one_estimates(), None);
    assert_eq!(empty.bias(), None);
    assert_eq!(empty.bias_corrected_estimate(), None);
    assert_eq!(empty.standard_error(), None);

    assert_eq!(singleton.count(), 1);
    assert_close(singleton.estimate().unwrap(), 2.0);
    assert_eq!(singleton.delete_one_estimates(), None);
    assert_eq!(singleton.standard_error(), None);
}

#[test]
fn scalar_jackknife_matches_closed_form_delete_one_statistics() {
    let jackknife = ScalarJackknife::from_values([1.0, 2.0, 4.0, 8.0]);
    let delete_one = jackknife.delete_one_estimates().unwrap();

    assert_eq!(jackknife.count(), 4);
    assert_eq!(jackknife.values(), &[1.0, 2.0, 4.0, 8.0]);
    assert_close(jackknife.estimate().unwrap(), 3.75);
    assert_close(delete_one[0], 14.0 / 3.0);
    assert_close(delete_one[1], 13.0 / 3.0);
    assert_close(delete_one[2], 11.0 / 3.0);
    assert_close(delete_one[3], 7.0 / 3.0);
    assert_close(jackknife.bias().unwrap(), 0.0);
    assert_close(jackknife.bias_corrected_estimate().unwrap(), 3.75);
    assert_close(
        jackknife.standard_error().unwrap(),
        (115.0_f64 / 48.0).sqrt(),
    );
}

#[test]
fn scalar_jackknife_merge_concatenates_block_estimates() {
    let first = ScalarJackknife::from_values([1.0, 2.0]);
    let second = ScalarJackknife::from_values([4.0, 8.0]);

    let merged = first.merge(second);

    assert_eq!(merged.values(), &[1.0, 2.0, 4.0, 8.0]);
    assert_close(merged.estimate().unwrap(), 3.75);
    assert_close(merged.standard_error().unwrap(), (115.0_f64 / 48.0).sqrt());
}

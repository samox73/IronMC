//! Mergeable statistical accumulators.
//!
//! This module starts with scalar `f64` moments because those are the smallest useful building
//! block for Monte Carlo observables. Accumulators store mergeable sufficient statistics rather
//! than already-normalized estimates, so independent chain outputs can be reduced with [`Merge`].

use nalgebra::{DMatrix, DVector};
use rmc_core::{Merge, Result, RmcError};

/// Common accumulator operations.
pub trait Accumulator<T>: Merge {
    /// Number of samples accumulated so far.
    fn count(&self) -> u64;

    /// Add one sample to the accumulator.
    fn accumulate(&mut self, sample: T);

    /// Whether the accumulator has no samples.
    fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Accumulator that can report a mean estimate.
pub trait MeanAccumulator<T>: Accumulator<T> {
    /// Mean of all accumulated samples, or `None` when empty.
    fn mean(&self) -> Option<T>;
}

/// Accumulator that can report variance estimates.
pub trait VarianceAccumulator<T>: MeanAccumulator<T> {
    /// Population variance, normalized by `n`.
    fn population_variance(&self) -> Option<T>;

    /// Unbiased sample variance, normalized by `n - 1`.
    fn sample_variance(&self) -> Option<T>;
}

/// Online scalar mean/variance accumulator.
///
/// `ScalarMoments` uses Welford's algorithm for numerically stable single-pass accumulation and
/// the Chan-Golub-LeVeque merge formula for combining independent partial accumulators. This makes
/// it suitable as a typed measurement output for `run_parallel`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScalarMoments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl ScalarMoments {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an accumulator from an iterator of samples.
    pub fn from_samples(samples: impl IntoIterator<Item = f64>) -> Self {
        let mut acc = Self::new();
        for sample in samples {
            acc.accumulate(sample);
        }
        acc
    }

    /// Sum reconstructed from the stored count and mean.
    pub fn sum(&self) -> f64 {
        self.mean * self.count as f64
    }

    /// Sum of squared deviations from the current mean.
    pub fn sum_squared_deviations(&self) -> f64 {
        self.m2
    }

    /// Standard error of the mean, using the unbiased sample variance.
    pub fn standard_error(&self) -> Option<f64> {
        self.sample_variance()
            .map(|variance| (variance / self.count as f64).sqrt())
    }
}

impl Accumulator<f64> for ScalarMoments {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, sample: f64) {
        self.count += 1;
        let count = self.count as f64;
        let delta = sample - self.mean;
        self.mean += delta / count;
        let delta_after = sample - self.mean;
        self.m2 += delta * delta_after;
    }
}

impl MeanAccumulator<f64> for ScalarMoments {
    fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }
}

impl VarianceAccumulator<f64> for ScalarMoments {
    fn population_variance(&self) -> Option<f64> {
        (self.count > 0).then_some(self.m2 / self.count as f64)
    }

    fn sample_variance(&self) -> Option<f64> {
        if self.count > 1 {
            Some(self.m2 / (self.count - 1) as f64)
        } else {
            None
        }
    }
}

impl Merge for ScalarMoments {
    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let combined_count = self.count + other.count;
        let self_count = self.count as f64;
        let other_count = other.count as f64;
        let combined_count_f64 = combined_count as f64;
        let delta = other.mean - self.mean;

        Self {
            count: combined_count,
            mean: self.mean + delta * other_count / combined_count_f64,
            m2: self.m2 + other.m2 + delta * delta * self_count * other_count / combined_count_f64,
        }
    }
}

/// Online weighted scalar mean/variance accumulator.
///
/// `WeightedScalarMoments` stores the total weight, squared-weight sum, mean, and weighted sum of
/// squared deviations. It accepts non-negative finite weights. Zero-weight samples are ignored.
/// The population variance is normalized by total weight; the sample variance uses the standard
/// reliability-weight correction `sum(w) - sum(w^2) / sum(w)`, which reduces to `n - 1` for unit
/// weights.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WeightedScalarMoments {
    count: u64,
    total_weight: f64,
    squared_weight_sum: f64,
    mean: f64,
    m2: f64,
}

impl WeightedScalarMoments {
    /// Create an empty weighted accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a weighted accumulator from `(sample, weight)` pairs.
    pub fn from_weighted_samples(samples: impl IntoIterator<Item = (f64, f64)>) -> Result<Self> {
        let mut acc = Self::new();
        for (sample, weight) in samples {
            acc.try_accumulate_weighted(sample, weight)?;
        }
        Ok(acc)
    }

    /// Add one weighted sample.
    pub fn try_accumulate_weighted(&mut self, sample: f64, weight: f64) -> Result<()> {
        if !weight.is_finite() {
            return Err(RmcError::InvalidArgument(
                "weight must be finite".to_string(),
            ));
        }
        if weight < 0.0 {
            return Err(RmcError::InvalidArgument(
                "weight must be non-negative".to_string(),
            ));
        }
        if weight == 0.0 {
            return Ok(());
        }

        let new_total_weight = self.total_weight + weight;
        let delta = sample - self.mean;
        self.mean += delta * weight / new_total_weight;
        let delta_after = sample - self.mean;
        self.m2 += weight * delta * delta_after;
        self.total_weight = new_total_weight;
        self.squared_weight_sum += weight * weight;
        self.count += 1;
        Ok(())
    }

    /// Number of positive-weight samples accumulated.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sum of all positive weights.
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Sum of squared positive weights.
    pub fn squared_weight_sum(&self) -> f64 {
        self.squared_weight_sum
    }

    /// Effective sample size, `sum(w)^2 / sum(w^2)`.
    pub fn effective_sample_size(&self) -> Option<f64> {
        (self.squared_weight_sum > 0.0)
            .then_some(self.total_weight * self.total_weight / self.squared_weight_sum)
    }

    /// Whether the accumulator has no positive-weight samples.
    pub fn is_empty(&self) -> bool {
        self.total_weight == 0.0
    }

    /// Mean of all weighted samples, or `None` when empty.
    pub fn mean(&self) -> Option<f64> {
        (!self.is_empty()).then_some(self.mean)
    }

    /// Weighted sum reconstructed from the stored total weight and mean.
    pub fn weighted_sum(&self) -> f64 {
        self.mean * self.total_weight
    }

    /// Weighted sum of squared deviations from the current mean.
    pub fn weighted_sum_squared_deviations(&self) -> f64 {
        self.m2
    }

    /// Population variance, normalized by total weight.
    pub fn population_variance(&self) -> Option<f64> {
        (!self.is_empty()).then_some(self.m2 / self.total_weight)
    }

    /// Reliability-weighted unbiased sample variance.
    pub fn sample_variance(&self) -> Option<f64> {
        if self.total_weight == 0.0 {
            return None;
        }

        let denominator = self.total_weight - self.squared_weight_sum / self.total_weight;
        (denominator > 0.0).then_some(self.m2 / denominator)
    }

    /// Standard error of the weighted mean, using the effective sample size.
    pub fn standard_error(&self) -> Option<f64> {
        let variance = self.sample_variance()?;
        let effective_sample_size = self.effective_sample_size()?;
        Some((variance / effective_sample_size).sqrt())
    }
}

impl Accumulator<f64> for WeightedScalarMoments {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, sample: f64) {
        self.try_accumulate_weighted(sample, 1.0)
            .expect("unit weight is valid");
    }
}

impl MeanAccumulator<f64> for WeightedScalarMoments {
    fn mean(&self) -> Option<f64> {
        Self::mean(self)
    }
}

impl VarianceAccumulator<f64> for WeightedScalarMoments {
    fn population_variance(&self) -> Option<f64> {
        Self::population_variance(self)
    }

    fn sample_variance(&self) -> Option<f64> {
        Self::sample_variance(self)
    }
}

impl Merge for WeightedScalarMoments {
    fn merge(self, other: Self) -> Self {
        if self.total_weight == 0.0 {
            return other;
        }
        if other.total_weight == 0.0 {
            return self;
        }

        let combined_weight = self.total_weight + other.total_weight;
        let delta = other.mean - self.mean;

        Self {
            count: self.count + other.count,
            total_weight: combined_weight,
            squared_weight_sum: self.squared_weight_sum + other.squared_weight_sum,
            mean: self.mean + delta * other.total_weight / combined_weight,
            m2: self.m2
                + other.m2
                + delta * delta * self.total_weight * other.total_weight / combined_weight,
        }
    }
}

/// Online covariance accumulator for pairs of scalar `f64` samples.
///
/// `ScalarCovariance` tracks per-axis means, per-axis squared deviations, and cross deviations.
/// Like [`ScalarMoments`], it can be merged from independent partial accumulators without storing
/// the original samples.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScalarCovariance {
    count: u64,
    mean_x: f64,
    mean_y: f64,
    m2_x: f64,
    m2_y: f64,
    c2: f64,
}

impl ScalarCovariance {
    /// Create an empty covariance accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a covariance accumulator from `(x, y)` sample pairs.
    pub fn from_pairs(samples: impl IntoIterator<Item = (f64, f64)>) -> Self {
        let mut acc = Self::new();
        for sample in samples {
            acc.accumulate(sample);
        }
        acc
    }

    /// Mean of the first component, or `None` when empty.
    pub fn mean_x(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean_x)
    }

    /// Mean of the second component, or `None` when empty.
    pub fn mean_y(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean_y)
    }

    /// Sum of squared deviations of the first component.
    pub fn sum_squared_deviations_x(&self) -> f64 {
        self.m2_x
    }

    /// Sum of squared deviations of the second component.
    pub fn sum_squared_deviations_y(&self) -> f64 {
        self.m2_y
    }

    /// Sum of cross deviations from the current means.
    pub fn sum_cross_deviations(&self) -> f64 {
        self.c2
    }

    /// Population variance of the first component, normalized by `n`.
    pub fn population_variance_x(&self) -> Option<f64> {
        (self.count > 0).then_some(self.m2_x / self.count as f64)
    }

    /// Population variance of the second component, normalized by `n`.
    pub fn population_variance_y(&self) -> Option<f64> {
        (self.count > 0).then_some(self.m2_y / self.count as f64)
    }

    /// Unbiased sample variance of the first component, normalized by `n - 1`.
    pub fn sample_variance_x(&self) -> Option<f64> {
        if self.count > 1 {
            Some(self.m2_x / (self.count - 1) as f64)
        } else {
            None
        }
    }

    /// Unbiased sample variance of the second component, normalized by `n - 1`.
    pub fn sample_variance_y(&self) -> Option<f64> {
        if self.count > 1 {
            Some(self.m2_y / (self.count - 1) as f64)
        } else {
            None
        }
    }

    /// Population covariance, normalized by `n`.
    pub fn population_covariance(&self) -> Option<f64> {
        (self.count > 0).then_some(self.c2 / self.count as f64)
    }

    /// Unbiased sample covariance, normalized by `n - 1`.
    pub fn sample_covariance(&self) -> Option<f64> {
        if self.count > 1 {
            Some(self.c2 / (self.count - 1) as f64)
        } else {
            None
        }
    }

    /// Pearson correlation coefficient.
    pub fn correlation(&self) -> Option<f64> {
        if self.m2_x > 0.0 && self.m2_y > 0.0 {
            Some(self.c2 / (self.m2_x * self.m2_y).sqrt())
        } else {
            None
        }
    }
}

impl Accumulator<(f64, f64)> for ScalarCovariance {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, (x, y): (f64, f64)) {
        self.count += 1;
        let count = self.count as f64;

        let delta_x = x - self.mean_x;
        self.mean_x += delta_x / count;
        let delta_x_after = x - self.mean_x;
        self.m2_x += delta_x * delta_x_after;

        let delta_y = y - self.mean_y;
        self.mean_y += delta_y / count;
        let delta_y_after = y - self.mean_y;
        self.m2_y += delta_y * delta_y_after;

        self.c2 += delta_x * delta_y_after;
    }
}

impl MeanAccumulator<(f64, f64)> for ScalarCovariance {
    fn mean(&self) -> Option<(f64, f64)> {
        (self.count > 0).then_some((self.mean_x, self.mean_y))
    }
}

impl VarianceAccumulator<(f64, f64)> for ScalarCovariance {
    fn population_variance(&self) -> Option<(f64, f64)> {
        (self.count > 0).then_some((self.m2_x / self.count as f64, self.m2_y / self.count as f64))
    }

    fn sample_variance(&self) -> Option<(f64, f64)> {
        if self.count > 1 {
            Some((
                self.m2_x / (self.count - 1) as f64,
                self.m2_y / (self.count - 1) as f64,
            ))
        } else {
            None
        }
    }
}

impl Merge for ScalarCovariance {
    fn merge(self, other: Self) -> Self {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let combined_count = self.count + other.count;
        let self_count = self.count as f64;
        let other_count = other.count as f64;
        let combined_count_f64 = combined_count as f64;
        let delta_x = other.mean_x - self.mean_x;
        let delta_y = other.mean_y - self.mean_y;
        let merge_weight = self_count * other_count / combined_count_f64;

        Self {
            count: combined_count,
            mean_x: self.mean_x + delta_x * other_count / combined_count_f64,
            mean_y: self.mean_y + delta_y * other_count / combined_count_f64,
            m2_x: self.m2_x + other.m2_x + delta_x * delta_x * merge_weight,
            m2_y: self.m2_y + other.m2_y + delta_y * delta_y * merge_weight,
            c2: self.c2 + other.c2 + delta_x * delta_y * merge_weight,
        }
    }
}

/// Scalar autocorrelation accumulator for a fixed set of non-negative lags.
///
/// `ScalarAutocorrelation` stores raw per-lag pair sums rather than normalized estimates. This lets
/// independent chains be merged without introducing artificial cross-chain lag pairs. After a merge,
/// the short active history buffer is cleared; future samples start a new independent segment.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarAutocorrelation {
    max_lag: usize,
    count: u64,
    sum: f64,
    sum_sq: f64,
    pair_counts: Vec<u64>,
    pair_sum_left: Vec<f64>,
    pair_sum_right: Vec<f64>,
    pair_sum_products: Vec<f64>,
    history: Vec<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    history_head: usize,
}

impl ScalarAutocorrelation {
    /// Create an empty autocorrelation accumulator for lags `0..=max_lag`.
    pub fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
            pair_counts: vec![0; max_lag],
            pair_sum_left: vec![0.0; max_lag],
            pair_sum_right: vec![0.0; max_lag],
            pair_sum_products: vec![0.0; max_lag],
            history: Vec::with_capacity(max_lag),
            history_head: 0,
        }
    }

    /// Build an autocorrelation accumulator from samples.
    pub fn from_samples(max_lag: usize, samples: impl IntoIterator<Item = f64>) -> Self {
        let mut acc = Self::new(max_lag);
        for sample in samples {
            acc.accumulate(sample);
        }
        acc
    }

    fn active_history_head(&self) -> usize {
        if self.history.len() < self.max_lag && self.history_head == 0 {
            self.history.len()
        } else {
            self.history_head % self.max_lag
        }
    }

    /// Largest lag tracked by this accumulator.
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Mean of all accumulated samples, or `None` when empty.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.sum / self.count as f64)
    }

    /// Sum of all accumulated samples.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Sum of squared samples.
    pub fn sum_squares(&self) -> f64 {
        self.sum_sq
    }

    /// Number of same-chain pairs available for `lag`.
    pub fn pair_count(&self, lag: usize) -> Option<u64> {
        if lag == 0 {
            Some(self.count)
        } else {
            self.pair_counts.get(lag - 1).copied()
        }
    }

    /// Population variance, equal to lag-zero autocovariance.
    pub fn population_variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        Some(self.sum_sq / self.count as f64 - mean * mean)
    }

    /// Population autocovariance at `lag`, normalized by the number of available pairs.
    pub fn autocovariance(&self, lag: usize) -> Option<f64> {
        if lag == 0 {
            return self.population_variance();
        }

        let idx = lag.checked_sub(1)?;
        if idx >= self.max_lag {
            return None;
        }

        let pair_count = self.pair_counts[idx];
        if pair_count == 0 {
            return None;
        }

        let mean = self.mean()?;
        let centered_sum = self.pair_sum_products[idx]
            - mean * (self.pair_sum_left[idx] + self.pair_sum_right[idx])
            + pair_count as f64 * mean * mean;
        Some(centered_sum / pair_count as f64)
    }

    /// Normalized autocorrelation at `lag`.
    pub fn autocorrelation(&self, lag: usize) -> Option<f64> {
        if lag > self.max_lag {
            return None;
        }

        let variance = self.population_variance()?;
        if variance <= 0.0 {
            return None;
        }

        if lag == 0 {
            Some(1.0)
        } else {
            self.autocovariance(lag)
                .map(|autocovariance| autocovariance / variance)
        }
    }

    /// Integrated autocorrelation time estimate through `cutoff_lag`.
    ///
    /// The estimate is `0.5 + sum_{lag=1..cutoff} rho(lag)`. Callers choose the cutoff policy; this
    /// method only evaluates the requested finite window.
    pub fn integrated_autocorrelation_time(&self, cutoff_lag: usize) -> Option<f64> {
        if self.population_variance()? <= 0.0 {
            return None;
        }

        let cutoff = cutoff_lag.min(self.max_lag);
        let mut tau = 0.5;
        for lag in 1..=cutoff {
            tau += self.autocorrelation(lag)?;
        }
        Some(tau)
    }
}

impl Accumulator<f64> for ScalarAutocorrelation {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, sample: f64) {
        let available_lags = self.max_lag.min(self.history.len());
        let history_head = if self.max_lag > 0 {
            self.active_history_head()
        } else {
            0
        };
        for lag in 1..=available_lags {
            let previous = self.history[(history_head + self.max_lag - lag) % self.max_lag];
            let idx = lag - 1;
            self.pair_counts[idx] += 1;
            self.pair_sum_left[idx] += previous;
            self.pair_sum_right[idx] += sample;
            self.pair_sum_products[idx] += previous * sample;
        }

        self.count += 1;
        self.sum += sample;
        self.sum_sq += sample * sample;

        if self.max_lag > 0 {
            if self.history.len() < self.max_lag {
                self.history.push(sample);
                self.history_head = self.history.len() % self.max_lag;
            } else {
                self.history[history_head] = sample;
                self.history_head = (history_head + 1) % self.max_lag;
            }
        }
    }
}

impl Merge for ScalarAutocorrelation {
    fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.max_lag, other.max_lag,
            "cannot merge autocorrelation accumulators with different max_lag values"
        );
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let max_lag = self.max_lag;
        let mut merged = Self {
            max_lag,
            count: self.count + other.count,
            sum: self.sum + other.sum,
            sum_sq: self.sum_sq + other.sum_sq,
            pair_counts: vec![0; max_lag],
            pair_sum_left: vec![0.0; max_lag],
            pair_sum_right: vec![0.0; max_lag],
            pair_sum_products: vec![0.0; max_lag],
            history: Vec::with_capacity(max_lag),
            history_head: 0,
        };

        for idx in 0..max_lag {
            merged.pair_counts[idx] = self.pair_counts[idx] + other.pair_counts[idx];
            merged.pair_sum_left[idx] = self.pair_sum_left[idx] + other.pair_sum_left[idx];
            merged.pair_sum_right[idx] = self.pair_sum_right[idx] + other.pair_sum_right[idx];
            merged.pair_sum_products[idx] =
                self.pair_sum_products[idx] + other.pair_sum_products[idx];
        }

        merged
    }
}

/// Fixed-size scalar batch-means accumulator.
///
/// Raw moments are tracked over every sample, while batch-error estimates are tracked over completed
/// batches only. When two non-empty accumulators are merged, their incomplete active batches are not
/// combined across the merge boundary; this preserves the independent-chain interpretation used by
/// `run_parallel`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarBatchMeans {
    batch_size: usize,
    raw_count: u64,
    raw_sum: f64,
    raw_sum_sq: f64,
    completed_batches: ScalarMoments,
    current_batch_count: usize,
    current_batch_sum: f64,
}

impl ScalarBatchMeans {
    /// Create an empty accumulator with `batch_size` samples per completed batch.
    pub fn new(batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(RmcError::InvalidArgument(
                "batch_size must be > 0".to_string(),
            ));
        }

        Ok(Self {
            batch_size,
            raw_count: 0,
            raw_sum: 0.0,
            raw_sum_sq: 0.0,
            completed_batches: ScalarMoments::new(),
            current_batch_count: 0,
            current_batch_sum: 0.0,
        })
    }

    /// Build a batch-means accumulator from samples.
    pub fn from_samples(batch_size: usize, samples: impl IntoIterator<Item = f64>) -> Result<Self> {
        let mut acc = Self::new(batch_size)?;
        for sample in samples {
            acc.accumulate(sample);
        }
        Ok(acc)
    }

    /// Number of samples per completed batch.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Number of completed batches.
    pub fn completed_batch_count(&self) -> u64 {
        self.completed_batches.count()
    }

    /// Number of samples currently waiting in the active incomplete batch.
    pub fn partial_batch_len(&self) -> usize {
        self.current_batch_count
    }

    /// Sum of all accumulated raw samples.
    pub fn sum(&self) -> f64 {
        self.raw_sum
    }

    /// Sum of squared raw samples.
    pub fn sum_squares(&self) -> f64 {
        self.raw_sum_sq
    }

    /// Mean of the completed batch means.
    pub fn mean_of_completed_batches(&self) -> Option<f64> {
        self.completed_batches.mean()
    }

    /// Unbiased sample variance of the completed batch means.
    pub fn completed_batch_sample_variance(&self) -> Option<f64> {
        self.completed_batches.sample_variance()
    }

    /// Standard error of the raw-sample mean estimated from completed batch means.
    pub fn batch_standard_error(&self) -> Option<f64> {
        let batch_count = self.completed_batch_count();
        if batch_count > 1 {
            self.completed_batches
                .sample_variance()
                .map(|variance| (variance / batch_count as f64).sqrt())
        } else {
            None
        }
    }
}

impl Accumulator<f64> for ScalarBatchMeans {
    fn count(&self) -> u64 {
        self.raw_count
    }

    fn accumulate(&mut self, sample: f64) {
        self.raw_count += 1;
        self.raw_sum += sample;
        self.raw_sum_sq += sample * sample;
        self.current_batch_count += 1;
        self.current_batch_sum += sample;

        if self.current_batch_count == self.batch_size {
            self.completed_batches
                .accumulate(self.current_batch_sum / self.batch_size as f64);
            self.current_batch_count = 0;
            self.current_batch_sum = 0.0;
        }
    }
}

impl MeanAccumulator<f64> for ScalarBatchMeans {
    fn mean(&self) -> Option<f64> {
        (self.raw_count > 0).then_some(self.raw_sum / self.raw_count as f64)
    }
}

impl VarianceAccumulator<f64> for ScalarBatchMeans {
    fn population_variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        Some(self.raw_sum_sq / self.raw_count as f64 - mean * mean)
    }

    fn sample_variance(&self) -> Option<f64> {
        if self.raw_count > 1 {
            let mean = self.mean().expect("raw_count > 1 implies a mean");
            let sum_squared_deviations = self.raw_sum_sq - self.raw_count as f64 * mean * mean;
            Some(sum_squared_deviations / (self.raw_count - 1) as f64)
        } else {
            None
        }
    }
}

impl Merge for ScalarBatchMeans {
    fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.batch_size, other.batch_size,
            "cannot merge batch-means accumulators with different batch_size values"
        );
        if self.raw_count == 0 {
            return other;
        }
        if other.raw_count == 0 {
            return self;
        }

        Self {
            batch_size: self.batch_size,
            raw_count: self.raw_count + other.raw_count,
            raw_sum: self.raw_sum + other.raw_sum,
            raw_sum_sq: self.raw_sum_sq + other.raw_sum_sq,
            completed_batches: self.completed_batches.merge(other.completed_batches),
            current_batch_count: 0,
            current_batch_sum: 0.0,
        }
    }
}

/// Fixed-size scalar block-means accumulator retaining completed block estimates.
///
/// This sits one level above [`ScalarBatchMeans`]: raw sample moments are tracked the same way, but
/// every completed block mean is stored so downstream resampling, especially jackknife over block
/// means, can be computed without replaying the original samples. Merging preserves only completed
/// block means and drops active partial blocks at the chain boundary.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarBlockMeans {
    block_size: usize,
    raw_count: u64,
    raw_sum: f64,
    raw_sum_sq: f64,
    completed_block_means: Vec<f64>,
    current_block_count: usize,
    current_block_sum: f64,
}

impl ScalarBlockMeans {
    /// Create an empty accumulator with `block_size` samples per completed block.
    pub fn new(block_size: usize) -> Result<Self> {
        if block_size == 0 {
            return Err(RmcError::InvalidArgument(
                "block_size must be > 0".to_string(),
            ));
        }

        Ok(Self {
            block_size,
            raw_count: 0,
            raw_sum: 0.0,
            raw_sum_sq: 0.0,
            completed_block_means: Vec::new(),
            current_block_count: 0,
            current_block_sum: 0.0,
        })
    }

    /// Build a block-means accumulator from samples.
    pub fn from_samples(block_size: usize, samples: impl IntoIterator<Item = f64>) -> Result<Self> {
        let mut acc = Self::new(block_size)?;
        for sample in samples {
            acc.accumulate(sample);
        }
        Ok(acc)
    }

    /// Number of samples per completed block.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of completed blocks.
    pub fn completed_block_count(&self) -> usize {
        self.completed_block_means.len()
    }

    /// Number of samples currently waiting in the active incomplete block.
    pub fn partial_block_len(&self) -> usize {
        self.current_block_count
    }

    /// Borrow completed block means.
    pub fn completed_block_means(&self) -> &[f64] {
        &self.completed_block_means
    }

    /// Sum of all accumulated raw samples.
    pub fn sum(&self) -> f64 {
        self.raw_sum
    }

    /// Sum of squared raw samples.
    pub fn sum_squares(&self) -> f64 {
        self.raw_sum_sq
    }

    /// Mean of completed block means.
    pub fn mean_of_completed_blocks(&self) -> Option<f64> {
        (!self.completed_block_means.is_empty()).then(|| {
            self.completed_block_means.iter().sum::<f64>() / self.completed_block_means.len() as f64
        })
    }

    /// Unbiased sample variance of completed block means.
    pub fn completed_block_sample_variance(&self) -> Option<f64> {
        block_mean_moments(&self.completed_block_means).sample_variance()
    }

    /// Standard error of the raw-sample mean estimated from completed block means.
    pub fn block_standard_error(&self) -> Option<f64> {
        let block_count = self.completed_block_means.len();
        if block_count > 1 {
            self.completed_block_sample_variance()
                .map(|variance| (variance / block_count as f64).sqrt())
        } else {
            None
        }
    }

    /// Jackknife accumulator over completed block means.
    pub fn jackknife(&self) -> ScalarJackknife {
        ScalarJackknife::from_values(self.completed_block_means.iter().copied())
    }
}

impl Accumulator<f64> for ScalarBlockMeans {
    fn count(&self) -> u64 {
        self.raw_count
    }

    fn accumulate(&mut self, sample: f64) {
        self.raw_count += 1;
        self.raw_sum += sample;
        self.raw_sum_sq += sample * sample;
        self.current_block_count += 1;
        self.current_block_sum += sample;

        if self.current_block_count == self.block_size {
            self.completed_block_means
                .push(self.current_block_sum / self.block_size as f64);
            self.current_block_count = 0;
            self.current_block_sum = 0.0;
        }
    }
}

impl MeanAccumulator<f64> for ScalarBlockMeans {
    fn mean(&self) -> Option<f64> {
        (self.raw_count > 0).then_some(self.raw_sum / self.raw_count as f64)
    }
}

impl VarianceAccumulator<f64> for ScalarBlockMeans {
    fn population_variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        Some(self.raw_sum_sq / self.raw_count as f64 - mean * mean)
    }

    fn sample_variance(&self) -> Option<f64> {
        if self.raw_count > 1 {
            let mean = self.mean().expect("raw_count > 1 implies a mean");
            let sum_squared_deviations = self.raw_sum_sq - self.raw_count as f64 * mean * mean;
            Some(sum_squared_deviations / (self.raw_count - 1) as f64)
        } else {
            None
        }
    }
}

impl Merge for ScalarBlockMeans {
    fn merge(mut self, other: Self) -> Self {
        assert_eq!(
            self.block_size, other.block_size,
            "cannot merge block-means accumulators with different block_size values"
        );
        if self.raw_count == 0 {
            return other;
        }
        if other.raw_count == 0 {
            return self;
        }

        self.raw_count += other.raw_count;
        self.raw_sum += other.raw_sum;
        self.raw_sum_sq += other.raw_sum_sq;
        self.completed_block_means
            .extend(other.completed_block_means);
        self.current_block_count = 0;
        self.current_block_sum = 0.0;
        self
    }
}

/// Scalar jackknife accumulator over independent block estimates.
///
/// Values are stored because delete-one jackknife estimates require access to each block estimate.
/// This is intended for a moderate number of batch/block summaries, not for every hot-path MC
/// sample.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScalarJackknife {
    values: Vec<f64>,
}

impl ScalarJackknife {
    /// Create an empty jackknife accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a jackknife accumulator from independent block estimates.
    pub fn from_values(values: impl IntoIterator<Item = f64>) -> Self {
        let mut acc = Self::new();
        for value in values {
            acc.accumulate(value);
        }
        acc
    }

    /// Borrow the stored block estimates.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mean estimate over all stored block estimates.
    pub fn estimate(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }

        Some(self.values.iter().sum::<f64>() / self.values.len() as f64)
    }

    /// Delete-one jackknife estimates of the mean.
    pub fn delete_one_estimates(&self) -> Option<Vec<f64>> {
        let count = self.values.len();
        if count < 2 {
            return None;
        }

        let sum = self.values.iter().sum::<f64>();
        Some(
            self.values
                .iter()
                .map(|value| (sum - value) / (count - 1) as f64)
                .collect(),
        )
    }

    /// Bias estimate for the mean estimator.
    pub fn bias(&self) -> Option<f64> {
        let estimate = self.estimate()?;
        let delete_one = self.delete_one_estimates()?;
        let delete_one_mean = delete_one.iter().sum::<f64>() / delete_one.len() as f64;
        Some((self.values.len() as f64 - 1.0) * (delete_one_mean - estimate))
    }

    /// Bias-corrected mean estimate.
    pub fn bias_corrected_estimate(&self) -> Option<f64> {
        Some(self.estimate()? - self.bias()?)
    }

    /// Jackknife standard error of the mean estimate.
    pub fn standard_error(&self) -> Option<f64> {
        let delete_one = self.delete_one_estimates()?;
        let delete_one_mean = delete_one.iter().sum::<f64>() / delete_one.len() as f64;
        let sum_squared_deviations = delete_one
            .iter()
            .map(|estimate| {
                let delta = estimate - delete_one_mean;
                delta * delta
            })
            .sum::<f64>();
        let n = delete_one.len() as f64;
        Some(((n - 1.0) / n * sum_squared_deviations).sqrt())
    }
}

impl Accumulator<f64> for ScalarJackknife {
    fn count(&self) -> u64 {
        self.values.len() as u64
    }

    fn accumulate(&mut self, sample: f64) {
        self.values.push(sample);
    }
}

impl MeanAccumulator<f64> for ScalarJackknife {
    fn mean(&self) -> Option<f64> {
        self.estimate()
    }
}

impl Merge for ScalarJackknife {
    fn merge(mut self, other: Self) -> Self {
        self.values.extend(other.values);
        self
    }
}

fn block_mean_moments(values: &[f64]) -> ScalarMoments {
    ScalarMoments::from_samples(values.iter().copied())
}

/// Online per-component moments for vector-valued `f64` samples.
///
/// `VectorMoments` tracks the mean and per-component squared deviations for fixed-dimension samples.
/// It does not store cross-covariances; use [`VectorCovariance`] when a full covariance matrix is
/// needed.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct VectorMoments {
    count: u64,
    mean: DVector<f64>,
    m2: DVector<f64>,
}

impl VectorMoments {
    /// Create an empty vector-moments accumulator.
    pub fn new(dimension: usize) -> Result<Self> {
        validate_dimension(dimension)?;
        Ok(Self {
            count: 0,
            mean: DVector::zeros(dimension),
            m2: DVector::zeros(dimension),
        })
    }

    /// Build an accumulator from fixed-dimension samples.
    pub fn from_samples(
        dimension: usize,
        samples: impl IntoIterator<Item = DVector<f64>>,
    ) -> Result<Self> {
        let mut acc = Self::new(dimension)?;
        for sample in samples {
            acc.try_accumulate(sample)?;
        }
        Ok(acc)
    }

    /// Sample dimension.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Add one sample, returning an error if its dimension differs.
    pub fn try_accumulate(&mut self, sample: DVector<f64>) -> Result<()> {
        self.validate_sample_dimension(&sample)?;
        self.accumulate_validated(sample);
        Ok(())
    }

    /// Sum reconstructed from stored count and mean.
    pub fn sum(&self) -> DVector<f64> {
        &self.mean * self.count as f64
    }

    /// Per-component sums of squared deviations from the current mean.
    pub fn sum_squared_deviations(&self) -> DVector<f64> {
        self.m2.clone()
    }

    /// Per-component standard errors of the mean, using unbiased sample variances.
    pub fn standard_error(&self) -> Option<DVector<f64>> {
        self.sample_variance()
            .map(|variance| variance.map(|value| (value / self.count as f64).sqrt()))
    }

    fn validate_sample_dimension(&self, sample: &DVector<f64>) -> Result<()> {
        if sample.len() != self.dimension() {
            return Err(RmcError::InvalidArgument(format!(
                "sample dimension {} does not match accumulator dimension {}",
                sample.len(),
                self.dimension()
            )));
        }
        Ok(())
    }

    fn accumulate_validated(&mut self, sample: DVector<f64>) {
        self.count += 1;
        let count = self.count as f64;
        let delta = &sample - &self.mean;
        self.mean += &delta / count;
        let delta_after = sample - &self.mean;
        self.m2 += delta.component_mul(&delta_after);
    }
}

impl Accumulator<DVector<f64>> for VectorMoments {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, sample: DVector<f64>) {
        self.try_accumulate(sample)
            .expect("sample dimension must match accumulator dimension");
    }
}

impl MeanAccumulator<DVector<f64>> for VectorMoments {
    fn mean(&self) -> Option<DVector<f64>> {
        (self.count > 0).then(|| self.mean.clone())
    }
}

impl VarianceAccumulator<DVector<f64>> for VectorMoments {
    fn population_variance(&self) -> Option<DVector<f64>> {
        (self.count > 0).then(|| &self.m2 / self.count as f64)
    }

    fn sample_variance(&self) -> Option<DVector<f64>> {
        (self.count > 1).then(|| &self.m2 / (self.count - 1) as f64)
    }
}

impl Merge for VectorMoments {
    fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "cannot merge vector moments with different dimensions"
        );
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let combined_count = self.count + other.count;
        let self_count = self.count as f64;
        let other_count = other.count as f64;
        let combined_count_f64 = combined_count as f64;
        let delta = &other.mean - &self.mean;

        Self {
            count: combined_count,
            mean: self.mean + &delta * (other_count / combined_count_f64),
            m2: self.m2
                + other.m2
                + delta.component_mul(&delta) * (self_count * other_count / combined_count_f64),
        }
    }
}

/// Online covariance accumulator for fixed-dimension vector samples.
///
/// The accumulator stores a full sum-of-cross-deviations matrix and supports exact deterministic
/// merging of independent partial accumulators through the Chan-Golub-LeVeque formula.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct VectorCovariance {
    count: u64,
    mean: DVector<f64>,
    m2: DMatrix<f64>,
}

impl VectorCovariance {
    /// Create an empty vector covariance accumulator.
    pub fn new(dimension: usize) -> Result<Self> {
        validate_dimension(dimension)?;
        Ok(Self {
            count: 0,
            mean: DVector::zeros(dimension),
            m2: DMatrix::zeros(dimension, dimension),
        })
    }

    /// Build an accumulator from fixed-dimension samples.
    pub fn from_samples(
        dimension: usize,
        samples: impl IntoIterator<Item = DVector<f64>>,
    ) -> Result<Self> {
        let mut acc = Self::new(dimension)?;
        for sample in samples {
            acc.try_accumulate(sample)?;
        }
        Ok(acc)
    }

    /// Sample dimension.
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// Add one sample, returning an error if its dimension differs.
    pub fn try_accumulate(&mut self, sample: DVector<f64>) -> Result<()> {
        self.validate_sample_dimension(&sample)?;
        self.accumulate_validated(sample);
        Ok(())
    }

    /// Sum reconstructed from stored count and mean.
    pub fn sum(&self) -> DVector<f64> {
        &self.mean * self.count as f64
    }

    /// Sum of cross deviations from the current mean vector.
    pub fn sum_cross_deviations(&self) -> DMatrix<f64> {
        self.m2.clone()
    }

    /// Population covariance matrix, normalized by `n`.
    pub fn population_covariance(&self) -> Option<DMatrix<f64>> {
        (self.count > 0).then(|| &self.m2 / self.count as f64)
    }

    /// Unbiased sample covariance matrix, normalized by `n - 1`.
    pub fn sample_covariance(&self) -> Option<DMatrix<f64>> {
        (self.count > 1).then(|| &self.m2 / (self.count - 1) as f64)
    }

    /// Population variance vector, equal to the covariance diagonal normalized by `n`.
    pub fn population_variance_vector(&self) -> Option<DVector<f64>> {
        self.population_covariance()
            .map(|covariance| covariance.diagonal())
    }

    /// Unbiased sample variance vector, equal to the covariance diagonal normalized by `n - 1`.
    pub fn sample_variance_vector(&self) -> Option<DVector<f64>> {
        self.sample_covariance()
            .map(|covariance| covariance.diagonal())
    }

    fn validate_sample_dimension(&self, sample: &DVector<f64>) -> Result<()> {
        if sample.len() != self.dimension() {
            return Err(RmcError::InvalidArgument(format!(
                "sample dimension {} does not match accumulator dimension {}",
                sample.len(),
                self.dimension()
            )));
        }
        Ok(())
    }

    fn accumulate_validated(&mut self, sample: DVector<f64>) {
        self.count += 1;
        let count = self.count as f64;
        let delta = &sample - &self.mean;
        self.mean += &delta / count;
        let delta_after = sample - &self.mean;
        self.m2 += &delta * delta_after.transpose();
    }
}

impl Accumulator<DVector<f64>> for VectorCovariance {
    fn count(&self) -> u64 {
        self.count
    }

    fn accumulate(&mut self, sample: DVector<f64>) {
        self.try_accumulate(sample)
            .expect("sample dimension must match accumulator dimension");
    }
}

impl MeanAccumulator<DVector<f64>> for VectorCovariance {
    fn mean(&self) -> Option<DVector<f64>> {
        (self.count > 0).then(|| self.mean.clone())
    }
}

impl VarianceAccumulator<DVector<f64>> for VectorCovariance {
    fn population_variance(&self) -> Option<DVector<f64>> {
        self.population_variance_vector()
    }

    fn sample_variance(&self) -> Option<DVector<f64>> {
        self.sample_variance_vector()
    }
}

impl Merge for VectorCovariance {
    fn merge(self, other: Self) -> Self {
        assert_eq!(
            self.dimension(),
            other.dimension(),
            "cannot merge vector covariance accumulators with different dimensions"
        );
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }

        let combined_count = self.count + other.count;
        let self_count = self.count as f64;
        let other_count = other.count as f64;
        let combined_count_f64 = combined_count as f64;
        let delta = &other.mean - &self.mean;
        let merge_weight = self_count * other_count / combined_count_f64;

        Self {
            count: combined_count,
            mean: self.mean + &delta * (other_count / combined_count_f64),
            m2: self.m2 + other.m2 + (&delta * delta.transpose()) * merge_weight,
        }
    }
}

fn validate_dimension(dimension: usize) -> Result<()> {
    if dimension == 0 {
        return Err(RmcError::InvalidArgument(
            "dimension must be > 0".to_string(),
        ));
    }
    Ok(())
}

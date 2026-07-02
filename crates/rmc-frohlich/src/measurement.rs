use rmc_core::mc::Measurement;
use rmc_core::Merge;
use rmc_grids::{Grid1d, LinearGrid};

use crate::diagram::{norm0, Diagram};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BatchedSum {
    n_batches: usize,
    batch_sums: Vec<f64>,
    batch_counts: Vec<u64>,
    next: usize,
}

impl BatchedSum {
    pub fn new(n_batches: usize) -> Self {
        assert!(n_batches >= 2, "jackknife needs at least two batches");
        Self {
            n_batches,
            batch_sums: vec![0.0; n_batches],
            batch_counts: vec![0; n_batches],
            next: 0,
        }
    }

    pub fn push(&mut self, value: f64) {
        let batch = self.next % self.n_batches;
        self.batch_sums[batch] += value;
        self.batch_counts[batch] += 1;
        self.next += 1;
    }

    pub fn reset(&mut self) {
        self.batch_sums.fill(0.0);
        self.batch_counts.fill(0);
        self.next = 0;
    }

    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    pub fn total_count(&self) -> u64 {
        self.batch_counts.iter().sum()
    }

    pub fn total_sum(&self) -> f64 {
        self.batch_sums.iter().sum()
    }

    pub fn mean(&self) -> Option<f64> {
        let count = self.total_count();
        (count > 0).then_some(self.total_sum() / count as f64)
    }

    fn batch_mean(&self, batch: usize) -> Option<f64> {
        let count = self.batch_counts[batch];
        (count > 0).then_some(self.batch_sums[batch] / count as f64)
    }
}

impl Merge for BatchedSum {
    fn merge(self, other: Self) -> Self {
        assert_eq!(self.n_batches, other.n_batches);
        let batch_sums = self
            .batch_sums
            .into_iter()
            .zip(other.batch_sums)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        let batch_counts = self
            .batch_counts
            .into_iter()
            .zip(other.batch_counts)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        Self {
            n_batches: self.n_batches,
            batch_sums,
            batch_counts,
            next: self.next + other.next,
        }
    }
}

/// A set of `num_bins` batched-sum series that all share ONE global sample schedule.
///
/// Every measured sample advances the schedule exactly once and contributes to at most one
/// bin (the "active" bin). The per-batch sample counts are therefore *global* — a sample that
/// lands in bin `i` still counts toward the batch denominator of every other bin. This is
/// exactly the normalization the ratio estimators need (`Σ = exact / zeroth`), and it makes
/// `push` `O(1)` instead of `O(num_bins)`: the previous representation kept one independent
/// `BatchedSum` per bin and pushed `0.0` into all inactive bins on every sample, which cost
/// `~num_bins` operations per cycle and dominated the entire run time.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BinnedBatchedSums {
    n_batches: usize,
    num_bins: usize,
    /// bin-major: `batch_sums[bin * n_batches + batch]`.
    batch_sums: Vec<f64>,
    /// shared across all bins (the global per-batch sample count).
    batch_counts: Vec<u64>,
    next: usize,
}

impl BinnedBatchedSums {
    pub fn new(num_bins: usize, n_batches: usize) -> Self {
        assert!(n_batches >= 2, "jackknife needs at least two batches");
        Self {
            n_batches,
            num_bins,
            batch_sums: vec![0.0; num_bins * n_batches],
            batch_counts: vec![0; n_batches],
            next: 0,
        }
    }

    /// Record one sample that contributes `value` to `bin`, advancing the global schedule.
    ///
    /// A sample that should contribute to no bin (e.g. a normalization-sector sample) is still
    /// recorded by calling this with its `value` equal to `0.0`; the batch count still advances
    /// so the denominators stay aligned with the scalar accumulators.
    #[inline]
    pub fn push(&mut self, bin: usize, value: f64) {
        let batch = self.next % self.n_batches;
        self.batch_sums[bin * self.n_batches + batch] += value;
        self.batch_counts[batch] += 1;
        self.next += 1;
    }

    pub fn n_batches(&self) -> usize {
        self.n_batches
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Total number of samples recorded across all batches (the global sample count).
    pub fn total_count(&self) -> u64 {
        self.batch_counts.iter().sum()
    }

    fn batch_mean(&self, bin: usize, batch: usize) -> Option<f64> {
        let count = self.batch_counts[batch];
        (count > 0).then_some(self.batch_sums[bin * self.n_batches + batch] / count as f64)
    }

    fn bin_mean(&self, bin: usize) -> Option<f64> {
        let total = self.total_count();
        if total == 0 {
            return None;
        }
        let base = bin * self.n_batches;
        let sum: f64 = self.batch_sums[base..base + self.n_batches].iter().sum();
        Some(sum / total as f64)
    }
}

impl Merge for BinnedBatchedSums {
    fn merge(self, other: Self) -> Self {
        assert_eq!(self.n_batches, other.n_batches);
        assert_eq!(self.num_bins, other.num_bins);
        let batch_sums = self
            .batch_sums
            .into_iter()
            .zip(other.batch_sums)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        let batch_counts = self
            .batch_counts
            .into_iter()
            .zip(other.batch_counts)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect();
        Self {
            n_batches: self.n_batches,
            num_bins: self.num_bins,
            batch_sums,
            batch_counts,
            next: self.next + other.next,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GridSpec {
    pub first: f64,
    pub last: f64,
    pub len: usize,
}

impl GridSpec {
    pub fn new(first: f64, last: f64, len: usize) -> Self {
        Self { first, last, len }
    }

    pub fn grid(self) -> LinearGrid {
        LinearGrid::new(self.first, self.last, self.len).expect("measurement grid must be valid")
    }

    pub fn bin_count(self) -> usize {
        self.len - 1
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Estimate {
    pub mean: f64,
    pub stderr: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct SeriesEstimate {
    pub tau: Vec<f64>,
    pub mean: Vec<f64>,
    pub stderr: Vec<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PolaronStats {
    pub zeroth: BatchedSum,
    pub exact: BinnedBatchedSums,
    pub hist: BinnedBatchedSums,
    pub energy: BatchedSum,
    pub a: BatchedSum,
    pub order: BatchedSum,
    pub grid: GridSpec,
    pub energy_estimate: f64,
    pub energy_estimates: Vec<f64>,
    pub self_consistent_count: usize,
    pub self_consistent_period: usize,
    pub self_consistent_periods: Vec<usize>,
    pub period_multiplier: f64,
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub max_tau: f64,
    pub sample_count: usize,
}

impl PolaronStats {
    pub fn jackknife_selfenergy(&self) -> SeriesEstimate {
        let grid = self.grid.grid();
        let dispersion = self.dispersion();
        let n0 = norm0(self.max_tau, dispersion);
        let binsize = grid.step().abs();
        let n_batches = self.zeroth.n_batches();
        let mut mean = Vec::with_capacity(self.exact.num_bins());
        let mut stderr = Vec::with_capacity(self.exact.num_bins());
        let mut batch_num = Vec::with_capacity(n_batches);
        let mut batch_den = Vec::with_capacity(n_batches);
        for bin in 0..self.exact.num_bins() {
            batch_num.clear();
            batch_den.clear();
            // `exact` (binned) and `zeroth` (scalar) share the same global sample schedule and
            // per-batch counts, so a batch is non-empty for both or neither.
            for batch in 0..n_batches {
                if let (Some(num_mean), Some(den_mean)) =
                    (self.exact.batch_mean(bin, batch), self.zeroth.batch_mean(batch))
                {
                    batch_num.push(num_mean);
                    batch_den.push(den_mean);
                }
            }
            let estimate = jackknife_from_batch_means(&batch_num, &batch_den, |num, den| {
                if den == 0.0 {
                    f64::NAN
                } else {
                    num * n0 / (den * binsize)
                }
            });
            mean.push(estimate.mean);
            stderr.push(estimate.stderr);
        }
        SeriesEstimate {
            tau: grid.bin_centers().collect(),
            mean,
            stderr,
        }
    }

    pub fn jackknife_energy(&self) -> Estimate {
        let n0 = norm0(self.max_tau, self.dispersion());
        jackknife_ratio(&self.energy, &self.zeroth, |energy, zeroth| {
            if zeroth == 0.0 {
                f64::NAN
            } else {
                energy * n0 / zeroth
            }
        })
    }

    pub fn jackknife_quasiparticle_weight(&self) -> Estimate {
        let n0 = norm0(self.max_tau, self.dispersion());
        jackknife_ratio(&self.a, &self.zeroth, |a, zeroth| {
            if zeroth == 0.0 {
                f64::NAN
            } else {
                1.0 / (1.0 + a * n0 / zeroth)
            }
        })
    }

    pub fn get_exact(&self) -> Vec<f64> {
        let grid = self.grid.grid();
        let n0 = norm0(self.max_tau, self.dispersion());
        let zeroth = self.zeroth.mean().unwrap_or(f64::NAN);
        (0..self.exact.num_bins())
            .map(|idx| {
                let binsize = grid.bin_width(idx).expect("bin must exist");
                self.exact.bin_mean(idx).unwrap_or(0.0) / binsize * n0 / zeroth
            })
            .collect()
    }

    fn dispersion(&self) -> f64 {
        self.momentum * self.momentum / (2.0 * Diagram::MASS) - self.mu
    }
}

impl Merge for PolaronStats {
    fn merge(self, other: Self) -> Self {
        assert_eq!(self.grid, other.grid);
        Self {
            zeroth: self.zeroth.merge(other.zeroth),
            exact: self.exact.merge(other.exact),
            hist: self.hist.merge(other.hist),
            energy: self.energy.merge(other.energy),
            a: self.a.merge(other.a),
            order: self.order.merge(other.order),
            grid: self.grid,
            energy_estimate: other.energy_estimate,
            energy_estimates: [self.energy_estimates, other.energy_estimates].concat(),
            self_consistent_count: self.self_consistent_count + other.self_consistent_count,
            self_consistent_period: other.self_consistent_period,
            self_consistent_periods: [self.self_consistent_periods, other.self_consistent_periods]
                .concat(),
            period_multiplier: self.period_multiplier,
            alpha: self.alpha,
            mu: self.mu,
            momentum: self.momentum,
            max_tau: self.max_tau,
            sample_count: self.sample_count + other.sample_count,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PolaronMeasurement {
    stats: PolaronStats,
}

impl PolaronMeasurement {
    pub fn new(
        num_bins: usize,
        max_tau: f64,
        n_batches: usize,
        energy_estimate: f64,
        self_consistent_period: usize,
        period_multiplier: f64,
        template: &Diagram,
    ) -> Self {
        let grid = GridSpec::new(0.0, max_tau, num_bins + 1);
        Self {
            stats: PolaronStats {
                zeroth: BatchedSum::new(n_batches),
                exact: BinnedBatchedSums::new(num_bins, n_batches),
                hist: BinnedBatchedSums::new(num_bins, n_batches),
                energy: BatchedSum::new(n_batches),
                a: BatchedSum::new(n_batches),
                order: BatchedSum::new(n_batches),
                grid,
                energy_estimate,
                energy_estimates: vec![energy_estimate],
                self_consistent_count: 0,
                self_consistent_period,
                self_consistent_periods: vec![self_consistent_period],
                period_multiplier,
                alpha: template.alpha,
                mu: template.mu,
                momentum: template.momentum,
                max_tau: template.max_tau,
                sample_count: 0,
            },
        }
    }

    pub fn stats(&self) -> &PolaronStats {
        &self.stats
    }

    fn reevaluate_energy_estimate(&mut self, d: &Diagram) {
        let estimate = self.stats.jackknife_energy();
        if !estimate.mean.is_finite() {
            return;
        }
        let new_estimate = Diagram::bare_dispersion(&d.momentum_out()) + estimate.mean;
        self.stats.energy_estimate = new_estimate;
        self.stats.energy_estimates.push(new_estimate);
        self.stats.energy.reset();
        self.stats.a.reset();
        self.stats.self_consistent_count = 0;
        self.stats.self_consistent_period =
            ((self.stats.self_consistent_period as f64) * self.stats.period_multiplier) as usize;
        self.stats
            .self_consistent_periods
            .push(self.stats.self_consistent_period);
    }
}

impl Measurement<Diagram> for PolaronMeasurement {
    type Output = PolaronStats;

    fn measure(&mut self, d: &Diagram) {
        let grid = self.stats.grid.grid();
        let Some(index) = grid.bin_index(d.tau()) else {
            return;
        };

        let is_zeroth = d.order == 0;
        let t0 = grid.bin_center(index).expect("bin center must exist");
        let exact_value = if is_zeroth {
            0.0
        } else {
            d.exact_estimator(t0)
        };
        let exp_energy = if is_zeroth {
            0.0
        } else {
            ((self.stats.energy_estimate - d.mu) * d.tau()).exp()
        };

        self.stats.zeroth.push(if is_zeroth { 1.0 } else { 0.0 });
        self.stats
            .energy
            .push(if is_zeroth { 0.0 } else { -exp_energy });
        self.stats
            .a
            .push(if is_zeroth { 0.0 } else { d.tau() * exp_energy });
        self.stats.order.push(d.order as f64);

        // Only the active bin receives a contribution; the shared batch schedule handles the
        // normalization for every other bin. At order 0 the contribution is `0.0` (the sample
        // still advances the schedule so denominators stay aligned with `zeroth`).
        let hist_value = if is_zeroth { 0.0 } else { 1.0 };
        self.stats.exact.push(index, exact_value);
        self.stats.hist.push(index, hist_value);

        self.stats.sample_count += 1;
        self.stats.self_consistent_count += 1;
        if self.stats.self_consistent_count > self.stats.self_consistent_period {
            self.reevaluate_energy_estimate(d);
        }
    }

    fn finish(self) -> Self::Output {
        self.stats
    }
}

pub fn jackknife_ratio<F>(num: &BatchedSum, den: &BatchedSum, f: F) -> Estimate
where
    F: Fn(f64, f64) -> f64,
{
    assert_eq!(num.n_batches(), den.n_batches());
    let mut batch_num = Vec::new();
    let mut batch_den = Vec::new();
    for batch in 0..num.n_batches() {
        if let (Some(num_mean), Some(den_mean)) = (num.batch_mean(batch), den.batch_mean(batch)) {
            batch_num.push(num_mean);
            batch_den.push(den_mean);
        }
    }
    jackknife_from_batch_means(&batch_num, &batch_den, f)
}

/// Delete-one jackknife of a ratio functional `f(num, den)` over per-batch means.
///
/// `batch_num[k]` / `batch_den[k]` are the numerator/denominator means of the (non-empty)
/// batches, paired on the same batch schedule.
fn jackknife_from_batch_means<F>(batch_num: &[f64], batch_den: &[f64], f: F) -> Estimate
where
    F: Fn(f64, f64) -> f64,
{
    debug_assert_eq!(batch_num.len(), batch_den.len());
    let n = batch_num.len();
    if n < 2 {
        return Estimate {
            mean: f64::NAN,
            stderr: f64::NAN,
        };
    }

    let total_num: f64 = batch_num.iter().sum();
    let total_den: f64 = batch_den.iter().sum();
    let mut theta = Vec::with_capacity(n);
    for batch in 0..n {
        let loo_num = (total_num - batch_num[batch]) / (n - 1) as f64;
        let loo_den = (total_den - batch_den[batch]) / (n - 1) as f64;
        theta.push(f(loo_num, loo_den));
    }

    let finite = theta
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let n = finite.len();
    if n < 2 {
        return Estimate {
            mean: f64::NAN,
            stderr: f64::NAN,
        };
    }

    let mean = finite.iter().sum::<f64>() / n as f64;
    let variance_sum = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    Estimate {
        mean,
        stderr: (((n - 1) as f64 / n as f64) * variance_sum).sqrt(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `O(1)` `BinnedBatchedSums` must be numerically identical to the old representation:
    /// one independent `BatchedSum` per bin, with `0.0` pushed into every inactive bin on every
    /// sample. This locks in that the optimization is a pure speedup, not a change in results.
    #[test]
    fn binned_matches_per_bin_batched_sum() {
        let n_batches = 4;
        let num_bins = 3;
        let mut binned = BinnedBatchedSums::new(num_bins, n_batches);
        let mut per_bin: Vec<BatchedSum> =
            (0..num_bins).map(|_| BatchedSum::new(n_batches)).collect();

        // (active bin, contributed value); includes an order-0-style sample (value 0.0).
        let samples = [
            (0usize, 1.5),
            (2, 3.0),
            (0, -0.5),
            (1, 2.0),
            (2, 0.0),
            (0, 4.0),
            (1, -1.0),
            (2, 0.25),
            (0, 0.0),
            (1, 7.0),
        ];
        for &(active, value) in &samples {
            binned.push(active, value);
            for (bin, acc) in per_bin.iter_mut().enumerate() {
                acc.push(if bin == active { value } else { 0.0 });
            }
        }

        assert_eq!(binned.total_count(), samples.len() as u64);
        for bin in 0..num_bins {
            for batch in 0..n_batches {
                assert_eq!(
                    binned.batch_mean(bin, batch),
                    per_bin[bin].batch_mean(batch),
                    "batch mean mismatch at bin {bin}, batch {batch}"
                );
            }
            assert_eq!(binned.bin_mean(bin), per_bin[bin].mean(), "bin mean at {bin}");
        }
    }

    /// A constant ratio series jackknifed per bin recovers the constant with zero error, exactly
    /// as the scalar `jackknife_ratio` does — exercising the shared numerator/denominator path.
    #[test]
    fn binned_selfenergy_ratio_is_constant() {
        let n_batches = 8;
        let num_bins = 2;
        let mut exact = BinnedBatchedSums::new(num_bins, n_batches);
        let mut zeroth = BatchedSum::new(n_batches);
        // Every sample lands in bin 0 with exact=6, and zeroth=3 → ratio 2.0.
        for _ in 0..80 {
            exact.push(0, 6.0);
            zeroth.push(3.0);
        }
        let mut batch_num = Vec::new();
        let mut batch_den = Vec::new();
        for batch in 0..n_batches {
            if let (Some(nm), Some(dm)) = (exact.batch_mean(0, batch), zeroth.batch_mean(batch)) {
                batch_num.push(nm);
                batch_den.push(dm);
            }
        }
        let estimate = jackknife_from_batch_means(&batch_num, &batch_den, |num, den| num / den);
        assert!((estimate.mean - 2.0).abs() < 1.0e-12);
        assert!(estimate.stderr < 1.0e-12);
    }
}

use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

use crate::{Result, RmcError};

use super::sink::{ResultSink, ScopedResultSink, SinkMeasurement};
use super::traits::{StepOutcome, SteppingUpdateSet, Update, UpdateSet, UpdateStats};

pub struct SinkMeasurementSet<State> {
    entries: Vec<Box<dyn SinkMeasurement<State>>>,
    active: Vec<usize>,
}

impl<State> Default for SinkMeasurementSet<State> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> SinkMeasurementSet<State> {
    /// Create an empty dynamic result-sink measurement set.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            active: Vec::new(),
        }
    }

    /// Add a named sink measurement.
    pub fn add<M>(&mut self, measurement: M) -> Result<()>
    where
        M: SinkMeasurement<State> + 'static,
    {
        if measurement.name().is_empty() {
            return Err(RmcError::InvalidArgument(
                "measurement name must be non-empty".to_string(),
            ));
        }
        if self.find(measurement.name()).is_some() {
            return Err(RmcError::InvalidArgument(format!(
                "measurement '{}' is already registered",
                measurement.name()
            )));
        }
        self.entries.push(Box::new(measurement));
        Ok(())
    }

    pub fn refresh_active(&mut self) {
        self.active.clear();
        self.active.reserve(self.entries.len());
        self.active.extend(0..self.entries.len());
    }

    pub fn clear_active(&mut self) {
        self.active.clear();
    }

    pub fn active_indices(&self) -> &[usize] {
        &self.active
    }

    pub fn measure_all(&mut self, state: &State) {
        for idx in &self.active {
            self.entries[*idx].measure(state);
        }
    }

    pub fn write_all(&self, sink: &mut dyn ResultSink) -> Result<()> {
        for idx in &self.active {
            let measurement = &self.entries[*idx];
            let mut scoped = ScopedResultSink::new(sink, measurement.name());
            measurement.write_result(&mut scoped)?;
        }
        Ok(())
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|measurement| measurement.name() == name)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&(dyn SinkMeasurement<State> + '_)> {
        self.find(name).map(|idx| self.entries[idx].as_ref())
    }

    pub fn get_by_name_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut (dyn SinkMeasurement<State> + '_)> {
        let idx = self.find(name)?;
        Some(self.entries[idx].as_mut())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> impl Iterator<Item = &dyn SinkMeasurement<State>> {
        self.entries.iter().map(|measurement| measurement.as_ref())
    }
}

/// Static update set containing one concrete update type.
///
/// This is the simplest monomorphized hot-path update set. The update can be stateless
/// (`Update<()>`) or stateful (`Update<State>`), depending on the kernel/run state.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SingleUpdateSet<U> {
    update: U,
    stats: [UpdateStats; 1],
}

impl<U> SingleUpdateSet<U> {
    /// Create a single-update set.
    pub fn new(update: U) -> Self {
        Self {
            update,
            stats: [UpdateStats::default()],
        }
    }

    pub fn update(&self) -> &U {
        &self.update
    }

    pub fn update_mut(&mut self) -> &mut U {
        &mut self.update
    }

    pub fn into_update(self) -> U {
        self.update
    }
}

impl<U> UpdateSet for SingleUpdateSet<U> {
    fn stats(&self) -> &[UpdateStats] {
        &self.stats
    }

    fn reset_stats(&mut self) {
        self.stats = Default::default();
    }
}

impl<State, R, U> SteppingUpdateSet<State, R> for SingleUpdateSet<U>
where
    R: Rng,
    U: Update<State>,
{
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        Ok(())
    }

    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome> {
        step_update(state, rng, 0, &mut self.update, 1.0, &mut self.stats[0])
    }
}

/// One entry in a [`WeightedUpdateSet`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightedUpdate<U> {
    update: U,
    weight: f64,
    ratio: f64,
}

impl<U> WeightedUpdate<U> {
    /// Create a weighted update with detailed-balance ratio `1.0`.
    pub fn new(update: U, weight: f64) -> Self {
        Self {
            update,
            weight,
            ratio: 1.0,
        }
    }

    /// Create a weighted update with an explicit proposal-ratio multiplier.
    pub fn with_ratio(update: U, weight: f64, ratio: f64) -> Self {
        Self {
            update,
            weight,
            ratio,
        }
    }

    pub fn update(&self) -> &U {
        &self.update
    }

    pub fn update_mut(&mut self) -> &mut U {
        &mut self.update
    }

    pub fn weight(&self) -> f64 {
        self.weight
    }

    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    pub fn into_update(self) -> U {
        self.update
    }
}

/// Static weighted update set containing any number of one concrete update type.
///
/// For heterogeneous simulations, make `U` an enum whose variants wrap the concrete update
/// implementations. That keeps storage and dispatch monomorphized while avoiding `Box<dyn Update>`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WeightedUpdateSet<U> {
    entries: Vec<WeightedUpdate<U>>,
    stats: Vec<UpdateStats>,
    #[cfg_attr(feature = "serde", serde(skip))]
    selection: Option<WeightedIndex<f64>>,
}

impl<U> WeightedUpdateSet<U> {
    /// Create a weighted update set from explicit entries.
    pub fn new(entries: Vec<WeightedUpdate<U>>) -> Result<Self> {
        let mut set = Self {
            stats: vec![UpdateStats::default(); entries.len()],
            entries,
            selection: None,
        };
        set.rebuild_distribution()?;
        Ok(set)
    }

    /// Create a two-entry inverse pair using reciprocal proposal-selection ratios.
    pub fn inverse_pair(
        first: U,
        first_weight: f64,
        second: U,
        second_weight: f64,
    ) -> Result<Self> {
        if (first_weight == 0.0) != (second_weight == 0.0) {
            return Err(RmcError::InvalidArgument(format!(
                "inverse pair must have both weights zero or both non-zero (got {first_weight} and {second_weight})"
            )));
        }
        if first_weight == 0.0 {
            return Err(RmcError::InvalidState(
                "at least one update weight must be > 0".to_string(),
            ));
        }

        Self::new(vec![
            WeightedUpdate::with_ratio(first, first_weight, second_weight / first_weight),
            WeightedUpdate::with_ratio(second, second_weight, first_weight / second_weight),
        ])
    }

    /// Rebuild the weighted selection distribution after entry weights change.
    pub fn rebuild_distribution(&mut self) -> Result<()> {
        if self.entries.is_empty() {
            return Err(RmcError::InvalidState(
                "at least one update must be registered".to_string(),
            ));
        }
        if let Some(weight) = self.entries.iter().find_map(|entry| {
            if entry.weight < 0.0 {
                Some(entry.weight)
            } else {
                None
            }
        }) {
            return Err(RmcError::InvalidArgument(format!(
                "update weights must be >= 0 (got {weight})"
            )));
        }
        if !self.entries.iter().any(|entry| entry.weight > 0.0) {
            return Err(RmcError::InvalidState(
                "at least one update weight must be > 0".to_string(),
            ));
        }

        let weights = self
            .entries
            .iter()
            .map(|entry| entry.weight)
            .collect::<Vec<_>>();
        self.selection =
            Some(WeightedIndex::new(weights).map_err(|err| {
                RmcError::InvalidArgument(format!("invalid update weights: {err}"))
            })?);
        Ok(())
    }

    pub fn entries(&self) -> &[WeightedUpdate<U>] {
        &self.entries
    }

    pub fn entries_mut(&mut self) -> &mut [WeightedUpdate<U>] {
        self.selection = None;
        &mut self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn weights(&self) -> Vec<f64> {
        self.entries.iter().map(|entry| entry.weight).collect()
    }

    pub fn ratios(&self) -> Vec<f64> {
        self.entries.iter().map(|entry| entry.ratio).collect()
    }
}

impl<U> UpdateSet for WeightedUpdateSet<U> {
    fn stats(&self) -> &[UpdateStats] {
        &self.stats
    }

    fn reset_stats(&mut self) {
        self.stats.fill(UpdateStats::default());
    }
}

impl<State, R, U> SteppingUpdateSet<State, R> for WeightedUpdateSet<U>
where
    R: Rng,
    U: Update<State>,
{
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        if self.selection.is_none() {
            self.rebuild_distribution()?;
        }
        Ok(())
    }

    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome> {
        let selection = self.selection.as_ref().ok_or_else(|| {
            RmcError::InvalidState(
                "update distribution has not been rebuilt; call prepare() first".to_string(),
            )
        })?;
        let idx = selection.sample(rng);
        let entry = &mut self.entries[idx];
        step_update(
            state,
            rng,
            idx,
            &mut entry.update,
            entry.ratio,
            &mut self.stats[idx],
        )
    }
}

/// Static weighted update set containing two concrete update types.
///
/// This covers the common forward/backward pair and small mixed-update cases without dynamic
/// dispatch. Larger static compositions are still deferred.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TwoUpdateSet<A, B> {
    first: A,
    second: B,
    weights: [f64; 2],
    ratios: [f64; 2],
    stats: [UpdateStats; 2],
}

impl<A, B> TwoUpdateSet<A, B> {
    pub fn new(first: A, first_weight: f64, second: B, second_weight: f64) -> Result<Self> {
        Self::with_ratios(first, first_weight, 1.0, second, second_weight, 1.0)
    }

    pub fn inverse_pair(
        first: A,
        first_weight: f64,
        second: B,
        second_weight: f64,
    ) -> Result<Self> {
        if (first_weight == 0.0) != (second_weight == 0.0) {
            return Err(RmcError::InvalidArgument(format!(
                "inverse pair must have both weights zero or both non-zero (got {first_weight} and {second_weight})"
            )));
        }
        if first_weight == 0.0 {
            return Err(RmcError::InvalidState(
                "at least one update weight must be > 0".to_string(),
            ));
        }
        Self::with_ratios(
            first,
            first_weight,
            second_weight / first_weight,
            second,
            second_weight,
            first_weight / second_weight,
        )
    }

    pub fn with_ratios(
        first: A,
        first_weight: f64,
        first_ratio: f64,
        second: B,
        second_weight: f64,
        second_ratio: f64,
    ) -> Result<Self> {
        validate_two_weights(first_weight, second_weight)?;
        Ok(Self {
            first,
            second,
            weights: [first_weight, second_weight],
            ratios: [first_ratio, second_ratio],
            stats: [UpdateStats::default(), UpdateStats::default()],
        })
    }

    pub fn first(&self) -> &A {
        &self.first
    }

    pub fn second(&self) -> &B {
        &self.second
    }

    pub fn first_mut(&mut self) -> &mut A {
        &mut self.first
    }

    pub fn second_mut(&mut self) -> &mut B {
        &mut self.second
    }

    pub fn weights(&self) -> [f64; 2] {
        self.weights
    }

    pub fn ratios(&self) -> [f64; 2] {
        self.ratios
    }
}

impl<A, B> UpdateSet for TwoUpdateSet<A, B> {
    fn stats(&self) -> &[UpdateStats] {
        &self.stats
    }

    fn reset_stats(&mut self) {
        self.stats = Default::default();
    }
}

impl<State, R, A, B> SteppingUpdateSet<State, R> for TwoUpdateSet<A, B>
where
    R: Rng,
    A: Update<State>,
    B: Update<State>,
{
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        validate_two_weights(self.weights[0], self.weights[1])
    }

    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome> {
        let total = self.weights[0] + self.weights[1];
        let first_selected = rng.gen::<f64>() * total < self.weights[0];
        if first_selected {
            step_update(
                state,
                rng,
                0,
                &mut self.first,
                self.ratios[0],
                &mut self.stats[0],
            )
        } else {
            step_update(
                state,
                rng,
                1,
                &mut self.second,
                self.ratios[1],
                &mut self.stats[1],
            )
        }
    }
}

fn validate_two_weights(first: f64, second: f64) -> Result<()> {
    if first < 0.0 || second < 0.0 {
        return Err(RmcError::InvalidArgument(format!(
            "update weights must be >= 0 (got {first} and {second})"
        )));
    }
    if first == 0.0 && second == 0.0 {
        return Err(RmcError::InvalidState(
            "at least one update weight must be > 0".to_string(),
        ));
    }
    Ok(())
}

// Single state-generic stepping helper shared by all static update sets. For acceptance, a
// probability `>= 1.0` short-circuits *without* drawing a uniform; otherwise one `rng.gen::<f64>()`
// is consumed. RNG consumption on the accept path therefore depends on the proposed probability —
// relevant when reasoning about stream alignment across runs.
fn step_update<State, R, U>(
    state: &mut State,
    rng: &mut R,
    update_index: usize,
    update: &mut U,
    ratio: f64,
    stats: &mut UpdateStats,
) -> Result<StepOutcome>
where
    R: Rng,
    U: Update<State>,
{
    stats.nprops += 1;
    let probability = update.attempt(state, rng) * ratio;
    // Accepted-certain moves do not consume an acceptance draw.
    let (accepted, impossible) = if probability < 0.0 {
        stats.nimps += 1;
        update.reject(state);
        (false, true)
    } else if probability >= 1.0 || rng.gen::<f64>() < probability {
        stats.naccs += 1;
        update.accept(state);
        (true, false)
    } else {
        update.reject(state);
        (false, false)
    };

    Ok(StepOutcome {
        update_index,
        probability,
        accepted,
        impossible,
    })
}

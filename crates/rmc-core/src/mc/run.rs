use crate::{Result, RmcError};

use super::sets::SinkMeasurementSet;
use super::sink::ResultSink;
use super::traits::{Kernel, Measurement, RunCallbacks};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationParams {
    /// Maximum number of MC steps to execute.
    pub max_steps: u64,
    /// Number of MC steps between measurements/cycle callbacks.
    pub steps_per_cycle: u64,
    /// Number of cycles between checkpoint callbacks and `stop_when` checks.
    ///
    /// `0` disables checkpoint callbacks and stop checks.
    pub cycles_per_check: u64,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            max_steps: 1,
            steps_per_cycle: 1,
            cycles_per_check: 1,
        }
    }
}

impl SimulationParams {
    /// Validate the parameters.
    ///
    /// Note on partial final cycles: if `max_steps` is not a multiple of `steps_per_cycle`, the
    /// final cycle runs fewer than `steps_per_cycle` steps and is *still measured* once. Callers who
    /// need every cycle to contain exactly `steps_per_cycle` steps should choose `max_steps` as a
    /// multiple of `steps_per_cycle` (and of `cycles_per_check * steps_per_cycle` for aligned
    /// checkpoints).
    pub fn validate(&self) -> Result<()> {
        if self.steps_per_cycle == 0 {
            return Err(RmcError::InvalidArgument(
                "steps_per_cycle must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SimulationStats {
    /// Number of kernel steps completed.
    pub steps_done: u64,
    /// Number of measurement/callback cycles completed.
    pub cycles_done: u64,
}

impl crate::Merge for SimulationStats {
    fn merge(self, other: Self) -> Self {
        Self {
            steps_done: self.steps_done + other.steps_done,
            cycles_done: self.cycles_done + other.cycles_done,
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SimulationCtx {
    /// Total steps completed at the callback point.
    pub steps_done: u64,
    /// Total cycles completed at the callback point.
    pub cycles_done: u64,
    /// One-based step index inside the current cycle for `on_step`; `0` for cycle/checkpoint
    /// callbacks.
    pub steps_in_cycle: u64,
}

/// Callback implementation that does nothing and never stops a run.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopCallbacks;

impl RunCallbacks<SimulationCtx> for NoopCallbacks {}

/// Measurement that ignores every cycle and returns `()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullMeasurement;

impl<State> Measurement<State> for NullMeasurement {
    type Output = ();

    fn measure(&mut self, _state: &State) {}

    fn finish(self) -> Self::Output {}
}

/// The single MC run loop shared by every entry point.
///
/// It steps the `kernel` against `state`, invokes `measure_cycle(state)` once per cycle, fires
/// callbacks, and honours `stop_when`.
fn drive<State, R, K, C>(
    state: &mut State,
    rng: &mut R,
    kernel: &mut K,
    params: SimulationParams,
    callbacks: &mut C,
    mut measure_cycle: impl FnMut(&State),
) -> Result<SimulationStats>
where
    K: Kernel<State, R>,
    C: RunCallbacks<SimulationCtx>,
{
    params.validate()?;
    kernel.prepare(state)?;

    let mut stats = SimulationStats::default();
    while stats.steps_done < params.max_steps {
        for steps_in_cycle in 0..params.steps_per_cycle {
            if stats.steps_done >= params.max_steps {
                break;
            }

            kernel.step(state, rng)?;
            stats.steps_done += 1;
            callbacks.on_step(&SimulationCtx {
                steps_done: stats.steps_done,
                cycles_done: stats.cycles_done,
                steps_in_cycle: steps_in_cycle + 1,
            });
        }

        measure_cycle(state);
        stats.cycles_done += 1;
        let ctx = SimulationCtx {
            steps_done: stats.steps_done,
            cycles_done: stats.cycles_done,
            steps_in_cycle: 0,
        };
        callbacks.on_cycle(&ctx);

        if params.cycles_per_check > 0 && stats.cycles_done % params.cycles_per_check == 0 {
            callbacks.on_checkpoint(&ctx);
            if callbacks.stop_when(&ctx) {
                break;
            }
        }
    }

    Ok(stats)
}

/// Run with sink measurements that emit named artifacts into `sink`.
///
/// This path is output-only: measurements do not return Rust-native structured values. Use
/// [`run_typed`] when typed results should flow back by ownership.
pub fn run_with_sink<State, R, K>(
    state: State,
    rng: &mut R,
    kernel: &mut K,
    measurements: &mut SinkMeasurementSet<State>,
    sink: &mut dyn ResultSink,
    params: SimulationParams,
) -> Result<(State, SimulationStats)>
where
    K: Kernel<State, R>,
{
    let mut callbacks = NoopCallbacks;
    run_with_sink_and_callbacks(
        state,
        rng,
        kernel,
        measurements,
        sink,
        params,
        &mut callbacks,
    )
}

pub fn run_with_sink_and_callbacks<State, R, K, C>(
    mut state: State,
    rng: &mut R,
    kernel: &mut K,
    measurements: &mut SinkMeasurementSet<State>,
    sink: &mut dyn ResultSink,
    params: SimulationParams,
    callbacks: &mut C,
) -> Result<(State, SimulationStats)>
where
    K: Kernel<State, R>,
    C: RunCallbacks<SimulationCtx>,
{
    measurements.refresh_active();
    let stats = drive(&mut state, rng, kernel, params, callbacks, |state| {
        measurements.measure_all(state)
    })?;
    measurements.write_all(sink)?;
    Ok((state, stats))
}

/// Run with a typed measurement, returning the final `state`, run stats, and the measurement's
/// `Output` by ownership (no `Any`/downcast).
///
/// This is the preferred entry point when callers need results. A stateless run passes `()` as the
/// state; a stateful run passes an owned chain state such as a lattice or particle configuration.
pub fn run_typed<State, R, K, M>(
    state: State,
    rng: &mut R,
    kernel: &mut K,
    measurement: M,
    params: SimulationParams,
) -> Result<(State, SimulationStats, M::Output)>
where
    K: Kernel<State, R>,
    M: Measurement<State>,
{
    let mut callbacks = NoopCallbacks;
    run_typed_with_callbacks(state, rng, kernel, measurement, params, &mut callbacks)
}

pub fn run_typed_with_callbacks<State, R, K, M, C>(
    mut state: State,
    rng: &mut R,
    kernel: &mut K,
    mut measurement: M,
    params: SimulationParams,
    callbacks: &mut C,
) -> Result<(State, SimulationStats, M::Output)>
where
    K: Kernel<State, R>,
    M: Measurement<State>,
    C: RunCallbacks<SimulationCtx>,
{
    let stats = drive(&mut state, rng, kernel, params, callbacks, |state| {
        measurement.measure(state)
    })?;
    Ok((state, stats, measurement.finish()))
}

use rayon::prelude::*;
use rayon::ThreadPool;

use crate::random::{ChainId, SeedSource};
use crate::{Merge, Result, RmcError};

use super::run::{
    run_typed, run_typed_with_callbacks, NoopCallbacks, NullMeasurement, SimulationParams,
    SimulationStats,
};
use super::traits::{Kernel, Measurement, RunCallbacks};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParallelConfig {
    /// Number of independent chains to run.
    pub chains: u64,
    /// Master seed used to derive one RNG stream per chain.
    pub seed: SeedSource,
    /// Per-chain simulation parameters.
    pub params: SimulationParams,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RunPlan {
    /// Number of independent chains to run.
    pub chains: u64,
    /// Master seed used to derive one RNG stream per chain.
    pub seed: SeedSource,
    /// Optional warmup phase. Set `max_steps` to `0` to skip it.
    pub warmup: SimulationParams,
    /// Main measured phase.
    pub main: SimulationParams,
}

impl RunPlan {
    /// Validate plan and per-phase run parameters.
    pub fn validate(&self) -> Result<()> {
        if self.chains == 0 {
            return Err(RmcError::InvalidArgument("chains must be > 0".to_string()));
        }
        self.warmup.validate()?;
        self.main.validate()
    }
}

impl ParallelConfig {
    /// Validate parallel and per-chain run parameters.
    pub fn validate(&self) -> Result<()> {
        if self.chains == 0 {
            return Err(RmcError::InvalidArgument("chains must be > 0".to_string()));
        }
        self.params.validate()
    }
}

/// Run `config.chains` independent chains in parallel, each with its own `State`, kernel,
/// measurement, and reproducible per-chain RNG, then reduce the per-chain `M::Output`s via [`Merge`].
///
/// A stateless simulation uses `State = ()` and a `build` closure that returns `((), kernel, meas)`.
/// Reduction preserves chain order (`into_par_iter().collect()` then left-fold), so the aggregate is
/// reproducible across thread counts.
pub fn run_parallel<State, K, M, B>(
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
{
    run_parallel_impl(config, &build).map(|(stats, output, _kernels)| (stats, output))
}

/// Like [`run_parallel`], but also returns the final per-chain kernels in chain order.
///
/// This exposes update-set counters and other kernel-owned run state without requiring callers to
/// reimplement the parallel chain orchestration.
pub fn run_parallel_full<State, K, M, B>(
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output, Vec<K>)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
{
    run_parallel_impl(config, &build)
}

/// Like [`run_parallel`], but constructs one callback value per chain.
///
/// This is useful for progress bars, per-chain logging, or convergence probes that do not require
/// access to the chain state. The callback factory receives the stable [`ChainId`] for the chain.
pub fn run_parallel_with_callbacks<State, K, M, B, C, CB>(
    config: ParallelConfig,
    build: B,
    callbacks: CB,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
    C: RunCallbacks<super::run::SimulationCtx> + Send,
    CB: Fn(ChainId) -> C + Send + Sync,
{
    run_parallel_impl_with_callbacks(config, &build, &callbacks)
        .map(|(stats, output, _kernels)| (stats, output))
}

/// Like [`run_parallel_full`], but constructs one callback value per chain.
pub fn run_parallel_full_with_callbacks<State, K, M, B, C, CB>(
    config: ParallelConfig,
    build: B,
    callbacks: CB,
) -> Result<(SimulationStats, M::Output, Vec<K>)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
    C: RunCallbacks<super::run::SimulationCtx> + Send,
    CB: Fn(ChainId) -> C + Send + Sync,
{
    run_parallel_impl_with_callbacks(config, &build, &callbacks)
}

/// Like [`run_parallel`] but executes inside the provided rayon thread pool.
pub fn run_parallel_in_pool<State, K, M, B>(
    pool: &ThreadPool,
    config: ParallelConfig,
    build: B,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
{
    pool.install(|| run_parallel_impl(config, &build))
        .map(|(stats, output, _kernels)| (stats, output))
}

/// Like [`run_parallel_with_callbacks`] but executes inside the provided rayon thread pool.
pub fn run_parallel_in_pool_with_callbacks<State, K, M, B, C, CB>(
    pool: &ThreadPool,
    config: ParallelConfig,
    build: B,
    callbacks: CB,
) -> Result<(SimulationStats, M::Output)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    B: Fn(ChainId) -> (State, K, M) + Send + Sync,
    C: RunCallbacks<super::run::SimulationCtx> + Send,
    CB: Fn(ChainId) -> C + Send + Sync,
{
    pool.install(|| run_parallel_impl_with_callbacks(config, &build, &callbacks))
        .map(|(stats, output, _kernels)| (stats, output))
}

/// Run a warmup phase followed by a measured phase for each chain, then merge measured outputs.
///
/// The warmup and measured kernels are built separately so warmup-only counters do not leak into
/// the returned measured kernels.
pub fn run_plan_full<State, WK, K, M, WB, MB>(
    plan: RunPlan,
    build_warmup: WB,
    build_main: MB,
) -> Result<(SimulationStats, M::Output, Vec<K>, Vec<State>)>
where
    State: Send,
    WK: Kernel<State, crate::random::DefaultRng> + Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    WB: Fn(ChainId) -> (State, WK) + Send + Sync,
    MB: Fn(ChainId, &State) -> (K, M) + Send + Sync,
{
    run_plan_full_with_callbacks(
        plan,
        build_warmup,
        build_main,
        |_chain| NoopCallbacks,
        |_chain| NoopCallbacks,
    )
}

/// Like [`run_plan_full`], but constructs one callback value per chain and phase.
pub fn run_plan_full_with_callbacks<State, WK, K, M, WB, MB, WC, MC, WCB, MCB>(
    plan: RunPlan,
    build_warmup: WB,
    build_main: MB,
    warmup_callbacks: WCB,
    main_callbacks: MCB,
) -> Result<(SimulationStats, M::Output, Vec<K>, Vec<State>)>
where
    State: Send,
    WK: Kernel<State, crate::random::DefaultRng> + Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    WB: Fn(ChainId) -> (State, WK) + Send + Sync,
    MB: Fn(ChainId, &State) -> (K, M) + Send + Sync,
    WC: RunCallbacks<super::run::SimulationCtx> + Send,
    MC: RunCallbacks<super::run::SimulationCtx> + Send,
    WCB: Fn(ChainId) -> WC + Send + Sync,
    MCB: Fn(ChainId) -> MC + Send + Sync,
{
    plan.validate()?;

    let partials = (0..plan.chains)
        .into_par_iter()
        .map(|chain| {
            let chain_id = ChainId(chain);
            let mut rng = plan.seed.rng_for(chain_id);
            let (mut state, mut warmup_kernel) = build_warmup(chain_id);
            if plan.warmup.max_steps > 0 {
                let mut callbacks = warmup_callbacks(chain_id);
                let (warm_state, _stats, _output) = run_typed_with_callbacks(
                    state,
                    &mut rng,
                    &mut warmup_kernel,
                    NullMeasurement,
                    plan.warmup,
                    &mut callbacks,
                )?;
                state = warm_state;
            }

            let (mut kernel, measurement) = build_main(chain_id, &state);
            let mut callbacks = main_callbacks(chain_id);
            run_typed_with_callbacks(
                state,
                &mut rng,
                &mut kernel,
                measurement,
                plan.main,
                &mut callbacks,
            )
            .map(|(state, stats, output)| (stats, output, kernel, state))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    let mut kernels = Vec::with_capacity(partials.len());
    let mut states = Vec::with_capacity(partials.len());
    for partial in partials {
        let (stats, output, kernel, state) = partial?;
        kernels.push(kernel);
        states.push(state);
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged
        .map(|(stats, output)| (stats, output, kernels, states))
        .ok_or_else(|| RmcError::InvalidState("run plan produced no chains".to_string()))
}

fn run_parallel_impl<State, K, M>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (State, K, M) + Sync),
) -> Result<(SimulationStats, M::Output, Vec<K>)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
{
    config.validate()?;

    let partials = (0..config.chains)
        .into_par_iter()
        .map(|chain| {
            let chain_id = ChainId(chain);
            let mut rng = config.seed.rng_for(chain_id);
            let (state, mut kernel, measurement) = build(chain_id);
            run_typed(state, &mut rng, &mut kernel, measurement, config.params)
                .map(|(_state, stats, output)| (stats, output, kernel))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    let mut kernels = Vec::with_capacity(partials.len());
    for partial in partials {
        let (stats, output, kernel) = partial?;
        kernels.push(kernel);
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged
        .map(|(stats, output)| (stats, output, kernels))
        .ok_or_else(|| RmcError::InvalidState("parallel run produced no chains".to_string()))
}

fn run_parallel_impl_with_callbacks<State, K, M, C>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (State, K, M) + Sync),
    callbacks: &(impl Fn(ChainId) -> C + Sync),
) -> Result<(SimulationStats, M::Output, Vec<K>)>
where
    State: Send,
    K: Kernel<State, crate::random::DefaultRng> + Send,
    M: Measurement<State> + Send,
    M::Output: Merge + Send,
    C: RunCallbacks<super::run::SimulationCtx> + Send,
{
    config.validate()?;

    let partials = (0..config.chains)
        .into_par_iter()
        .map(|chain| {
            let chain_id = ChainId(chain);
            let mut rng = config.seed.rng_for(chain_id);
            let (state, mut kernel, measurement) = build(chain_id);
            let mut callbacks = callbacks(chain_id);
            run_typed_with_callbacks(
                state,
                &mut rng,
                &mut kernel,
                measurement,
                config.params,
                &mut callbacks,
            )
            .map(|(_state, stats, output)| (stats, output, kernel))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    let mut kernels = Vec::with_capacity(partials.len());
    for partial in partials {
        let (stats, output, kernel) = partial?;
        kernels.push(kernel);
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged
        .map(|(stats, output)| (stats, output, kernels))
        .ok_or_else(|| RmcError::InvalidState("parallel run produced no chains".to_string()))
}

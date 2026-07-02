use rayon::prelude::*;
use rayon::ThreadPool;

use crate::random::{ChainId, SeedSource};
use crate::{Merge, Result, RmcError};

use super::run::{run_typed, run_typed_with_callbacks, SimulationParams, SimulationStats};
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
}

fn run_parallel_impl<State, K, M>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (State, K, M) + Sync),
) -> Result<(SimulationStats, M::Output)>
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
                .map(|(_state, stats, output)| (stats, output))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    for partial in partials {
        let (stats, output) = partial?;
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged.ok_or_else(|| RmcError::InvalidState("parallel run produced no chains".to_string()))
}

fn run_parallel_impl_with_callbacks<State, K, M, C>(
    config: ParallelConfig,
    build: &(impl Fn(ChainId) -> (State, K, M) + Sync),
    callbacks: &(impl Fn(ChainId) -> C + Sync),
) -> Result<(SimulationStats, M::Output)>
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
            .map(|(_state, stats, output)| (stats, output))
        })
        .collect::<Vec<_>>();

    let mut merged: Option<(SimulationStats, M::Output)> = None;
    for partial in partials {
        let (stats, output) = partial?;
        merged = Some(match merged {
            Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
            None => (stats, output),
        });
    }

    merged.ok_or_else(|| RmcError::InvalidState("parallel run produced no chains".to_string()))
}

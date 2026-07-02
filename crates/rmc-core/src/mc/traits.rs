use rand::Rng;

use crate::Result;

/// A single Monte Carlo update acting on a chain's `State`.
///
/// This is the one update trait: there is no separate "stateless" variant. A stateless update is
/// simply `Update<()>` — its `state` argument is the zero-sized unit and carries no information.
/// Owning the mutated data in `State` (threaded by the run loop) is preferred over reaching for
/// `Arc`/`Rc` shared cells, because each parallel chain then owns an independent `State`.
///
/// `attempt` is generic over the concrete RNG (`R: Rng + ?Sized`) so every `rng.next_u64()` /
/// `rng.gen()` call inside an update can inline instead of going through a vtable.
pub trait Update<State> {
    /// Propose a move and return its acceptance probability.
    ///
    /// A returned value `< 0.0` marks the move as *impossible* (rejected, counted separately);
    /// `>= 1.0` is always accepted; otherwise it is accepted with the returned probability.
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut State, rng: &mut R) -> f64;

    /// Commit the proposed move to `state`.
    fn accept(&mut self, state: &mut State);

    /// Roll back the proposed move. Defaults to a no-op (the common case where `attempt` does not
    /// mutate `state` until `accept`).
    fn reject(&mut self, _state: &mut State) {}
}

/// A per-cycle measurement of a chain's `State`, returning a typed `Output` by ownership.
///
/// A stateless measurement is `Measurement<()>`. Results flow back through `finish` with no
/// `Any`/downcast; see [`run_typed`](crate::mc::run_typed).
pub trait Measurement<State> {
    type Output;

    fn measure(&mut self, state: &State);
    fn finish(self) -> Self::Output;
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepOutcome {
    pub update_index: usize,
    pub probability: f64,
    pub accepted: bool,
    pub impossible: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UpdateStats {
    pub nprops: u64,
    pub naccs: u64,
    pub nimps: u64,
}

/// Common, dispatch-free view over an update set's per-update statistics.
pub trait UpdateSet {
    fn stats(&self) -> &[UpdateStats];
}

/// An update set that can be driven for one MC step against a chain's `State`.
///
/// The static implementors ([`SingleUpdateSet`](crate::mc::SingleUpdateSet),
/// [`TwoUpdateSet`](crate::mc::TwoUpdateSet), [`WeightedUpdateSet`](crate::mc::WeightedUpdateSet))
/// are monomorphized over `R`, so RNG draws inline.
pub trait SteppingUpdateSet<State, R>: UpdateSet {
    fn prepare(&mut self, state: &mut State) -> Result<()>;
    fn select_and_step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;
}

/// A kernel advances a chain by one MC step. A stateless kernel is `Kernel<(), R>`.
pub trait Kernel<State, R> {
    fn prepare(&mut self, _state: &mut State) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome>;
}

pub trait RunCallbacks<C> {
    fn on_step(&mut self, _ctx: &C) {}
    fn on_cycle(&mut self, _ctx: &C) {}
    fn on_checkpoint(&mut self, _ctx: &C) {}
    fn stop_when(&mut self, _ctx: &C) -> bool {
        false
    }
}

use rand::Rng;

use crate::Result;

use super::traits::{Kernel, StepOutcome, SteppingUpdateSet};

/// Metropolis kernel over a monomorphized (static) update set.
///
/// One kernel serves both stateless and stateful simulations: the `State` lives in the
/// [`Kernel`] impl, so the same `MetropolisKernel` is `Kernel<(), R>` for a stateless update set and
/// `Kernel<Lattice, R>` for a stateful one.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MetropolisKernel<S> {
    updates: S,
}

impl<S> MetropolisKernel<S> {
    /// Create a kernel around a static update set.
    pub fn new(updates: S) -> Self {
        Self { updates }
    }

    /// Borrow the contained update set.
    pub fn updates(&self) -> &S {
        &self.updates
    }

    /// Mutably borrow the contained update set.
    pub fn updates_mut(&mut self) -> &mut S {
        &mut self.updates
    }

    /// Consume the kernel and return its update set.
    pub fn into_updates(self) -> S {
        self.updates
    }
}

impl<State, R, S> Kernel<State, R> for MetropolisKernel<S>
where
    R: Rng,
    S: SteppingUpdateSet<State, R>,
{
    fn prepare(&mut self, state: &mut State) -> Result<()> {
        self.updates.prepare(state)
    }

    fn step(&mut self, state: &mut State, rng: &mut R) -> Result<StepOutcome> {
        self.updates.select_and_step(state, rng)
    }

    fn reset_stats(&mut self) {
        self.updates.reset_stats()
    }
}

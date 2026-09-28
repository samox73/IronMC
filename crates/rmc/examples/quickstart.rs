//! Quick start (mirrored in the top-level README): Metropolis sampling of a unit Gaussian,
//! π(x) ∝ exp(-x²/2), on 8 independent chains, with ⟨x²⟩ = 1 and a jackknife error bar.

use rmc::mc::{MetropolisKernel, Runner, SimulationParams, SingleUpdateSet, Update, UpdateSet};
use rmc::random::{ChainId, Rng, SeedSource};
use rmc::stats::BinnedScalar;

/// Uniform shift x → x + width·(2u − 1): symmetric, so the Metropolis acceptance is π(x')/π(x).
struct Shift {
    width: f64,
    proposed: f64,
}

impl Update<f64> for Shift {
    fn attempt<R: Rng + ?Sized>(&mut self, x: &mut f64, rng: &mut R) -> f64 {
        self.proposed = *x + self.width * (2.0 * rng.gen::<f64>() - 1.0);
        (-(self.proposed * self.proposed - *x * *x) / 2.0).exp()
    }

    fn accept(&mut self, x: &mut f64) {
        *x = self.proposed;
    }
}

/// Everything one chain needs: initial state, kernel, and measurement. Called once per chain.
fn build_chain(
    _chain: ChainId,
) -> (f64, MetropolisKernel<SingleUpdateSet<Shift>>, BinnedScalar<impl Fn(&f64) -> f64>) {
    let kernel = MetropolisKernel::new(SingleUpdateSet::new(Shift { width: 2.5, proposed: 0.0 }));
    let x_squared = BinnedScalar::new(1_000, |x: &f64| x * x).expect("block size > 0");
    (0.0, kernel, x_squared)
}

fn main() -> rmc::Result<()> {
    let params = SimulationParams { max_steps: 1_000_000, steps_per_cycle: 10, cycles_per_check: 0 };
    let warmup = SimulationParams { max_steps: 10_000, ..params };

    let report = Runner::new(SeedSource::new(42), build_chain)
        .chains(8)
        .warmup(warmup)
        .run(params)?;

    let x2 = report.output; // the 8 chains' jackknife blocks, merged
    let stats = report.kernels[0].updates().stats()[0];
    println!(
        "<x^2> = {:.4} +/- {:.4} (exact 1), {} steps, acceptance {:.2} (chain 0)",
        x2.estimate().unwrap(),
        x2.standard_error().unwrap(),
        report.stats.steps_done,
        stats.naccs as f64 / stats.nprops as f64,
    );
    Ok(())
}

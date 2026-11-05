//! Seed-to-seed calibration of the Fröhlich error bars.
//!
//! ```sh
//! cargo run --release -p rmc-frohlich --example calibrate -- fixtures/bench-frohlich.json 8 30000000
//! ```
//!
//! Runs the config under `seeds` consecutive seeds (`max_steps` overridden by the third argument)
//! and compares the empirical scatter of the reported `energy`/`quasiparticle_weight` means with
//! their mean stated stderr. Honest error bars give a ratio near 1.

use rmc_frohlich::app::{BenchReport, run_seeds};
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::measurement::Estimate;
use rmc_stats::ErrorBarCalibration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [path, seeds, steps] = args.as_slice() else {
        return Err("usage: calibrate <config.json> <seeds> <max_steps>".into());
    };
    let cfg = RunConfig {
        max_steps: steps.parse()?,
        ..RunConfig::load_json(path)?
    };
    let reports = run_seeds(&cfg, seeds.parse()?)?;

    for (offset, report) in reports.iter().enumerate() {
        let s = &report.summary;
        println!(
            "seed {:<20} E {:+.6} +/- {:.6}   Z {:.6} +/- {:.6}   E_est {:+.6}",
            cfg.seed.wrapping_add(offset as u64),
            s.energy.mean,
            s.energy.stderr,
            s.quasiparticle_weight.mean,
            s.quasiparticle_weight.stderr,
            s.final_energy_estimate,
        );
    }
    let pick = |name: &str, f: fn(&BenchReport) -> Estimate| {
        let c = ErrorBarCalibration::from_estimates(reports.iter().map(|r| {
            let e = f(r);
            (e.mean, e.stderr)
        }));
        println!(
            "{name}: empirical std {:.6}, mean stated stderr {:.6}, ratio {:.3}",
            c.empirical_std,
            c.mean_stderr,
            c.ratio()
        );
    };
    pick("E", |r| r.summary.energy);
    pick("Z", |r| r.summary.quasiparticle_weight);
    Ok(())
}

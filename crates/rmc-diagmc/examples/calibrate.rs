//! Seed-to-seed calibration of the `rmc-diagmc` error bars on the α = 1 Fröhlich polaron.
//!
//! ```sh
//! cargo run --release -p rmc-diagmc --example calibrate -- 8 10000000
//! ```
//!
//! Runs the fixed-τ stage-zero config (μ = −1.213, τ_max = 30, τ_fit = 15, 1M warmup steps,
//! `DiagConfig::default()` otherwise) under `seeds` consecutive seeds and compares the empirical
//! scatter of `e0_thermo`, `sign` and `mean_order` with their mean stated stderr. Honest error
//! bars give a ratio near 1; the Fröhlich sign is exactly 1, so its ratio is 0/0.

use std::sync::Arc;

use rmc_diagmc::{DiagConfig, Estimate, FrohlichModel, Report, run_seeds};
use rmc_stats::ErrorBarCalibration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [seeds, steps] = args.as_slice() else {
        return Err("usage: calibrate <seeds> <max_steps>".into());
    };
    let cfg = DiagConfig {
        mu: -1.013 - 0.2,
        max_steps: steps.parse()?,
        warmup_steps: 1_000_000,
        fixed_tau: true,
        max_tau: 30.0,
        tau_fit: 15.0,
        ..DiagConfig::default()
    };
    let reports = run_seeds(Arc::new(FrohlichModel::new(1.0)), &cfg, seeds.parse()?)?;

    for (offset, r) in reports.iter().enumerate() {
        println!(
            "seed {:<3} E_thermo {:+.6} +/- {:.6}   sign {:.6} +/- {:.6}   order {:.6} +/- {:.6}",
            cfg.seed.wrapping_add(offset as u64),
            r.e0_thermo.mean,
            r.e0_thermo.stderr,
            r.sign.mean,
            r.sign.stderr,
            r.mean_order.mean,
            r.mean_order.stderr,
        );
    }
    let pick = |name: &str, f: fn(&Report) -> Estimate| {
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
    pick("e0_thermo", |r| r.e0_thermo);
    pick("sign", |r| r.sign);
    pick("mean_order", |r| r.mean_order);
    Ok(())
}

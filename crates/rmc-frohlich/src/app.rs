use std::fs;
use std::path::{Path, PathBuf};

use indicatif::{MultiProgress, ProgressBar};
use rmc_core::mc::{
    run_chain, IndicatifProgress, MetropolisKernel, NoopCallbacks, NullMeasurement, Runner,
    SimulationStats, WeightedUpdateSet,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_io::{load_payload_json, save_payload_json};

use crate::config::RunConfig;
use crate::diagram::Diagram;
use crate::fourier::analyze_stats;
use crate::measurement::Estimate;
use crate::measurement::{PolaronMeasurement, PolaronStats};
use crate::update_stats::{self, UpdateStatEntry};
use crate::updates::{default_update_set, PolaronUpdate};

pub type PolaronKernel = MetropolisKernel<WeightedUpdateSet<PolaronUpdate>>;
pub type AppResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointPayload {
    pub config: RunConfig,
    pub diagram: Diagram,
    pub stats: SimulationStats,
    pub measurement: PolaronStats,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RunOutput {
    pub stats: SimulationStats,
    pub measurement: PolaronStats,
    pub final_state: Option<Diagram>,
    pub update_stats: Vec<UpdateStatEntry>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BenchReport {
    pub steps_done: u64,
    pub warmup_steps: u64,
    pub warmup_secs: f64,
    pub sample_secs: f64,
    pub steps_per_sec: f64,
    pub summary: ValidationSummary,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ValidationSummary {
    pub steps_done: u64,
    pub cycles_done: u64,
    pub sample_count: usize,
    pub chains: u64,
    pub energy: Estimate,
    pub quasiparticle_weight: Estimate,
    pub target_energy: f64,
    pub target_quasiparticle_weight: f64,
    pub energy_delta: f64,
    pub quasiparticle_weight_delta: f64,
    pub zeroth_mean: Option<f64>,
    pub mean_order: Option<f64>,
    pub finite_selfenergy_bins: usize,
    pub total_selfenergy_bins: usize,
    pub final_energy_estimate: f64,
    pub energy_estimate_history: Vec<f64>,
    pub update_stats: Vec<UpdateStatEntry>,
}

impl ValidationSummary {
    pub fn text(&self) -> String {
        let mut text = format!(
            concat!(
                "steps_done: {steps}\n",
                "cycles_done: {cycles}\n",
                "samples: {samples}\n",
                "chains: {chains}\n",
                "E: {e:.8} +/- {e_err:.8}  target {e_target:.8}  delta {e_delta:.8}\n",
                "Z: {z:.8} +/- {z_err:.8}  target {z_target:.8}  delta {z_delta:.8}\n",
                "zeroth_mean: {zeroth:.8}\n",
                "mean_order: {order:.8}\n",
                "finite_selfenergy_bins: {finite}/{total}\n",
                "final_reweighting_energy_estimate: {reweight:.8}"
            ),
            steps = self.steps_done,
            cycles = self.cycles_done,
            samples = self.sample_count,
            chains = self.chains,
            e = self.energy.mean,
            e_err = self.energy.stderr,
            e_target = self.target_energy,
            e_delta = self.energy_delta,
            z = self.quasiparticle_weight.mean,
            z_err = self.quasiparticle_weight.stderr,
            z_target = self.target_quasiparticle_weight,
            z_delta = self.quasiparticle_weight_delta,
            zeroth = self.zeroth_mean.unwrap_or(f64::NAN),
            order = self.mean_order.unwrap_or(f64::NAN),
            finite = self.finite_selfenergy_bins,
            total = self.total_selfenergy_bins,
            reweight = self.final_energy_estimate,
        );
        text.push('\n');
        text.push_str(&update_stats::render(&self.update_stats));
        text
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResultsManifest {
    pub summary: ValidationSummary,
    pub files: Vec<String>,
}

pub fn build_diagram(cfg: &RunConfig) -> Diagram {
    Diagram::with_parameters(
        cfg.alpha,
        cfg.mu,
        cfg.momentum,
        cfg.max_tau,
        cfg.start_tau,
        cfg.min_order,
        cfg.max_order,
    )
}

pub fn build_kernel() -> rmc_core::Result<PolaronKernel> {
    Ok(MetropolisKernel::new(default_update_set()?))
}

pub fn build_measurement(cfg: &RunConfig, diagram: &Diagram) -> PolaronMeasurement {
    PolaronMeasurement::new(
        cfg.num_bins,
        cfg.max_tau,
        cfg.n_batches,
        cfg.max_steps.div_ceil(cfg.steps_per_cycle.max(1)) as usize,
        cfg.energy_estimate,
        cfg.initial_self_consistent_period,
        cfg.period_multiplier,
        diagram,
    )
}

pub fn build_chain(
    cfg: RunConfig,
) -> impl Fn(ChainId) -> (Diagram, PolaronKernel, PolaronMeasurement) {
    move |_chain| {
        let diagram = build_diagram(&cfg);
        let kernel = build_kernel().expect("default polaron update set must be valid");
        let measurement = build_measurement(&cfg, &diagram);
        (diagram, kernel, measurement)
    }
}

pub fn run_from_config(cfg: &RunConfig) -> AppResult<RunOutput> {
    run_from_config_with_progress(cfg, false)
}

/// Single-chain run that times *only* the sampling loop (no progress bar, no FFT, no file I/O),
/// for a clean engine throughput number comparable to the C++ sampling loop. Warmup is timed
/// separately and excluded from `steps_per_sec`.
pub fn run_bench(cfg: &RunConfig) -> AppResult<BenchReport> {
    use std::time::Instant;
    let mut rng = SeedSource::new(cfg.seed).rng_for(ChainId(0));
    let mut state = build_diagram(cfg);

    let mut warmup_secs = 0.0;
    if cfg.warmup_steps > 0 {
        let mut warmup_kernel = build_kernel()?;
        let t = Instant::now();
        let (warm_state, _warm_stats, _warm_output) = run_chain(
            state,
            &mut rng,
            &mut warmup_kernel,
            NullMeasurement,
            cfg.warmup_params(),
            NoopCallbacks,
        )?;
        warmup_secs = t.elapsed().as_secs_f64();
        state = warm_state;
    }

    let mut kernel = build_kernel()?;
    let measurement = build_measurement(cfg, &state);
    let t = Instant::now();
    let (final_state, stats, measurement) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        measurement,
        cfg.simulation_params(),
        NoopCallbacks,
    )?;
    let sample_secs = t.elapsed().as_secs_f64();
    let steps_done = stats.steps_done;

    let output = RunOutput {
        stats,
        measurement,
        final_state: Some(final_state),
        update_stats: update_stats::collect(&kernel),
    };
    let summary = summarize_output(cfg, &output);

    Ok(BenchReport {
        steps_done,
        warmup_steps: cfg.warmup_steps,
        warmup_secs,
        sample_secs,
        steps_per_sec: steps_done as f64 / sample_secs,
        summary,
    })
}

pub fn run_from_config_with_progress(cfg: &RunConfig, show_progress: bool) -> AppResult<RunOutput> {
    let runner = Runner::new(SeedSource::new(cfg.seed), build_chain(cfg.clone()))
        .chains(cfg.chains)
        .warmup(cfg.warmup_params());

    let report = if show_progress {
        let multi = (cfg.chains > 1).then(MultiProgress::new);
        runner
            .warmup_callbacks(|chain: ChainId| {
                progress_callback(
                    cfg.warmup_steps,
                    if cfg.chains > 1 {
                        format!("warmup {}", chain.0)
                    } else {
                        "warmup".to_string()
                    },
                    if cfg.chains > 1 {
                        format!("warmup {} done", chain.0)
                    } else {
                        "warmup done".to_string()
                    },
                    multi.as_ref(),
                )
            })
            .callbacks(|chain: ChainId| {
                progress_callback(
                    cfg.max_steps,
                    format!("chain {}", chain.0),
                    format!("chain {} done", chain.0),
                    multi.as_ref(),
                )
            })
            .run(cfg.simulation_params())?
    } else {
        runner.run(cfg.simulation_params())?
    };

    Ok(RunOutput {
        stats: report.stats,
        measurement: report.output,
        final_state: (cfg.chains == 1).then(|| report.states.into_iter().next().unwrap()),
        update_stats: update_stats::merge(
            report.kernels.iter().map(update_stats::collect).collect(),
        ),
    })
}

fn progress_callback(
    total_steps: u64,
    label: impl Into<String>,
    finish_message: impl Into<String>,
    multi: Option<&MultiProgress>,
) -> IndicatifProgress {
    let bar = match multi {
        Some(multi) => multi.add(ProgressBar::new(total_steps)),
        None => ProgressBar::new(total_steps),
    };
    bar.set_style(rmc_core::mc::default_progress_style());
    bar.set_prefix(label.into());
    IndicatifProgress::new(bar).with_finish_message(finish_message)
}

pub fn save_checkpoint(path: impl AsRef<Path>, payload: &CheckpointPayload) -> AppResult<()> {
    Ok(save_payload_json(path, payload)?)
}

pub fn load_checkpoint(path: impl AsRef<Path>) -> AppResult<CheckpointPayload> {
    Ok(load_payload_json(path)?)
}

pub fn checkpoint_from_output(cfg: RunConfig, output: RunOutput) -> Option<CheckpointPayload> {
    Some(CheckpointPayload {
        config: cfg,
        diagram: output.final_state?,
        stats: output.stats,
        measurement: output.measurement,
    })
}

pub fn summarize_output(cfg: &RunConfig, output: &RunOutput) -> ValidationSummary {
    const TARGET_ENERGY: f64 = -1.013;
    const TARGET_Z: f64 = 0.59;

    let energy = output.measurement.jackknife_energy();
    let quasiparticle_weight = output.measurement.jackknife_quasiparticle_weight();
    let selfenergy = output.measurement.jackknife_selfenergy();
    let finite_selfenergy_bins = selfenergy
        .mean
        .iter()
        .chain(selfenergy.stderr.iter())
        .filter(|value| value.is_finite())
        .count();
    let total_selfenergy_bins = selfenergy.mean.len() + selfenergy.stderr.len();

    ValidationSummary {
        steps_done: output.stats.steps_done,
        cycles_done: output.stats.cycles_done,
        sample_count: output.measurement.sample_count,
        chains: cfg.chains,
        energy,
        quasiparticle_weight,
        target_energy: TARGET_ENERGY,
        target_quasiparticle_weight: TARGET_Z,
        energy_delta: energy.mean - TARGET_ENERGY,
        quasiparticle_weight_delta: quasiparticle_weight.mean - TARGET_Z,
        zeroth_mean: output.measurement.zeroth.mean(),
        mean_order: output.measurement.order.mean(),
        finite_selfenergy_bins,
        total_selfenergy_bins,
        final_energy_estimate: output.measurement.energy_estimate,
        energy_estimate_history: output.measurement.energy_estimates.clone(),
        update_stats: output.update_stats.clone(),
    }
}

pub fn write_results(
    cfg: &RunConfig,
    output: &RunOutput,
    dir: impl AsRef<Path>,
) -> AppResult<ResultsManifest> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let summary = summarize_output(cfg, output);
    let selfenergy = output.measurement.jackknife_selfenergy();
    let exact = output.measurement.get_exact();
    let mut files = Vec::new();

    write_json(dir.join("config.json"), cfg)?;
    files.push("config.json".to_string());
    write_json(dir.join("summary.json"), &summary)?;
    files.push("summary.json".to_string());
    fs::write(dir.join("summary.txt"), summary.text())?;
    files.push("summary.txt".to_string());
    write_json(dir.join("raw_stats.json"), &output.measurement)?;
    files.push("raw_stats.json".to_string());
    write_json(dir.join("selfenergy.json"), &selfenergy)?;
    files.push("selfenergy.json".to_string());
    write_json(dir.join("selfenergy_exact.json"), &exact)?;
    files.push("selfenergy_exact.json".to_string());

    if let Ok(fft) = analyze_stats(&output.measurement, cfg.fft_beta) {
        write_json(dir.join("fft.json"), &fft)?;
        files.push("fft.json".to_string());
    }

    if let Some(payload) = checkpoint_from_output(cfg.clone(), output.clone()) {
        save_checkpoint(dir.join("checkpoint.json"), &payload)?;
        files.push("checkpoint.json".to_string());
    }

    let manifest = ResultsManifest { summary, files };
    write_json(dir.join("manifest.json"), &manifest)?;
    Ok(manifest)
}

fn write_json<T: serde::Serialize>(path: PathBuf, value: &T) -> AppResult<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

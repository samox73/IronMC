use std::time::{SystemTime, UNIX_EPOCH};

use rmc_frohlich::app::{
    checkpoint_from_output, load_checkpoint, run_from_config, save_checkpoint, write_results,
};
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::sanity;

fn small_config() -> RunConfig {
    RunConfig {
        num_bins: 30,
        n_batches: 4,
        max_steps: 100,
        warmup_steps: 0,
        steps_per_cycle: 5,
        cycles_per_check: 1_000,
        initial_self_consistent_period: 1_000,
        ..RunConfig::default()
    }
}

#[test]
fn single_run_produces_stats_and_final_state() {
    let cfg = small_config();
    let output = run_from_config(&cfg).unwrap();
    assert_eq!(output.stats.steps_done, cfg.max_steps);
    assert_eq!(output.update_stats.len(), 8);
    assert_eq!(
        output
            .update_stats
            .iter()
            .map(|row| row.proposed)
            .sum::<u64>(),
        output.stats.steps_done
    );
    assert_eq!(
        output.measurement.sample_count as u64,
        cfg.max_steps / cfg.steps_per_cycle
    );
    let final_state = output.final_state.as_ref().unwrap();
    sanity::check_sanity(final_state).unwrap();
}

#[test]
fn parallel_run_reduces_measurement_outputs() {
    let mut cfg = small_config();
    cfg.chains = 2;
    let output = run_from_config(&cfg).unwrap();
    assert_eq!(output.stats.steps_done, cfg.max_steps * cfg.chains);
    assert_eq!(output.update_stats.len(), 8);
    assert_eq!(
        output
            .update_stats
            .iter()
            .map(|row| row.proposed)
            .sum::<u64>(),
        output.stats.steps_done
    );
    assert_eq!(
        output.measurement.sample_count as u64,
        (cfg.max_steps / cfg.steps_per_cycle) * cfg.chains
    );
    assert!(output.final_state.is_none());
}

#[test]
fn parallel_warmup_run_reduces_measurement_outputs() {
    let mut cfg = small_config();
    cfg.chains = 2;
    cfg.warmup_steps = 25;
    let output = run_from_config(&cfg).unwrap();
    assert_eq!(output.stats.steps_done, cfg.max_steps * cfg.chains);
    assert_eq!(output.update_stats.len(), 8);
    assert_eq!(
        output
            .update_stats
            .iter()
            .map(|row| row.proposed)
            .sum::<u64>(),
        output.stats.steps_done
    );
    assert_eq!(
        output.measurement.sample_count as u64,
        (cfg.max_steps / cfg.steps_per_cycle) * cfg.chains
    );
    assert!(output.final_state.is_none());
}

#[test]
fn checkpoint_payload_round_trips_json() {
    let cfg = small_config();
    let output = run_from_config(&cfg).unwrap();
    let payload = checkpoint_from_output(cfg.clone(), output).unwrap();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("rmc-frohlich-checkpoint-{stamp}.json"));

    save_checkpoint(&path, &payload).unwrap();
    let loaded = load_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    assert_eq!(loaded.config, cfg);
    assert_eq!(loaded.stats, payload.stats);
    assert_eq!(
        loaded.measurement.sample_count,
        payload.measurement.sample_count
    );
    assert_eq!(loaded.diagram.order, payload.diagram.order);
    assert_eq!(
        loaded.diagram.vertex_count(),
        payload.diagram.vertex_count()
    );
    sanity::check_sanity(&loaded.diagram).unwrap();
}

#[test]
fn write_results_creates_summary_and_artifacts() {
    let cfg = small_config();
    let output = run_from_config(&cfg).unwrap();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rmc-frohlich-results-{stamp}"));

    let manifest = write_results(&cfg, &output, &dir).unwrap();
    assert!(dir.join("summary.txt").exists());
    assert!(dir.join("summary.json").exists());
    assert!(dir.join("raw_stats.json").exists());
    assert!(dir.join("selfenergy.json").exists());
    assert!(dir.join("manifest.json").exists());
    assert!(manifest.summary.energy.mean.is_finite() || manifest.summary.energy.mean.is_nan());
    assert_eq!(manifest.summary.update_stats.len(), 8);
    assert_eq!(
        manifest
            .summary
            .update_stats
            .iter()
            .map(|row| row.proposed)
            .sum::<u64>(),
        manifest.summary.steps_done
    );
    assert!(manifest.summary.text().contains("UPDATE STATISTICS"));
    assert!(manifest.files.iter().any(|file| file == "summary.txt"));

    std::fs::remove_dir_all(&dir).unwrap();
}

use rmc_frohlich::app::run_from_config;
use rmc_frohlich::config::RunConfig;
use rmc_frohlich::flat::batched::run_batched;

#[test]
fn batched_estimators_match_slotmap_sanity_window() {
    let cfg = RunConfig {
        chains: 4,
        max_steps: 2_000,
        steps_per_cycle: 5,
        n_batches: 4,
        num_bins: 50,
        initial_self_consistent_period: usize::MAX,
        ..RunConfig::default()
    };

    let slotmap = run_from_config(&cfg).unwrap();
    let batched = run_batched(&cfg, cfg.chains as usize, 2).unwrap();
    let slotmap_energy = slotmap.measurement.jackknife_energy();
    let batched_energy = batched.measurement.jackknife_energy();

    assert_eq!(
        batched.measurement.sample_count,
        slotmap.measurement.sample_count
    );
    assert_eq!(
        batched
            .update_stats
            .iter()
            .map(|row| row.proposed)
            .sum::<u64>(),
        batched.stats.steps_done
    );
    assert!(batched_energy.mean.is_finite());
    assert!(slotmap_energy.mean.is_finite());
    assert!(
        (batched_energy.mean - slotmap_energy.mean).abs() < 1.0,
        "batched energy {:?}, slotmap energy {:?}",
        batched_energy,
        slotmap_energy
    );
}

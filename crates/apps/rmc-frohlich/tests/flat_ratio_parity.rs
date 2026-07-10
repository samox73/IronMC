use proptest::prelude::*;
use rmc_core::mc::{Kernel, MetropolisKernel};
use rmc_core::random::{ChainId, SeedSource};
use rmc_frohlich::flat::updates as flat_updates;
use rmc_frohlich::flat::FlatDiagram;
use rmc_frohlich::updates as slot_updates;
use rmc_frohlich::updates::default_update_set;
use rmc_frohlich::updates::phonon as slot_phonon;
use rmc_frohlich::Diagram;

fn diagram_after(seed: u64, steps: usize) -> Diagram {
    let mut diagram = Diagram::default();
    let mut rng = SeedSource::new(seed).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(default_update_set().unwrap());
    for _ in 0..steps {
        kernel.step(&mut diagram, &mut rng).unwrap();
    }
    diagram
}

fn assert_close(label: &str, actual: f64, expected: f64) -> Result<(), TestCaseError> {
    prop_assert!(
        (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-10,
        "{label}: actual {actual}, expected {expected}"
    );
    Ok(())
}

fn compare_attempts(seed: u64, burn_steps: usize, attempt_seed: u64) -> Result<(), TestCaseError> {
    let diagram = diagram_after(seed, burn_steps);
    let flat = FlatDiagram::from_diagram(&diagram, 256)
        .ok_or_else(|| TestCaseError::reject("diagram exceeded flat test capacity"))?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "change_tau",
        flat_updates::ChangeTau::default().attempt(&flat, &mut flat_rng),
        slot_updates::ChangeTau::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "change_internal_tau",
        flat_updates::ChangeInternalTau::default().attempt(&flat, &mut flat_rng),
        slot_updates::ChangeInternalTau::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "add_phonon",
        flat_updates::AddPhonon::default().attempt(&flat, &mut flat_rng),
        slot_phonon::AddPhonon::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "remove_phonon",
        flat_updates::RemovePhonon::default().attempt(&flat, &mut flat_rng),
        slot_phonon::RemovePhonon::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "rescale_diagram",
        flat_updates::RescaleDiagram::default().attempt(&flat, &mut flat_rng),
        slot_updates::RescaleDiagram::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "change_q_modulus",
        flat_updates::ChangeQModulus::default().attempt(&flat, &mut flat_rng),
        slot_updates::ChangeQModulus::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "change_q_direction",
        flat_updates::ChangeQDirection::default().attempt(&flat, &mut flat_rng),
        slot_updates::ChangeQDirection::default().attempt(&diagram, &mut slot_rng),
    )?;

    let mut slot_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(attempt_seed).rng_for(ChainId(0));
    assert_close(
        "change_topology",
        flat_updates::ChangeTopology::default().attempt(&flat, &mut flat_rng),
        slot_updates::ChangeTopology::default().attempt(&diagram, &mut slot_rng),
    )?;

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn flat_update_attempts_match_slotmap(
        seed in any::<u64>(),
        burn_steps in 0usize..500,
        attempt_seed in any::<u64>(),
    ) {
        compare_attempts(seed, burn_steps, attempt_seed)?;
    }
}

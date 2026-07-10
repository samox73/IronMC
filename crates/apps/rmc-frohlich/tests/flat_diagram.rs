use nalgebra::Vector3;
use rmc_core::mc::{run_chain, Kernel, MetropolisKernel, NoopCallbacks, SimulationParams};
use rmc_core::random::{ChainId, SeedSource};
use rmc_frohlich::flat::updates::default_update_set as default_flat_update_set;
use rmc_frohlich::flat::{FlatDiagram, NULL};
use rmc_frohlich::measurement::PolaronMeasurement;
use rmc_frohlich::updates::default_update_set;
use rmc_frohlich::Diagram;

fn q(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

fn q3(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x, y, z]
}

fn assert_matches_diagram(flat: &FlatDiagram, diagram: &Diagram) {
    assert_matches_diagram_at(flat, diagram, None);
}

fn assert_matches_diagram_at(flat: &FlatDiagram, diagram: &Diagram, step: Option<usize>) {
    assert_eq!(flat.order, diagram.order);
    assert_eq!(flat.vertex_count(), diagram.vertex_count());
    let flat_slots = flat.ordered_slots();
    let keys = diagram.ordered_keys();
    assert_eq!(flat_slots.len(), keys.len());

    for (slot, key) in flat_slots.into_iter().zip(keys) {
        let slot = slot as usize;
        let vertex = diagram.v(key);
        assert!(
            (flat.tau[slot] - vertex.tau).abs() <= vertex.tau.abs().max(1.0) * 1.0e-10,
            "tau mismatch at step {step:?}, slot {slot}: flat {} diagram {}",
            flat.tau[slot],
            vertex.tau
        );
        assert!(
            (flat.p_out[slot][0] - vertex.p_out.x).abs() <= vertex.p_out.x.abs().max(1.0) * 1.0e-10
        );
        assert!(
            (flat.p_out[slot][1] - vertex.p_out.y).abs() <= vertex.p_out.y.abs().max(1.0) * 1.0e-10
        );
        assert!(
            (flat.p_out[slot][2] - vertex.p_out.z).abs() <= vertex.p_out.z.abs().max(1.0) * 1.0e-10
        );
        assert!((flat.q[slot][0] - vertex.q.x).abs() <= vertex.q.x.abs().max(1.0) * 1.0e-10);
        assert!((flat.q[slot][1] - vertex.q.y).abs() <= vertex.q.y.abs().max(1.0) * 1.0e-10);
        assert!((flat.q[slot][2] - vertex.q.z).abs() <= vertex.q.z.abs().max(1.0) * 1.0e-10);
        assert_eq!(flat.phonons_above[slot] as usize, vertex.phonons_above);
    }
}

#[test]
fn flat_zero_order_matches_slotmap_diagram() {
    let diagram = Diagram::default();
    let flat = FlatDiagram::from_diagram(&diagram, 256).unwrap();

    assert_eq!(flat.capacity(), 514);
    assert_eq!(flat.head, 0);
    assert_eq!(flat.tail, 1);
    assert_eq!(flat.link[flat.head as usize], NULL);
    assert_matches_diagram(&flat, &diagram);
}

#[test]
fn flat_fake_order_one_round_trips() {
    let mut diagram = Diagram::default();
    let mut flat = FlatDiagram::from_diagram(&diagram, 256).unwrap();

    diagram.set_to_fake_order_one(q(0.25, -0.1, 0.05));
    flat.set_to_fake_order_one(q3(0.25, -0.1, 0.05));
    assert_matches_diagram(&flat, &diagram);

    diagram.clear_fake_order_one();
    flat.clear_fake_order_one();
    assert_matches_diagram(&flat, &diagram);
}

#[test]
fn flat_insert_and_remove_arc_matches_slotmap_diagram() {
    let mut diagram = Diagram::default();
    diagram.set_to_fake_order_one(q(0.2, 0.0, 0.0));
    let mut flat = FlatDiagram::from_diagram(&diagram, 256).unwrap();

    let (left, right) = diagram.insert_arc(0.25, 0.75, q(0.1, -0.05, 0.2));
    let (flat_left, flat_right) = flat.insert_arc(0.25, 0.75, q3(0.1, -0.05, 0.2)).unwrap();
    assert_matches_diagram(&flat, &diagram);

    diagram.remove_arc(left, right);
    flat.remove_arc(flat_left, flat_right);
    assert_matches_diagram(&flat, &diagram);
}

#[test]
fn flat_rejects_capacity_overflow() {
    let mut flat = FlatDiagram::with_parameters(1.0, -1.1, 0.0, 30.0, 1.0, 0, 10_000, 0);
    flat.set_to_fake_order_one(q3(0.2, 0.0, 0.0));
    assert!(flat.insert_arc(0.25, 0.75, q3(0.1, -0.05, 0.2)).is_none());
}

#[test]
fn flat_updates_match_slotmap_lockstep_smoke() {
    let mut diagram = Diagram::default();
    let mut flat = FlatDiagram::from_diagram(&diagram, 256).unwrap();
    let mut diagram_rng = SeedSource::new(37).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(37).rng_for(ChainId(0));
    let mut diagram_kernel = MetropolisKernel::new(default_update_set().unwrap());
    let mut flat_kernel = MetropolisKernel::new(default_flat_update_set().unwrap());

    for step in 0..100_000 {
        let diagram_outcome = diagram_kernel.step(&mut diagram, &mut diagram_rng).unwrap();
        let flat_outcome = flat_kernel.step(&mut flat, &mut flat_rng).unwrap();
        assert_eq!(flat_outcome.update_index, diagram_outcome.update_index);
        assert_eq!(flat_outcome.accepted, diagram_outcome.accepted);
        assert_eq!(flat_outcome.impossible, diagram_outcome.impossible);
        assert!(
            (flat_outcome.probability - diagram_outcome.probability).abs()
                <= diagram_outcome.probability.abs().max(1.0) * 1.0e-10,
            "probability mismatch at step {step}: flat {} diagram {}",
            flat_outcome.probability,
            diagram_outcome.probability
        );
        assert_matches_diagram_at(&flat, &diagram, Some(step));
    }
}

#[test]
fn flat_run_chain_matches_slotmap_estimators() {
    let params = SimulationParams {
        max_steps: 10_000,
        steps_per_cycle: 5,
        cycles_per_check: 1_000_000,
    };
    let expected_samples = params.max_steps.div_ceil(params.steps_per_cycle) as usize;
    let diagram = Diagram::default();
    let flat = FlatDiagram::from_diagram(&diagram, 256).unwrap();
    let mut diagram_rng = SeedSource::new(41).rng_for(ChainId(0));
    let mut flat_rng = SeedSource::new(41).rng_for(ChainId(0));
    let mut diagram_kernel = MetropolisKernel::new(default_update_set().unwrap());
    let mut flat_kernel = MetropolisKernel::new(default_flat_update_set().unwrap());
    let diagram_measurement = PolaronMeasurement::new(
        50,
        30.0,
        8,
        expected_samples,
        -1.0168,
        usize::MAX,
        1.5,
        &diagram,
    );
    let flat_measurement = PolaronMeasurement::new_flat(
        50,
        30.0,
        8,
        expected_samples,
        -1.0168,
        usize::MAX,
        1.5,
        &flat,
    );

    let (_diagram, diagram_stats, diagram_measurement) = run_chain(
        diagram,
        &mut diagram_rng,
        &mut diagram_kernel,
        diagram_measurement,
        params,
        NoopCallbacks,
    )
    .unwrap();
    let (_flat, flat_stats, flat_measurement) = run_chain(
        flat,
        &mut flat_rng,
        &mut flat_kernel,
        flat_measurement,
        params,
        NoopCallbacks,
    )
    .unwrap();

    assert_eq!(flat_stats, diagram_stats);
    assert_eq!(
        flat_measurement.sample_count,
        diagram_measurement.sample_count
    );
    assert_eq!(
        flat_measurement.zeroth.total_count(),
        diagram_measurement.zeroth.total_count()
    );
    assert_estimate_close(
        "energy",
        flat_measurement.jackknife_energy().mean,
        diagram_measurement.jackknife_energy().mean,
    );
    assert_estimate_close(
        "z",
        flat_measurement.jackknife_quasiparticle_weight().mean,
        diagram_measurement.jackknife_quasiparticle_weight().mean,
    );
    for (bin, (flat, diagram)) in flat_measurement
        .get_exact()
        .into_iter()
        .zip(diagram_measurement.get_exact())
        .enumerate()
    {
        assert_estimate_close(&format!("exact bin {bin}"), flat, diagram);
    }
}

fn assert_estimate_close(label: &str, actual: f64, expected: f64) {
    if actual.is_nan() || expected.is_nan() {
        assert!(
            actual.is_nan() && expected.is_nan(),
            "{label}: actual {actual}, expected {expected}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= expected.abs().max(1.0) * 1.0e-8,
            "{label}: actual {actual}, expected {expected}"
        );
    }
}

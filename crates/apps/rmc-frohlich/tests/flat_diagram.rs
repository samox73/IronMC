use nalgebra::Vector3;
use rmc_frohlich::flat::{FlatDiagram, NULL};
use rmc_frohlich::Diagram;

fn q(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

fn q3(x: f64, y: f64, z: f64) -> [f64; 3] {
    [x, y, z]
}

fn assert_matches_diagram(flat: &FlatDiagram, diagram: &Diagram) {
    assert_eq!(flat.order, diagram.order);
    assert_eq!(flat.vertex_count(), diagram.vertex_count());
    let flat_slots = flat.ordered_slots();
    let keys = diagram.ordered_keys();
    assert_eq!(flat_slots.len(), keys.len());

    for (slot, key) in flat_slots.into_iter().zip(keys) {
        let slot = slot as usize;
        let vertex = diagram.v(key);
        assert!((flat.tau[slot] - vertex.tau).abs() < 1.0e-12);
        assert!((flat.p_out[slot][0] - vertex.p_out.x).abs() < 1.0e-12);
        assert!((flat.p_out[slot][1] - vertex.p_out.y).abs() < 1.0e-12);
        assert!((flat.p_out[slot][2] - vertex.p_out.z).abs() < 1.0e-12);
        assert!((flat.q[slot][0] - vertex.q.x).abs() < 1.0e-12);
        assert!((flat.q[slot][1] - vertex.q.y).abs() < 1.0e-12);
        assert!((flat.q[slot][2] - vertex.q.z).abs() < 1.0e-12);
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

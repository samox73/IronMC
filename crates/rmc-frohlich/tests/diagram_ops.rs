use nalgebra::Vector3;
use rmc_frohlich::sanity;
use rmc_frohlich::{Diagram, Vertex};
use slotmap::Key;

fn q(x: f64, y: f64, z: f64) -> Vector3<f64> {
    Vector3::new(x, y, z)
}

fn vertices(d: &Diagram) -> Vec<Vertex> {
    d.ordered_keys()
        .into_iter()
        .map(|k| d.v(k).clone())
        .collect()
}

#[test]
fn order_zero_diagram_has_bare_endpoints() {
    let d = Diagram::default();
    assert_eq!(d.order, 0);
    assert_eq!(d.vertex_count(), 2);
    assert_eq!(d.v(d.head).tau, 0.0);
    assert_eq!(d.v(d.tail).tau, d.start_tau);
    assert!(d.v(d.head).link.is_null());
    assert!(d.v(d.tail).link.is_null());
    sanity::check_sanity(&d).unwrap();
}

#[test]
fn fake_order_one_sector_is_sane() {
    let mut d = Diagram::default();
    d.set_to_fake_order_one(q(0.25, -0.1, 0.05));
    assert_eq!(d.order, 1);
    assert_eq!(d.vertex_count(), 2);
    assert_eq!(d.v(d.head).link, d.tail);
    assert_eq!(d.v(d.tail).link, d.head);
    assert!(d.is_outgoing(d.head));
    assert!(d.is_incoming(d.tail));
    sanity::check_sanity(&d).unwrap();
}

#[test]
fn insert_and_remove_internal_arc_round_trips() {
    let mut d = Diagram::default();
    d.set_to_fake_order_one(q(0.2, 0.0, 0.0));
    let before = vertices(&d);

    d.insert_arc(0.25, 0.75, q(0.1, -0.05, 0.2));
    assert_eq!(d.order, 2);
    assert_eq!(d.vertex_count(), 4);
    sanity::check_sanity(&d).unwrap();
    let keys = d.ordered_keys();
    assert_eq!(d.v(keys[1]).phonons_above, 1);
    assert_eq!(d.v(keys[2]).phonons_above, 1);

    let partner = d.v(keys[1]).link;
    d.remove_arc(keys[1], partner);
    sanity::check_sanity(&d).unwrap();
    assert_eq!(d.order, 1);
    let after = vertices(&d);
    assert_eq!(after.len(), before.len());
    for (lhs, rhs) in after.iter().zip(before.iter()) {
        assert!((lhs.tau - rhs.tau).abs() < 1.0e-12);
        assert!((lhs.p_out - rhs.p_out).norm() < 1.0e-12);
        assert!((lhs.q - rhs.q).norm() < 1.0e-12);
        assert_eq!(lhs.phonons_above, rhs.phonons_above);
    }
}

#[test]
fn p_mean_and_exact_estimator_are_finite() {
    let d = Diagram::from_arcs(
        1.0,
        -1.1,
        0.0,
        30.0,
        1.0,
        &[(0.0, 1.0, q(0.2, 0.0, 0.0)), (0.2, 0.8, q(0.1, 0.2, 0.0))],
    );
    sanity::check_sanity(&d).unwrap();

    let keys = d.ordered_keys();
    let mean = d.get_p_mean_range(keys[1], keys[2], q(0.0, 0.0, 0.0));
    assert!(mean.iter().all(|x| x.is_finite()));

    let (mean_between, end) = d.get_p_mean_between(0.25, 0.75, keys[1]);
    assert_eq!(end, keys[2]);
    assert!(mean_between.iter().all(|x| x.is_finite()));

    let estimator = d.exact_estimator(0.5);
    assert!(estimator.is_finite());
    assert!(estimator > 0.0);
}

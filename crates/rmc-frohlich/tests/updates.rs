use nalgebra::Vector3;
use rmc_core::mc::{Kernel, MetropolisKernel};
use rmc_core::random::{ChainId, SeedSource};
use rmc_frohlich::sanity;
use rmc_frohlich::updates::default_update_set;
use rmc_frohlich::updates::phonon::{AddPhonon, RemovePhonon};
use rmc_frohlich::Diagram;

#[test]
fn add_and_remove_updates_preserve_sanity() {
    let mut d = Diagram::default();
    let mut rng = SeedSource::new(11).rng_for(ChainId(0));
    let mut add = AddPhonon::default();

    let ratio = add.attempt(&d, &mut rng);
    assert!(ratio.is_finite());
    assert!(ratio >= 0.0);
    add.accept(&mut d);
    sanity::check_sanity(&d).unwrap();
    assert_eq!(d.order, 1);

    let mut remove = RemovePhonon::default();
    let ratio = remove.attempt(&d, &mut rng);
    assert!(ratio.is_finite());
    assert!(ratio >= 0.0);
    remove.accept(&mut d);
    sanity::check_sanity(&d).unwrap();
    assert_eq!(d.order, 0);
}

#[test]
fn weighted_update_set_short_run_preserves_sanity() {
    let mut d = Diagram::from_arcs(
        1.0,
        -1.1,
        0.0,
        30.0,
        1.0,
        &[
            (0.0, 1.0, Vector3::new(0.2, 0.1, 0.0)),
            (0.2, 0.8, Vector3::new(0.1, -0.2, 0.05)),
        ],
    );
    let mut rng = SeedSource::new(29).rng_for(ChainId(0));
    let updates = default_update_set().unwrap();
    let mut kernel = MetropolisKernel::new(updates);

    let mut accepted = 0;
    for _ in 0..2_000 {
        let outcome = kernel.step(&mut d, &mut rng).unwrap();
        accepted += usize::from(outcome.accepted);
        sanity::check_sanity(&d).unwrap();
    }

    assert!(accepted > 0);
    assert!(d.order <= d.max_order);
}

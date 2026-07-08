#![cfg(feature = "serde")]

use rand_core::RngCore;
use rmc_core::mc::{
    MetropolisKernel, SimulationCtx, SimulationParams, SimulationStats, SingleUpdateSet,
    StepOutcome, TwoUpdateSet, Update, UpdateSet, UpdateStats, WeightedUpdate, WeightedUpdateSet,
};
use rmc_core::random::{ChainId, DefaultRng, SeedSource};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

fn round_trip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_string(value).unwrap();
    serde_json::from_str(&encoded).unwrap()
}

#[test]
fn serde_round_trips_seed_source_and_rng_state() {
    let seed = SeedSource::new(12345);
    let restored_seed: SeedSource = round_trip(&seed);

    assert_eq!(restored_seed, seed);
    assert_eq!(
        restored_seed.seed_for(ChainId(7)),
        seed.seed_for(ChainId(7))
    );

    let mut rng = seed.rng_for(ChainId(3));
    let _ = rng.next_u64();
    let mut restored_rng: DefaultRng = round_trip(&rng);

    assert_eq!(restored_rng.next_u64(), rng.next_u64());
}

#[test]
fn serde_round_trips_run_configuration_and_metadata() {
    let params = SimulationParams {
        max_steps: 100,
        steps_per_cycle: 5,
        cycles_per_check: 2,
    };
    let stats = SimulationStats {
        steps_done: 75,
        cycles_done: 15,
    };
    let ctx = SimulationCtx {
        steps_done: 75,
        cycles_done: 15,
        steps_in_cycle: 0,
    };
    let update_stats = UpdateStats {
        nprops: 10,
        naccs: 7,
        nimps: 1,
    };
    let outcome = StepOutcome {
        update_index: 2,
        probability: 0.25,
        accepted: true,
        impossible: false,
    };

    assert_eq!(round_trip::<SimulationParams>(&params), params);
    assert_eq!(round_trip::<SimulationStats>(&stats), stats);
    assert_eq!(round_trip::<SimulationCtx>(&ctx), ctx);
    assert_eq!(round_trip::<UpdateStats>(&update_stats), update_stats);
    assert_eq!(round_trip::<StepOutcome>(&outcome), outcome);
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SerializableUpdate {
    delta: i64,
}

impl Update<i64> for SerializableUpdate {
    fn attempt<R: rmc_core::random::Rng + ?Sized>(
        &mut self,
        _state: &mut i64,
        _rng: &mut R,
    ) -> f64 {
        1.0
    }

    fn accept(&mut self, state: &mut i64) {
        *state += self.delta;
    }
}

#[test]
fn serde_round_trips_static_update_sets() {
    let single = SingleUpdateSet::new(SerializableUpdate { delta: 3 });
    let restored_single: SingleUpdateSet<SerializableUpdate> = round_trip(&single);
    assert_eq!(restored_single.update().delta, 3);
    assert_eq!(restored_single.stats(), &[UpdateStats::default()]);

    let two = TwoUpdateSet::with_ratios(
        SerializableUpdate { delta: 1 },
        2.0,
        3.0,
        SerializableUpdate { delta: -1 },
        4.0,
        5.0,
    )
    .unwrap();
    let restored_two: TwoUpdateSet<SerializableUpdate, SerializableUpdate> = round_trip(&two);
    assert_eq!(restored_two.first().delta, 1);
    assert_eq!(restored_two.second().delta, -1);
    assert_eq!(restored_two.weights(), [2.0, 4.0]);
    assert_eq!(restored_two.ratios(), [3.0, 5.0]);

    let weighted = WeightedUpdateSet::new(vec![
        WeightedUpdate::new(SerializableUpdate { delta: 1 }, 2.0),
        WeightedUpdate::with_ratio(SerializableUpdate { delta: -1 }, 4.0, 0.5),
    ])
    .unwrap();
    let mut restored_weighted: WeightedUpdateSet<SerializableUpdate> = round_trip(&weighted);
    assert_eq!(restored_weighted.weights(), vec![2.0, 4.0]);
    assert_eq!(restored_weighted.ratios(), vec![1.0, 0.5]);
    restored_weighted.rebuild_distribution().unwrap();
}

#[test]
fn serde_round_trips_static_kernel() {
    let kernel = MetropolisKernel::new(SingleUpdateSet::new(SerializableUpdate { delta: 2 }));
    let restored: MetropolisKernel<SingleUpdateSet<SerializableUpdate>> = round_trip(&kernel);

    assert_eq!(restored.updates().update().delta, 2);
}

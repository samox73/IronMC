use rand_core::RngCore;
use rmc_core::mc::ResultSink;
use rmc_core::{
    mc::{
        run_chain, Measurement, MetropolisKernel, SimulationParams, SingleUpdateSet, Update,
        UpdateSet,
    },
    random::{ChainId, DefaultRng, Rng, SeedSource},
};
use rmc_io::{
    from_binary_slice, from_json_str, load_binary, load_json, load_payload_binary,
    load_payload_json, save_binary, save_binary_atomic, save_json, save_json_atomic,
    save_payload_binary, save_payload_json, to_binary_vec, to_json_string, Checkpoint, IoError,
    MapSink, CHECKPOINT_VERSION,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Payload {
    step: u64,
    state: i64,
}

#[test]
fn checkpoint_round_trips_through_json_string_and_file() {
    let checkpoint = Checkpoint::new(Payload { step: 7, state: -3 });
    let encoded = to_json_string(&checkpoint).unwrap();
    let decoded: Checkpoint<Payload> = from_json_str(&encoded).unwrap();

    assert_eq!(decoded, checkpoint);
    assert_eq!(decoded.version, CHECKPOINT_VERSION);
    assert_eq!(decoded.payload().state, -3);

    let path =
        std::env::temp_dir().join(format!("rmc-checkpoint-json-{}.json", std::process::id()));
    save_json(&path, &checkpoint).unwrap();
    let loaded: Checkpoint<Payload> = load_json(&path).unwrap();
    assert_eq!(loaded, checkpoint);

    let payload_path = std::env::temp_dir().join(format!(
        "rmc-checkpoint-payload-{}.json",
        std::process::id()
    ));
    save_payload_json(&payload_path, checkpoint.payload()).unwrap();
    let payload: Payload = load_payload_json(&payload_path).unwrap();
    assert_eq!(payload, *checkpoint.payload());

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(payload_path);
}

#[test]
fn checkpoint_round_trips_through_binary_bytes_and_file() {
    let checkpoint = Checkpoint::new(Payload { step: 9, state: 42 });
    let encoded = to_binary_vec(&checkpoint).unwrap();
    let decoded: Checkpoint<Payload> = from_binary_slice(&encoded).unwrap();

    assert_eq!(decoded, checkpoint);
    assert_eq!(decoded.version, CHECKPOINT_VERSION);
    assert_eq!(decoded.payload().state, 42);

    let path =
        std::env::temp_dir().join(format!("rmc-checkpoint-binary-{}.bin", std::process::id()));
    save_binary(&path, &checkpoint).unwrap();
    let loaded: Checkpoint<Payload> = load_binary(&path).unwrap();
    assert_eq!(loaded, checkpoint);

    let payload_path =
        std::env::temp_dir().join(format!("rmc-checkpoint-payload-{}.bin", std::process::id()));
    save_payload_binary(&payload_path, checkpoint.payload()).unwrap();
    let payload: Payload = load_payload_binary(&payload_path).unwrap();
    assert_eq!(payload, *checkpoint.payload());

    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file(payload_path);
}

#[test]
fn load_rejects_unsupported_checkpoint_version() {
    let err =
        from_json_str::<Payload>(r#"{"version":999,"payload":{"step":1,"state":2}}"#).unwrap_err();

    assert!(matches!(
        err,
        IoError::UnsupportedVersion {
            expected: CHECKPOINT_VERSION,
            found: 999
        }
    ));

    let encoded = to_binary_vec(&Checkpoint {
        version: 999,
        payload: Payload { step: 1, state: 2 },
    })
    .unwrap();
    let err = from_binary_slice::<Payload>(&encoded).unwrap_err();

    assert!(matches!(
        err,
        IoError::UnsupportedVersion {
            expected: CHECKPOINT_VERSION,
            found: 999
        }
    ));
}

#[test]
fn binary_load_reports_invalid_bytes() {
    let err = from_binary_slice::<Payload>(b"").unwrap_err();

    assert!(matches!(err, IoError::Binary(_)));
}

#[test]
fn map_sink_rejects_duplicate_paths() {
    let mut sink = MapSink::new();
    sink.put("energy/moments", &vec![1_i64, 2, 3]).unwrap();
    let err = sink.put("energy/moments", &4_i64).unwrap_err();

    assert!(matches!(err, rmc_core::RmcError::DuplicateResult(path) if path == "energy/moments"));
}

#[test]
fn map_sink_round_trips_as_checkpoint_payload() {
    let mut sink = MapSink::new();
    sink.put("energy/moments", &vec![1_i64, 2, 3]).unwrap();
    sink.put("mag/mean", &0.25_f64).unwrap();

    let checkpoint = sink.clone().into_checkpoint();
    let json = to_json_string(&checkpoint).unwrap();
    let decoded_json: Checkpoint<rmc_io::ResultMap> = from_json_str(&json).unwrap();
    assert_eq!(MapSink::from_checkpoint(decoded_json).unwrap(), sink);

    let encoded_checkpoint = sink.to_encoded_checkpoint().unwrap();
    let binary = to_binary_vec(&encoded_checkpoint).unwrap();
    let decoded_binary: Checkpoint<rmc_io::EncodedResultMap> = from_binary_slice(&binary).unwrap();
    assert_eq!(
        MapSink::from_encoded_checkpoint(decoded_binary).unwrap(),
        sink
    );
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RandomWalkUpdate;

impl Update<i64> for RandomWalkUpdate {
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut i64, rng: &mut R) -> f64 {
        if rng.gen_bool(0.5) {
            *state += 1;
        } else {
            *state -= 1;
        }
        1.0
    }

    fn accept(&mut self, _state: &mut i64) {}
}

#[derive(Clone, Debug, Default)]
struct LastStateValue(i64);

#[derive(Serialize, Deserialize)]
struct RestartPayload {
    state: i64,
    rng: DefaultRng,
    kernel: MetropolisKernel<SingleUpdateSet<RandomWalkUpdate>>,
}

#[test]
fn checkpointed_run_resumes_same_trajectory_as_uninterrupted_run() {
    let seed = SeedSource::new(0x7eed);
    let full_params = SimulationParams {
        max_steps: 40,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let first_params = SimulationParams {
        max_steps: 13,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let second_params = SimulationParams {
        max_steps: 27,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };

    let mut full_rng = seed.rng_for(ChainId(0));
    let mut full_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (full_state, _full_stats, full_last) = run_chain(
        0,
        &mut full_rng,
        &mut full_kernel,
        LastStateValue::default(),
        full_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    let mut split_rng = seed.rng_for(ChainId(0));
    let mut split_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (split_state, _first_stats, _first_last) = run_chain(
        0,
        &mut split_rng,
        &mut split_kernel,
        LastStateValue::default(),
        first_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    let checkpoint = Checkpoint::new(RestartPayload {
        state: split_state,
        rng: split_rng,
        kernel: split_kernel,
    });
    let path = std::env::temp_dir().join(format!("rmc-restart-{}.json", std::process::id()));
    save_json_atomic(&path, &checkpoint).unwrap();

    let restored: Checkpoint<RestartPayload> = load_json(&path).unwrap();
    let RestartPayload {
        state,
        mut rng,
        mut kernel,
    } = restored.into_payload();
    let (resumed_state, _second_stats, resumed_last) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        LastStateValue::default(),
        second_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    assert_eq!(resumed_state, full_state);
    assert_eq!(resumed_last, full_last);
    assert_eq!(rng.next_u64(), full_rng.next_u64());
    assert_eq!(kernel.updates().stats(), full_kernel.updates().stats());

    let _ = std::fs::remove_file(path);
}

#[test]
fn binary_checkpointed_run_resumes_same_trajectory_as_uninterrupted_run() {
    let seed = SeedSource::new(0x7eed);
    let full_params = SimulationParams {
        max_steps: 40,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let first_params = SimulationParams {
        max_steps: 13,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };
    let second_params = SimulationParams {
        max_steps: 27,
        steps_per_cycle: 4,
        cycles_per_check: 1,
    };

    let mut full_rng = seed.rng_for(ChainId(0));
    let mut full_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (full_state, _full_stats, full_last) = run_chain(
        0,
        &mut full_rng,
        &mut full_kernel,
        LastStateValue::default(),
        full_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    let mut split_rng = seed.rng_for(ChainId(0));
    let mut split_kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate));
    let (split_state, _first_stats, _first_last) = run_chain(
        0,
        &mut split_rng,
        &mut split_kernel,
        LastStateValue::default(),
        first_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    let checkpoint = Checkpoint::new(RestartPayload {
        state: split_state,
        rng: split_rng,
        kernel: split_kernel,
    });
    let path = std::env::temp_dir().join(format!("rmc-restart-{}.bin", std::process::id()));
    save_binary_atomic(&path, &checkpoint).unwrap();

    let restored: Checkpoint<RestartPayload> = load_binary(&path).unwrap();
    let RestartPayload {
        state,
        mut rng,
        mut kernel,
    } = restored.into_payload();
    let (resumed_state, _second_stats, resumed_last) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        LastStateValue::default(),
        second_params,
        rmc_core::mc::NoopCallbacks,
    )
    .unwrap();

    assert_eq!(resumed_state, full_state);
    assert_eq!(resumed_last, full_last);
    assert_eq!(rng.next_u64(), full_rng.next_u64());
    assert_eq!(kernel.updates().stats(), full_kernel.updates().stats());

    let _ = std::fs::remove_file(path);
}

impl Measurement<i64> for LastStateValue {
    type Output = i64;

    fn measure(&mut self, state: &i64) {
        self.0 = *state;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}

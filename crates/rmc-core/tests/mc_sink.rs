use std::collections::BTreeMap;

use rmc_core::mc::{
    run_with_sink, MetropolisKernel, ResultSink, SimulationParams, SingleUpdateSet,
    SinkMeasurement, SinkMeasurementSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::{Result, RmcError};

#[derive(Default)]
struct TestSink {
    values: BTreeMap<String, serde_json::Value>,
}

impl ResultSink for TestSink {
    fn put(&mut self, path: &str, value: &dyn erased_serde::Serialize) -> Result<()> {
        if self.values.contains_key(path) {
            return Err(RmcError::DuplicateResult(path.to_string()));
        }
        let value = erased_serde::serialize(value, serde_json::value::Serializer)
            .map_err(|err| RmcError::Message(err.to_string()))?;
        self.values.insert(path.to_string(), value);
        Ok(())
    }
}

struct ConstantMeasurement {
    name: &'static str,
    key: &'static str,
    value: i64,
}

impl SinkMeasurement<i64> for ConstantMeasurement {
    fn name(&self) -> &str {
        self.name
    }

    fn measure(&mut self, _state: &i64) {}

    fn write_result(&self, sink: &mut dyn ResultSink) -> Result<()> {
        sink.put(self.key, &self.value)
    }
}

struct DuplicateKeyMeasurement;

impl SinkMeasurement<i64> for DuplicateKeyMeasurement {
    fn name(&self) -> &str {
        "dup"
    }

    fn measure(&mut self, _state: &i64) {}

    fn write_result(&self, sink: &mut dyn ResultSink) -> Result<()> {
        sink.put("moments", &1_i64)?;
        sink.put("moments", &2_i64)
    }
}

#[test]
fn sink_measurement_set_namespaces_results() {
    let mut measurements = SinkMeasurementSet::new();
    measurements
        .add(ConstantMeasurement {
            name: "energy",
            key: "moments",
            value: 4,
        })
        .unwrap();
    measurements
        .add(ConstantMeasurement {
            name: "mag",
            key: "moments",
            value: 9,
        })
        .unwrap();
    measurements.refresh_active();

    let mut sink = TestSink::default();
    measurements.measure_all(&0);
    measurements.write_all(&mut sink).unwrap();

    assert_eq!(sink.values["energy/moments"], serde_json::json!(4));
    assert_eq!(sink.values["mag/moments"], serde_json::json!(9));
}

#[test]
fn sink_measurement_set_rejects_duplicate_measurement_names() {
    let mut measurements = SinkMeasurementSet::new();
    measurements
        .add(ConstantMeasurement {
            name: "energy",
            key: "moments",
            value: 4,
        })
        .unwrap();
    let err = measurements
        .add(ConstantMeasurement {
            name: "energy",
            key: "other",
            value: 5,
        })
        .unwrap_err();

    assert!(matches!(err, RmcError::InvalidArgument(_)));
}

#[test]
fn sink_measurement_set_rejects_duplicate_keys_within_measurement() {
    let mut measurements = SinkMeasurementSet::new();
    measurements.add(DuplicateKeyMeasurement).unwrap();
    measurements.refresh_active();

    let mut sink = TestSink::default();
    let err = measurements.write_all(&mut sink).unwrap_err();

    assert!(matches!(err, RmcError::DuplicateResult(path) if path == "dup/moments"));
}

#[derive(Clone)]
struct Increment;

impl Update<i64> for Increment {
    fn attempt<R: rand::Rng + ?Sized>(&mut self, state: &mut i64, _rng: &mut R) -> f64 {
        *state += 1;
        1.0
    }

    fn accept(&mut self, _state: &mut i64) {}
}

#[derive(Default)]
struct LastState {
    samples: u64,
    last: i64,
}

impl SinkMeasurement<i64> for LastState {
    fn name(&self) -> &str {
        "state"
    }

    fn measure(&mut self, state: &i64) {
        self.samples += 1;
        self.last = *state;
    }

    fn write_result(&self, sink: &mut dyn ResultSink) -> Result<()> {
        sink.put("samples", &self.samples)?;
        sink.put("last", &self.last)
    }
}

#[test]
fn run_with_sink_writes_results_after_run() {
    let mut rng = SeedSource::new(7).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(SingleUpdateSet::new(Increment));
    let mut measurements = SinkMeasurementSet::new();
    measurements.add(LastState::default()).unwrap();
    let mut sink = TestSink::default();

    let (state, stats) = run_with_sink(
        0_i64,
        &mut rng,
        &mut kernel,
        &mut measurements,
        &mut sink,
        SimulationParams {
            max_steps: 5,
            steps_per_cycle: 2,
            cycles_per_check: 1,
        },
    )
    .unwrap();

    assert_eq!(state, 5);
    assert_eq!(stats.cycles_done, 3);
    assert_eq!(sink.values["state/samples"], serde_json::json!(3));
    assert_eq!(sink.values["state/last"], serde_json::json!(5));
}

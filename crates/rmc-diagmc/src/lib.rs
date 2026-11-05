//! Model-agnostic imaginary-time diagrammatic Monte Carlo.

pub mod diagram;
pub mod measure;
pub mod model;
pub mod run;
pub mod updates;
pub mod weight;

pub use diagram::{Diagram, Vertex};
pub use measure::{DiagMeasurement, Estimate, GreenFunction, MeasurementData, Report, summarize};
pub use model::{FrohlichModel, Model};
pub use run::{DiagConfig, SlotmapLeg, run, run_leg, run_seeds};
pub use updates::{DiagUpdate, update_set, update_set_fixed_tau};

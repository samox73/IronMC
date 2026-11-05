//! Fröhlich-polaron self-energy Diagrammatic Monte Carlo port.

// Physics and measurement helpers keep flat scalar argument lists so the GPU kernel can mirror them.
#![allow(clippy::too_many_arguments)]

pub mod app;
pub mod config;
pub mod diagram;
pub mod flat;
pub mod fourier;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod measurement;
pub mod physics;
pub mod sanity;
pub mod update_stats;
pub mod updates;

pub use diagram::{Diagram, Vertex, norm0};

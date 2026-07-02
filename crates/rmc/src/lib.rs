//! Facade crate for the greenfield Rust Monte Carlo framework.
//!
//! `rmc` re-exports the lean engine crate and opt-in batteries behind feature flags.

pub use rmc_core::*;

#[cfg(feature = "grids")]
pub mod grids {
    pub use rmc_grids::*;
}

#[cfg(feature = "stats")]
pub mod stats {
    pub use rmc_stats::*;
}

#[cfg(feature = "numeric")]
pub mod numeric {
    pub use rmc_numeric::*;
}

#[cfg(feature = "io")]
pub mod io {
    pub use rmc_io::*;
}

//! Polaron Monte Carlo updates.

pub mod phonon;

use std::f64::consts::PI;

use nalgebra::{Rotation3, Vector3};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rmc_core::mc::{Update, WeightedUpdate, WeightedUpdateSet};
use rmc_core::random::{safe_exponential_sample, uniform_index};
use rmc_core::Result;
use slotmap::Key;

use crate::diagram::VKey;
use crate::diagram::{phi_from_cartesian, spherical_to_cartesian, theta_from_cartesian, Diagram};
use phonon::{AddPhonon, RemovePhonon};

#[derive(Clone, Debug)]
pub enum PolaronUpdate {
    ChangeTau(ChangeTau),
    ChangeInternalTau(ChangeInternalTau),
    AddPhonon(AddPhonon),
    RemovePhonon(RemovePhonon),
    RescaleDiagram(RescaleDiagram),
    ChangeQModulus(ChangeQModulus),
    ChangeQDirection(ChangeQDirection),
    ChangeTopology(ChangeTopology),
}

impl PolaronUpdate {
    pub fn name(&self) -> &'static str {
        match self {
            Self::ChangeTau(_) => "change_tau",
            Self::ChangeInternalTau(_) => "change_internal_tau",
            Self::AddPhonon(_) => "add_phonon",
            Self::RemovePhonon(_) => "remove_phonon",
            Self::RescaleDiagram(_) => "rescale_diagram",
            Self::ChangeQModulus(_) => "change_internal_q_modulus",
            Self::ChangeQDirection(_) => "change_internal_q_direction",
            Self::ChangeTopology(_) => "change_topology",
        }
    }
}

impl Update<Diagram> for PolaronUpdate {
    fn attempt<R: Rng + ?Sized>(&mut self, state: &mut Diagram, rng: &mut R) -> f64 {
        match self {
            Self::ChangeTau(update) => update.attempt(state, rng),
            Self::ChangeInternalTau(update) => update.attempt(state, rng),
            Self::AddPhonon(update) => update.attempt(state, rng),
            Self::RemovePhonon(update) => update.attempt(state, rng),
            Self::RescaleDiagram(update) => update.attempt(state, rng),
            Self::ChangeQModulus(update) => update.attempt(state, rng),
            Self::ChangeQDirection(update) => update.attempt(state, rng),
            Self::ChangeTopology(update) => update.attempt(state, rng),
        }
    }

    fn accept(&mut self, state: &mut Diagram) {
        match self {
            Self::ChangeTau(update) => update.accept(state),
            Self::ChangeInternalTau(update) => update.accept(state),
            Self::AddPhonon(update) => update.accept(state),
            Self::RemovePhonon(update) => update.accept(state),
            Self::RescaleDiagram(update) => update.accept(state),
            Self::ChangeQModulus(update) => update.accept(state),
            Self::ChangeQDirection(update) => update.accept(state),
            Self::ChangeTopology(update) => update.accept(state),
        }
    }
}

pub fn default_update_set() -> Result<WeightedUpdateSet<PolaronUpdate>> {
    WeightedUpdateSet::new(vec![
        WeightedUpdate::new(PolaronUpdate::ChangeTau(ChangeTau::default()), 1.0),
        WeightedUpdate::new(
            PolaronUpdate::ChangeInternalTau(ChangeInternalTau::default()),
            1.0,
        ),
        WeightedUpdate::new(PolaronUpdate::AddPhonon(AddPhonon::default()), 1.0),
        WeightedUpdate::new(PolaronUpdate::RemovePhonon(RemovePhonon::default()), 1.0),
        WeightedUpdate::new(
            PolaronUpdate::RescaleDiagram(RescaleDiagram::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeQModulus(ChangeQModulus::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeQDirection(ChangeQDirection::default()),
            1.0,
        ),
        WeightedUpdate::new(
            PolaronUpdate::ChangeTopology(ChangeTopology::default()),
            1.0,
        ),
    ])
}

#[derive(Clone, Debug, Default)]
pub struct ChangeTau {
    tau_prime: f64,
}

impl ChangeTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        let second_last = d.prev(d.tail);
        let lambda =
            d.dispersion(&d.v(second_last).p_out) + if d.order != 0 { Diagram::OMEGA } else { 0.0 };
        self.tau_prime =
            safe_exponential_sample(rng.gen(), lambda, d.v(second_last).tau, d.max_tau);
        1.0
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.set_vertex_tau(d.tail, self.tau_prime);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeInternalTau {
    vertex: VKey,
    tau_prime: f64,
}

impl ChangeInternalTau {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }
        let mut vertex = random_vertex(d, rng);
        while vertex == d.head || vertex == d.tail {
            vertex = random_vertex(d, rng);
        }
        let prev = d.prev(vertex);
        let next = d.next(vertex);
        let tau_previous = d.v(prev).tau;
        let tau_next = if next.is_null() {
            d.max_tau
        } else {
            d.v(next).tau
        };
        let lambda = Diagram::bare_dispersion(&d.v(prev).p_out)
            - Diagram::bare_dispersion(&d.v(vertex).p_out)
            + if d.is_incoming(vertex) {
                Diagram::OMEGA
            } else {
                -Diagram::OMEGA
            };
        self.vertex = vertex;
        self.tau_prime = safe_exponential_sample(rng.gen(), lambda, tau_previous, tau_next);
        if self.tau_prime.is_finite() {
            1.0
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.set_vertex_tau(self.vertex, self.tau_prime);
        debug_assert!(d.v(d.prev(self.vertex)).tau < d.v(self.vertex).tau);
        let next = d.next(self.vertex);
        debug_assert!(next.is_null() || d.v(self.vertex).tau < d.v(next).tau);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

#[derive(Clone, Debug, Default)]
pub struct RescaleDiagram {
    tau_prime: f64,
}

impl RescaleDiagram {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order <= 1 {
            return 0.0;
        }

        let mut energy = -d.mu;
        let mut k = d.head;
        while k != d.tail {
            let next = d.next(k);
            let energy_i = Diagram::bare_dispersion(&d.v(k).p_out);
            let delta_s_i = (d.v(next).tau - d.v(k).tau) / d.tau();
            let phonon_count = if d.is_incoming(k) {
                d.v(k).phonons_above
            } else {
                d.v(k).phonons_above + 1
            };
            energy += delta_s_i * (energy_i + Diagram::OMEGA * phonon_count as f64);
            k = next;
        }

        let n = (d.order - 1) as f64;
        let Ok(normal) = Normal::new(2.0 * n / energy, (2.0 * n).sqrt() / energy) else {
            return 0.0;
        };
        self.tau_prime = normal.sample(rng);
        if self.tau_prime < 0.0 || self.tau_prime > d.max_tau || !self.tau_prime.is_finite() {
            return 0.0;
        }

        let acceptance = (2.0 * n * (self.tau_prime / d.tau()).ln()
            - energy * (self.tau_prime - d.tau())
            + ((energy * self.tau_prime - 2.0 * n).powi(2) - (energy * d.tau() - 2.0 * n).powi(2))
                / (4.0 * n))
            .exp();
        if acceptance.is_finite() {
            acceptance
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.scale_taus(self.tau_prime / d.tau());
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeQModulus {
    vertex1: VKey,
    vertex2: VKey,
    q_prime: f64,
}

impl ChangeQModulus {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let q_norm = d.v(left).q.norm();
        if q_norm == 0.0 {
            return 0.0;
        }
        let q0 = d
            .get_p_mean_range(left, right, d.v(left).q)
            .dot(&(d.v(left).q / q_norm));
        let sigma = (Diagram::MASS / (d.v(right).tau - d.v(left).tau)).sqrt();
        let Ok(normal) = Normal::new(q0, sigma) else {
            return 0.0;
        };
        self.q_prime = normal.sample(rng);
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime < 0.0 || !self.q_prime.is_finite() {
            0.0
        } else {
            1.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        let new_q = d.v(self.vertex1).q / d.v(self.vertex1).q.norm() * self.q_prime;
        d.update_arc_q(self.vertex1, self.vertex2, new_q);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeQDirection {
    vertex1: VKey,
    vertex2: VKey,
    q_prime: Vector3<f64>,
}

impl ChangeQDirection {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let (left, right) = random_arc(d, rng);
        let p_mean = d.get_p_mean_range(left, right, d.v(left).q);
        let q_norm = d.v(left).q.norm();
        let a = (d.v(right).tau - d.v(left).tau) * p_mean.norm() * q_norm / Diagram::MASS;
        if a.abs() < f64::EPSILON || !a.is_finite() {
            return 0.0;
        }

        let phi = 2.0 * PI * rng.gen::<f64>();
        let log_val = (-2.0 * rng.gen::<f64>() * a.sinh() + a.exp()).ln();
        let theta = (log_val / a).acos();
        let theta_base = theta_from_cartesian(&p_mean);
        let phi_base = phi_from_cartesian(&p_mean);
        let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), phi_base)
            * Rotation3::from_axis_angle(&Vector3::y_axis(), theta_base);
        self.q_prime = rotation * spherical_to_cartesian(q_norm, theta, phi);
        self.vertex1 = left;
        self.vertex2 = right;
        if self.q_prime.iter().any(|x| x.is_nan()) {
            0.0
        } else {
            1.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        d.update_arc_q(self.vertex1, self.vertex2, self.q_prime);
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChangeTopology {
    vertex1: VKey,
    vertex2: VKey,
    p_prime: Vector3<f64>,
}

impl ChangeTopology {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            return 0.0;
        }
        let mut vertex1 = random_vertex(d, rng);
        if d.next(vertex1).is_null() {
            vertex1 = d.prev(vertex1);
        }
        let vertex2 = d.next(vertex1);
        let c1 = if d.is_outgoing(vertex1) { 1.0 } else { -1.0 };
        let c2 = if d.is_outgoing(vertex2) { 1.0 } else { -1.0 };
        if d.v(vertex1).link == vertex2 {
            return 0.0;
        }
        self.p_prime = d.v(vertex1).p_out + d.v(vertex1).q * c1 - d.v(vertex2).q * c2;

        if d.is_outgoing(vertex1)
            && d.is_incoming(vertex2)
            && d.v(vertex1).link != vertex2
            && (d.v(vertex1).phonons_above == 1 || d.v(vertex2).phonons_above == 1)
        {
            return 0.0;
        }

        self.vertex1 = vertex1;
        self.vertex2 = vertex2;
        let acceptance = (-(d.v(vertex2).tau - d.v(vertex1).tau)
            * (Diagram::bare_dispersion(&self.p_prime)
                - Diagram::bare_dispersion(&d.v(vertex1).p_out)
                - Diagram::OMEGA * (c1 - c2)))
            .exp();
        if acceptance.is_finite() {
            acceptance
        } else {
            0.0
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.is_incoming(self.vertex1) && d.is_outgoing(self.vertex2) {
            d.vm(self.vertex1).phonons_above += 1;
            d.vm(self.vertex2).phonons_above += 1;
        }
        if d.is_outgoing(self.vertex1)
            && d.is_incoming(self.vertex2)
            && d.v(self.vertex1).link != self.vertex2
        {
            d.vm(self.vertex1).phonons_above -= 1;
            d.vm(self.vertex2).phonons_above -= 1;
        }
        d.vm(self.vertex1).p_out = self.p_prime;
        if d.v(self.vertex1).link != self.vertex2 {
            d.swap_arc_connectivity(self.vertex1, self.vertex2);
        }
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }
}

fn random_vertex<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> VKey {
    d.storage[uniform_index(rng, d.storage.len())]
}

fn random_arc<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> (VKey, VKey) {
    let vertex1 = random_vertex(d, rng);
    let vertex2 = d.v(vertex1).link;
    if d.v(vertex2).tau < d.v(vertex1).tau {
        (vertex2, vertex1)
    } else {
        (vertex1, vertex2)
    }
}

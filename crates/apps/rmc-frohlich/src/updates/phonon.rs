use std::f64::consts::PI;

use nalgebra::Vector3;
use rand::Rng;
use rmc_core::random::{exponential_sample_bounded, uniform_index};
use slotmap::Key;

use crate::diagram::{draw_new_q_from_uniforms, Diagram, VKey};

/// Inserts a new phonon arc (pair of linked vertices) at a random time interval.
#[derive(Clone, Debug, Default)]
pub struct AddPhonon {
    vertex1: VKey,
    vertex2: VKey,
    tau1: f64,
    tau2: f64,
    q: Vector3<f64>,
}

impl AddPhonon {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == 0 {
            self.attempt_from_zero_fake(d, rng)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.order == 0 {
            d.set_to_fake_order_one(self.q);
        } else {
            d.insert_arc_between(self.vertex1, self.vertex2, self.tau1, self.tau2, self.q);
        }
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }

    fn attempt_from_zero_fake<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        self.q = draw_new_q(rng);
        2.0 * d.alpha * Diagram::OMEGA.powi(2) / PI
            * (-(Diagram::OMEGA
                + (self.q.norm_squared() / 2.0 - self.q.dot(&d.momentum_out())) / Diagram::MASS)
                * d.tau())
            .exp()
            * (1.0 + self.q.norm() / Diagram::p0()).powi(2)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order >= d.max_order {
            return 0.0;
        }

        let mut vertex1 = random_vertex(d, rng);
        while d.next(vertex1).is_null() {
            vertex1 = random_vertex(d, rng);
        }

        let next1 = d.next(vertex1);
        let delta_t = d.v(next1).tau - d.v(vertex1).tau;
        self.tau1 = d.v(vertex1).tau + rng.gen::<f64>() * delta_t;
        if self.tau1 - d.v(vertex1).tau < Diagram::DELTA_TAU_LIMIT
            || d.v(next1).tau - self.tau1 < Diagram::DELTA_TAU_LIMIT
        {
            return 0.0;
        }

        self.q = draw_new_q(rng);
        let lambda = Diagram::OMEGA * (1.0 + self.q.norm() / Diagram::p0()).powi(2);
        self.tau2 = exponential_sample_bounded(rng.gen(), lambda, self.tau1, d.max_tau);
        if self.tau2 - self.tau1 < Diagram::DELTA_TAU_LIMIT {
            return 0.0;
        }

        let (p_mean, vertex2) = d.get_p_mean_between(self.tau1, self.tau2, vertex1);
        let prev2 = if vertex2.is_null() {
            d.tail
        } else {
            d.prev(vertex2)
        };
        let prev_tau = d.v(prev2).tau;
        if self.tau2 - prev_tau < Diagram::DELTA_TAU_LIMIT
            || (!vertex2.is_null() && d.v(vertex2).tau - self.tau2 < Diagram::DELTA_TAU_LIMIT)
        {
            return 0.0;
        }
        self.vertex1 = vertex1;
        self.vertex2 = vertex2;

        let algo_ratio = (2 * d.order - 1) as f64 / d.order as f64;
        algo_ratio * 2.0 * d.alpha * Diagram::OMEGA * delta_t / PI
            * ((self.q.norm() * Diagram::p0() + self.q.dot(&p_mean)) * (self.tau2 - self.tau1)
                / Diagram::MASS
                - if vertex2.is_null() {
                    d.dispersion(&d.momentum_out()) * (self.tau2 - d.tau())
                } else {
                    0.0
                })
            .exp()
            * (1.0 - (-(lambda * (d.max_tau - self.tau1))).exp())
    }
}

/// Removes a randomly chosen phonon arc, the inverse of [`AddPhonon`].
#[derive(Clone, Debug, Default)]
pub struct RemovePhonon {
    vertex1: VKey,
    vertex2: VKey,
    q: Vector3<f64>,
}

impl RemovePhonon {
    pub fn attempt<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        if d.order == d.min_order {
            0.0
        } else if d.order == 1 {
            self.attempt_to_zero_fake(d)
        } else {
            self.higher_order(d, rng)
        }
    }

    pub fn accept(&mut self, d: &mut Diagram) {
        if d.order == 1 {
            d.clear_fake_order_one();
        } else {
            d.remove_arc(self.vertex1, self.vertex2);
        }
        debug_assert!(crate::sanity::check_sanity(d).is_ok());
    }

    fn attempt_to_zero_fake(&mut self, d: &Diagram) -> f64 {
        self.vertex1 = d.head;
        self.vertex2 = d.tail;
        self.q = d.v(d.head).q;
        PI / (2.0 * d.alpha * Diagram::OMEGA.powi(2))
            * ((Diagram::OMEGA
                + (self.q.norm_squared() / 2.0 - self.q.dot(&d.momentum_out())) / Diagram::MASS)
                * d.tau())
            .exp()
            / (1.0 + self.q.norm() / Diagram::p0()).powi(2)
    }

    fn higher_order<R: Rng + ?Sized>(&mut self, d: &Diagram, rng: &mut R) -> f64 {
        let mut v = random_vertex(d, rng);
        while v == d.head || d.v(v).link == d.head {
            v = random_vertex(d, rng);
        }
        let mut left = v;
        let mut right = d.v(v).link;
        if d.v(left).tau > d.v(right).tau {
            std::mem::swap(&mut left, &mut right);
        }
        self.vertex1 = left;
        self.vertex2 = right;
        self.q = d.v(left).q;

        if d.order != 2 {
            let second_last = d.prev(d.tail);
            let mut k = d.next(left);
            while k != right {
                if d.v(k).phonons_above == 1 && k != second_last {
                    return 0.0;
                }
                k = d.next(k);
            }
        }

        let delta_t = -d.v(d.prev(left)).tau
            + if d.next(left) == right {
                d.v(d.next(right)).tau
            } else {
                d.v(d.next(left)).tau
            };
        let p_mean = d.get_p_mean_range(left, right, self.q);
        let algo_ratio = (d.order - 1) as f64 / (2 * d.order - 3) as f64;
        algo_ratio * PI / (2.0 * d.alpha * Diagram::OMEGA * delta_t)
            * (-(self.q.norm() * Diagram::p0() + self.q.dot(&p_mean))
                * (d.v(right).tau - d.v(left).tau)
                / Diagram::MASS
                + if d.next(right).is_null() {
                    d.dispersion(&d.momentum_out()) * (d.v(right).tau - d.v(d.prev(right)).tau)
                } else {
                    0.0
                })
            .exp()
            / (1.0
                - (-(Diagram::OMEGA * (1.0 + self.q.norm() / Diagram::p0()).powi(2))
                    * (d.max_tau - d.v(left).tau))
                    .exp())
    }
}

fn draw_new_q<R: Rng + ?Sized>(rng: &mut R) -> Vector3<f64> {
    draw_new_q_from_uniforms(rng.gen(), rng.gen(), rng.gen())
}

fn random_vertex<R: Rng + ?Sized>(d: &Diagram, rng: &mut R) -> VKey {
    d.storage[uniform_index(rng, d.storage.len())]
}

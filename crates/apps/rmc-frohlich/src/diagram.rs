use std::f64::consts::PI;

use nalgebra::Vector3;
use slotmap::{new_key_type, Key, SlotMap};

new_key_type! {
    pub struct VKey;
}

/// An imaginary-time vertex in a Fröhlich-polaron self-energy diagram: either a bare-propagator
/// endpoint or one end of a phonon arc (linked via `link` to its partner).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub tau: f64,
    pub p_out: Vector3<f64>,
    pub q: Vector3<f64>,
    pub phonons_above: usize,
    pub link: VKey,
    pub prev: VKey,
    pub next: VKey,
    pub storage_idx: usize,
}

impl Vertex {
    pub fn new(tau: f64, p_out: Vector3<f64>, q: Vector3<f64>) -> Self {
        Self {
            tau,
            p_out,
            q,
            phonons_above: 0,
            link: VKey::null(),
            prev: VKey::null(),
            next: VKey::null(),
            storage_idx: usize::MAX,
        }
    }
}

/// A Fröhlich-polaron self-energy diagram: a time-ordered linked list of vertices (in `arena`,
/// from `head` to `tail`) connected pairwise into phonon arcs.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Diagram {
    pub arena: SlotMap<VKey, Vertex>,
    pub head: VKey,
    pub tail: VKey,
    pub storage: Vec<VKey>,
    pub order: usize,
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub max_tau: f64,
    pub start_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
}

impl Default for Diagram {
    fn default() -> Self {
        Self::new()
    }
}

impl Diagram {
    pub const MASS: f64 = 1.0;
    pub const OMEGA: f64 = 1.0;
    pub const DELTA_TAU_LIMIT: f64 = 10.0 * f64::EPSILON;

    pub fn new() -> Self {
        let mut diagram = Self {
            arena: SlotMap::with_key(),
            head: VKey::null(),
            tail: VKey::null(),
            storage: Vec::new(),
            order: 0,
            alpha: 1.0,
            mu: -1.1,
            momentum: 0.0,
            max_tau: 30.0,
            start_tau: 1.0,
            min_order: 0,
            max_order: 10_000,
        };
        diagram.set_to_0th_order();
        diagram
    }

    pub fn with_parameters(
        alpha: f64,
        mu: f64,
        momentum: f64,
        max_tau: f64,
        start_tau: f64,
        min_order: usize,
        max_order: usize,
    ) -> Self {
        let mut diagram = Self {
            arena: SlotMap::with_key(),
            head: VKey::null(),
            tail: VKey::null(),
            storage: Vec::new(),
            order: 0,
            alpha,
            mu,
            momentum,
            max_tau,
            start_tau,
            min_order,
            max_order,
        };
        diagram.set_to_0th_order();
        diagram
    }

    pub fn p0() -> f64 {
        (2.0 * Self::MASS * Self::OMEGA).sqrt()
    }

    pub fn v(&self, k: VKey) -> &Vertex {
        &self.arena[k]
    }

    pub fn vm(&mut self, k: VKey) -> &mut Vertex {
        &mut self.arena[k]
    }

    pub fn next(&self, k: VKey) -> VKey {
        self.v(k).next
    }

    pub fn prev(&self, k: VKey) -> VKey {
        self.v(k).prev
    }

    pub fn is_tail(&self, k: VKey) -> bool {
        k == self.tail
    }

    pub fn vertex_count(&self) -> usize {
        self.storage.len()
    }

    pub fn ordered_keys(&self) -> Vec<VKey> {
        let mut keys = Vec::with_capacity(self.vertex_count());
        let mut k = self.head;
        while !k.is_null() {
            keys.push(k);
            k = self.next(k);
        }
        keys
    }

    pub fn set_to_0th_order(&mut self) {
        self.arena.clear();
        self.storage.clear();
        self.order = 0;

        let global_p = self.momentum_vec();
        let head = self
            .arena
            .insert(Vertex::new(0.0, global_p, Vector3::zeros()));
        let tail = self
            .arena
            .insert(Vertex::new(self.start_tau, global_p, Vector3::zeros()));
        self.head = head;
        self.tail = tail;
        self.arena[head].next = tail;
        self.arena[tail].prev = head;
        self.push_storage(head);
        self.push_storage(tail);
    }

    pub fn momentum_vec(&self) -> Vector3<f64> {
        Vector3::new(self.momentum, 0.0, 0.0)
    }

    pub fn tau(&self) -> f64 {
        self.v(self.tail).tau
    }

    pub fn momentum_out(&self) -> Vector3<f64> {
        self.v(self.tail).p_out
    }

    pub fn front(&self) -> &Vertex {
        self.v(self.head)
    }

    pub fn back(&self) -> &Vertex {
        self.v(self.tail)
    }

    pub fn is_incoming(&self, k: VKey) -> bool {
        let link = self.v(k).link;
        !link.is_null() && self.v(link).tau < self.v(k).tau
    }

    pub fn is_outgoing(&self, k: VKey) -> bool {
        let link = self.v(k).link;
        !link.is_null() && self.v(link).tau > self.v(k).tau
    }

    pub fn bare_dispersion(p: &Vector3<f64>) -> f64 {
        p.norm_squared() / (2.0 * Self::MASS)
    }

    pub fn dispersion(&self, p: &Vector3<f64>) -> f64 {
        Self::bare_dispersion(p) - self.mu
    }

    pub fn bare_propagator(&self, p: &Vector3<f64>, tau: f64) -> f64 {
        (-self.dispersion(p) * tau).exp()
    }

    pub fn get_p_mean_range(&self, begin: VKey, end: VKey, addition: Vector3<f64>) -> Vector3<f64> {
        assert!(!begin.is_null());
        assert!(!end.is_null());
        let mut p_mean = Vector3::zeros();
        let mut k = begin;
        while k != end {
            let next = self.next(k);
            assert!(!next.is_null(), "range end was not reachable from begin");
            p_mean += (self.v(k).p_out + addition) * (self.v(next).tau - self.v(k).tau);
            k = next;
        }
        p_mean / (self.v(end).tau - self.v(begin).tau)
    }

    pub fn get_p_mean_between(&self, tau1: f64, tau2: f64, begin: VKey) -> (Vector3<f64>, VKey) {
        assert!(tau2 > tau1);
        assert!(!begin.is_null());
        assert!(tau1 >= self.v(begin).tau);

        let mut end = self.next(begin);
        let mut p_mean = self.v(begin).p_out;
        if !end.is_null() && self.v(end).tau < tau2 {
            let mut it = begin;
            p_mean = self.v(it).p_out * (self.v(end).tau - tau1);
            it = end;
            end = self.next(end);
            while !end.is_null() && self.v(end).tau < tau2 {
                p_mean += self.v(it).p_out * (self.v(end).tau - self.v(it).tau);
                it = end;
                end = self.next(end);
            }
            p_mean += self.v(it).p_out * (tau2 - self.v(it).tau);
            p_mean /= tau2 - tau1;
        }
        (p_mean, end)
    }

    pub fn exact_estimator(&self, t0: f64) -> f64 {
        assert!(
            self.order > 0,
            "exact estimator needs linked phonon vertices"
        );
        let lambda = t0 / self.tau() - 1.0;

        let mut electron_sum = 0.0;
        let mut k = self.head;
        while k != self.tail {
            let next = self.next(k);
            electron_sum += self.dispersion(&self.v(k).p_out) * (self.v(next).tau - self.v(k).tau);
            k = next;
        }

        let mut phonon_sum = 0.0;
        k = self.head;
        while !k.is_null() {
            let delta = self.v(k).tau - self.v(self.v(k).link).tau;
            if delta > 0.0 {
                phonon_sum += delta * Self::OMEGA;
            }
            k = self.next(k);
        }

        (t0 / self.tau()).powi(2 * (self.order as i32 - 1))
            * (-(lambda * (electron_sum + phonon_sum))).exp()
    }

    pub fn insert_arc(&mut self, tau1: f64, tau2: f64, q: Vector3<f64>) -> (VKey, VKey) {
        assert!(
            self.order > 0,
            "insert_arc requires the fake order-1 sector"
        );
        let left = self.find_left_of_tau(tau1);
        let before_right = self.find_first_after_from(self.next(left), tau2);
        self.insert_arc_between(left, before_right, tau1, tau2, q)
    }

    pub fn insert_arc_between(
        &mut self,
        left: VKey,
        before_right: VKey,
        tau1: f64,
        tau2: f64,
        q: Vector3<f64>,
    ) -> (VKey, VKey) {
        assert!(tau2 > tau1);
        let mut v1 = Vertex::new(tau1, self.v(left).p_out, q);
        v1.phonons_above = self.v(left).phonons_above;
        if self.is_outgoing(left) {
            v1.phonons_above += 1;
        }
        let new1 = self.splice_after(left, v1);

        let right_left = if before_right.is_null() {
            self.tail
        } else {
            self.prev(before_right)
        };
        let mut v2 = Vertex::new(tau2, self.v(right_left).p_out, q);
        v2.phonons_above = self.v(right_left).phonons_above;
        if self.is_outgoing(right_left) {
            v2.phonons_above += 1;
        }
        let new2 = self.splice_after(right_left, v2);

        self.vm(new1).link = new2;
        self.vm(new2).link = new1;

        let mut k = new1;
        while k != new2 {
            self.vm(k).p_out -= q;
            k = self.next(k);
        }
        k = self.next(new1);
        while k != new2 {
            self.vm(k).phonons_above += 1;
            k = self.next(k);
        }
        self.order += 1;
        (new1, new2)
    }

    pub fn remove_arc(&mut self, a: VKey, b: VKey) {
        assert_eq!(self.v(a).link, b);
        assert_eq!(self.v(b).link, a);
        let (left, right) = if self.v(a).tau < self.v(b).tau {
            (a, b)
        } else {
            (b, a)
        };
        let q = self.v(left).q;
        let mut k = self.next(left);
        while k != right {
            self.vm(k).p_out += q;
            self.vm(k).phonons_above -= 1;
            k = self.next(k);
        }
        self.unlink(left);
        self.unlink(right);
        self.order -= 1;
    }

    pub fn add_phonon(&mut self, tau1: f64, tau2: f64, q: Vector3<f64>) {
        self.insert_arc(tau1, tau2, q);
    }

    pub fn set_vertex_tau(&mut self, k: VKey, tau: f64) {
        self.vm(k).tau = tau;
    }

    pub fn scale_taus(&mut self, scale: f64) {
        let mut k = self.head;
        while !k.is_null() {
            self.vm(k).tau *= scale;
            k = self.next(k);
        }
    }

    pub fn update_arc_q(&mut self, left: VKey, right: VKey, new_q: Vector3<f64>) {
        assert_eq!(self.v(left).link, right);
        let old_q = self.v(left).q;
        let mut k = left;
        while k != right {
            self.vm(k).p_out += old_q;
            self.vm(k).p_out -= new_q;
            k = self.next(k);
        }
        self.vm(left).q = new_q;
        self.vm(right).q = new_q;
    }

    pub fn set_to_fake_order_one(&mut self, q: Vector3<f64>) {
        assert_eq!(self.vertex_count(), 2);
        assert_eq!(self.order, 0);
        self.vm(self.head).q = q;
        self.vm(self.tail).q = q;
        self.vm(self.head).p_out -= q;
        self.vm(self.head).link = self.tail;
        self.vm(self.tail).link = self.head;
        self.order = 1;
    }

    pub fn clear_fake_order_one(&mut self) {
        assert_eq!(self.vertex_count(), 2);
        assert_eq!(self.order, 1);
        let q = self.v(self.head).q;
        self.vm(self.head).p_out += q;
        let head = self.head;
        let tail = self.tail;
        for k in [head, tail] {
            self.vm(k).q = Vector3::zeros();
            self.vm(k).link = VKey::null();
            self.vm(k).phonons_above = 0;
        }
        self.order = 0;
    }

    pub(crate) fn swap_arc_connectivity(&mut self, i: VKey, j: VKey) {
        let link_i = self.v(i).link;
        let link_j = self.v(j).link;
        let q_i = self.v(i).q;
        let q_j = self.v(j).q;

        self.vm(i).q = q_j;
        self.vm(j).q = q_i;
        self.vm(i).link = link_j;
        self.vm(j).link = link_i;
        self.vm(link_j).link = i;
        self.vm(link_i).link = j;
    }

    pub fn from_arcs(
        alpha: f64,
        mu: f64,
        momentum: f64,
        max_tau: f64,
        tau: f64,
        arcs: &[(f64, f64, Vector3<f64>)],
    ) -> Self {
        let mut diagram = Self::with_parameters(alpha, mu, momentum, max_tau, tau, 0, 10_000);
        if let Some(&(tau1, tau2, q)) = arcs.first() {
            assert_eq!(tau1, 0.0, "first arc must be the fake endpoint sector");
            assert_eq!(tau2, tau, "first arc must be the fake endpoint sector");
            diagram.set_to_fake_order_one(q);
            for &(tau1, tau2, q) in &arcs[1..] {
                diagram.insert_arc(tau1, tau2, q);
            }
        }
        diagram
    }

    pub fn find_left_of_tau(&self, tau: f64) -> VKey {
        let mut k = self.head;
        loop {
            let next = self.next(k);
            assert!(!next.is_null(), "tau must be before or at the current tail");
            if self.v(next).tau > tau {
                return k;
            }
            k = next;
        }
    }

    pub fn find_first_after_from(&self, start: VKey, tau: f64) -> VKey {
        let mut k = start;
        while !k.is_null() {
            if self.v(k).tau > tau {
                return k;
            }
            k = self.next(k);
        }
        VKey::null()
    }

    pub fn push_storage(&mut self, k: VKey) {
        let idx = self.storage.len();
        self.storage.push(k);
        self.vm(k).storage_idx = idx;
    }

    pub fn swap_remove_storage(&mut self, k: VKey) {
        let idx = self.v(k).storage_idx;
        self.storage.swap_remove(idx);
        if idx < self.storage.len() {
            let moved = self.storage[idx];
            self.vm(moved).storage_idx = idx;
        }
    }

    pub fn splice_after(&mut self, left: VKey, mut vertex: Vertex) -> VKey {
        let right = self.next(left);
        vertex.prev = left;
        vertex.next = right;
        let k = self.arena.insert(vertex);
        self.vm(left).next = k;
        if right.is_null() {
            self.tail = k;
        } else {
            self.vm(right).prev = k;
        }
        self.push_storage(k);
        k
    }

    pub fn unlink(&mut self, k: VKey) -> Vertex {
        let prev = self.prev(k);
        let next = self.next(k);
        if prev.is_null() {
            self.head = next;
        } else {
            self.vm(prev).next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.vm(next).prev = prev;
        }
        self.swap_remove_storage(k);
        self.arena.remove(k).expect("vertex key must exist")
    }
}

/// Normalization of the zeroth-order (no-phonon) sector weight over `[0, max_tau]`.
pub fn norm0(max_tau: f64, energy: f64) -> f64 {
    (1.0 - (-energy * max_tau).exp()) / energy
}

pub fn spherical_to_cartesian(r: f64, theta: f64, phi: f64) -> Vector3<f64> {
    Vector3::new(
        r * phi.cos() * theta.sin(),
        r * phi.sin() * theta.sin(),
        r * theta.cos(),
    )
}

pub fn theta_from_cartesian(r: &Vector3<f64>) -> f64 {
    (r.z / r.norm()).acos()
}

pub fn phi_from_cartesian(r: &Vector3<f64>) -> f64 {
    r.y.atan2(r.x)
}

pub fn draw_new_q_from_uniforms(r1: f64, r2: f64, r3: f64) -> Vector3<f64> {
    let theta = (1.0 - 2.0 * r1).acos();
    let q = Diagram::p0() / r2 - Diagram::p0();
    spherical_to_cartesian(q, theta, 2.0 * PI * r3)
}

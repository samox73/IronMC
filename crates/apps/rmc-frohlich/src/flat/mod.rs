//! Fixed-capacity SoA diagram layout for the GPU path.

pub mod updates;

use slotmap::Key;

use crate::diagram::{Diagram, VKey};
use crate::physics::{self, Vec3};

pub const NULL: u32 = u32::MAX;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlatDiagram {
    pub tau: Vec<f64>,
    pub p_out: Vec<Vec3>,
    pub q: Vec<Vec3>,
    pub link: Vec<u32>,
    pub prev: Vec<u32>,
    pub next: Vec<u32>,
    pub storage_idx: Vec<u32>,
    pub phonons_above: Vec<u32>,
    pub storage: Vec<u32>,
    pub head: u32,
    pub tail: u32,
    pub order: usize,
    pub alpha: f64,
    pub mu: f64,
    pub momentum: f64,
    pub max_tau: f64,
    pub start_tau: f64,
    pub min_order: usize,
    pub max_order: usize,
    pub max_order_gpu: usize,
}

impl FlatDiagram {
    pub fn with_parameters(
        alpha: f64,
        mu: f64,
        momentum: f64,
        max_tau: f64,
        start_tau: f64,
        min_order: usize,
        max_order: usize,
        max_order_gpu: usize,
    ) -> Self {
        let capacity = 2 * max_order_gpu + 2;
        let mut diagram = Self {
            tau: vec![0.0; capacity],
            p_out: vec![[0.0; 3]; capacity],
            q: vec![[0.0; 3]; capacity],
            link: vec![NULL; capacity],
            prev: vec![NULL; capacity],
            next: vec![NULL; capacity],
            storage_idx: vec![NULL; capacity],
            phonons_above: vec![0; capacity],
            storage: Vec::with_capacity(capacity),
            head: NULL,
            tail: NULL,
            order: 0,
            alpha,
            mu,
            momentum,
            max_tau,
            start_tau,
            min_order,
            max_order,
            max_order_gpu,
        };
        diagram.set_to_0th_order();
        diagram
    }

    pub fn from_diagram(diagram: &Diagram, max_order_gpu: usize) -> Option<Self> {
        let keys = diagram.ordered_keys();
        if keys.len() > 2 * max_order_gpu + 2 {
            return None;
        }

        let mut flat = Self::with_parameters(
            diagram.alpha,
            diagram.mu,
            diagram.momentum,
            diagram.max_tau,
            diagram.start_tau,
            diagram.min_order,
            diagram.max_order,
            max_order_gpu,
        );
        flat.clear_storage();
        flat.order = diagram.order;
        flat.head = 0;
        flat.tail = (keys.len() - 1) as u32;

        for (i, &key) in keys.iter().enumerate() {
            let slot = i as u32;
            let vertex = diagram.v(key);
            flat.tau[i] = vertex.tau;
            flat.p_out[i] = v3(&vertex.p_out);
            flat.q[i] = v3(&vertex.q);
            flat.phonons_above[i] = vertex.phonons_above as u32;
            flat.prev[i] = if i == 0 { NULL } else { slot - 1 };
            flat.next[i] = if i + 1 == keys.len() { NULL } else { slot + 1 };
            flat.link[i] = key_pos(&keys, vertex.link);
            flat.push_storage(slot);
        }
        Some(flat)
    }

    pub fn capacity(&self) -> usize {
        self.tau.len()
    }

    pub fn vertex_count(&self) -> usize {
        self.storage.len()
    }

    pub fn tau(&self) -> f64 {
        self.tau[self.tail as usize]
    }

    pub fn momentum_out(&self) -> Vec3 {
        self.p_out[self.tail as usize]
    }

    pub fn has_arc_capacity(&self) -> bool {
        self.vertex_count() + 2 <= self.capacity()
    }

    pub fn ordered_slots(&self) -> Vec<u32> {
        let mut slots = Vec::with_capacity(self.vertex_count());
        let mut slot = self.head;
        while slot != NULL {
            slots.push(slot);
            slot = self.next[slot as usize];
        }
        slots
    }

    pub fn set_to_0th_order(&mut self) {
        self.clear_storage();
        self.order = 0;
        self.head = 0;
        self.tail = 1;
        let p = [self.momentum, 0.0, 0.0];
        self.write_vertex(0, 0.0, p, [0.0; 3]);
        self.write_vertex(1, self.start_tau, p, [0.0; 3]);
        self.next[0] = 1;
        self.prev[1] = 0;
        self.push_storage(0);
        self.push_storage(1);
    }

    pub fn is_incoming(&self, slot: u32) -> bool {
        let link = self.link[slot as usize];
        link != NULL && self.tau[link as usize] < self.tau[slot as usize]
    }

    pub fn is_outgoing(&self, slot: u32) -> bool {
        let link = self.link[slot as usize];
        link != NULL && self.tau[link as usize] > self.tau[slot as usize]
    }

    pub fn get_p_mean_range(&self, begin: u32, end: u32, addition: Vec3) -> Vec3 {
        assert!(begin != NULL);
        assert!(end != NULL);
        let mut p_mean = [0.0; 3];
        let mut slot = begin;
        while slot != end {
            let next = self.next[slot as usize];
            assert!(next != NULL, "range end was not reachable from begin");
            let dtau = self.tau[next as usize] - self.tau[slot as usize];
            p_mean = add(
                p_mean,
                scale(add(self.p_out[slot as usize], addition), dtau),
            );
            slot = next;
        }
        scale(
            p_mean,
            1.0 / (self.tau[end as usize] - self.tau[begin as usize]),
        )
    }

    pub fn get_p_mean_between(&self, tau1: f64, tau2: f64, begin: u32) -> (Vec3, u32) {
        assert!(tau2 > tau1);
        assert!(begin != NULL);
        assert!(tau1 >= self.tau[begin as usize]);

        let mut end = self.next[begin as usize];
        let mut p_mean = self.p_out[begin as usize];
        if end != NULL && self.tau[end as usize] < tau2 {
            let mut it = begin;
            p_mean = scale(self.p_out[it as usize], self.tau[end as usize] - tau1);
            it = end;
            end = self.next[end as usize];
            while end != NULL && self.tau[end as usize] < tau2 {
                p_mean = add(
                    p_mean,
                    scale(
                        self.p_out[it as usize],
                        self.tau[end as usize] - self.tau[it as usize],
                    ),
                );
                it = end;
                end = self.next[end as usize];
            }
            p_mean = add(
                p_mean,
                scale(self.p_out[it as usize], tau2 - self.tau[it as usize]),
            );
            p_mean = scale(p_mean, 1.0 / (tau2 - tau1));
        }
        (p_mean, end)
    }

    pub fn find_left_of_tau(&self, tau: f64) -> u32 {
        let mut slot = self.head;
        loop {
            let next = self.next[slot as usize];
            assert!(next != NULL, "tau must be before or at the current tail");
            if self.tau[next as usize] > tau {
                return slot;
            }
            slot = next;
        }
    }

    pub fn find_first_after_from(&self, start: u32, tau: f64) -> u32 {
        let mut slot = start;
        while slot != NULL {
            if self.tau[slot as usize] > tau {
                return slot;
            }
            slot = self.next[slot as usize];
        }
        NULL
    }

    pub fn insert_arc(&mut self, tau1: f64, tau2: f64, q: Vec3) -> Option<(u32, u32)> {
        assert!(
            self.order > 0,
            "insert_arc requires the fake order-1 sector"
        );
        let left = self.find_left_of_tau(tau1);
        let before_right = self.find_first_after_from(self.next[left as usize], tau2);
        self.insert_arc_between(left, before_right, tau1, tau2, q)
    }

    pub fn insert_arc_between(
        &mut self,
        left: u32,
        before_right: u32,
        tau1: f64,
        tau2: f64,
        q: Vec3,
    ) -> Option<(u32, u32)> {
        assert!(tau2 > tau1);
        if !self.has_arc_capacity() {
            return None;
        }

        let mut phonons1 = self.phonons_above[left as usize];
        if self.is_outgoing(left) {
            phonons1 += 1;
        }
        let new1 = self.splice_after(left, tau1, self.p_out[left as usize], q)?;
        self.phonons_above[new1 as usize] = phonons1;

        let right_left = if before_right == NULL {
            self.tail
        } else {
            self.prev[before_right as usize]
        };
        let mut phonons2 = self.phonons_above[right_left as usize];
        if self.is_outgoing(right_left) {
            phonons2 += 1;
        }
        let new2 = self.splice_after(right_left, tau2, self.p_out[right_left as usize], q)?;
        self.phonons_above[new2 as usize] = phonons2;

        self.link[new1 as usize] = new2;
        self.link[new2 as usize] = new1;

        let mut slot = new1;
        while slot != new2 {
            self.p_out[slot as usize] = sub(self.p_out[slot as usize], q);
            slot = self.next[slot as usize];
        }
        slot = self.next[new1 as usize];
        while slot != new2 {
            self.phonons_above[slot as usize] += 1;
            slot = self.next[slot as usize];
        }
        self.order += 1;
        Some((new1, new2))
    }

    pub fn remove_arc(&mut self, a: u32, b: u32) {
        assert_eq!(self.link[a as usize], b);
        assert_eq!(self.link[b as usize], a);
        let (left, right) = if self.tau[a as usize] < self.tau[b as usize] {
            (a, b)
        } else {
            (b, a)
        };
        let q = self.q[left as usize];
        let mut slot = self.next[left as usize];
        while slot != right {
            self.p_out[slot as usize] = add(self.p_out[slot as usize], q);
            self.phonons_above[slot as usize] -= 1;
            slot = self.next[slot as usize];
        }
        self.unlink(left);
        self.unlink(right);
        self.order -= 1;
    }

    pub fn set_to_fake_order_one(&mut self, q: Vec3) {
        assert_eq!(self.vertex_count(), 2);
        assert_eq!(self.order, 0);
        let head = self.head as usize;
        let tail = self.tail as usize;
        self.q[head] = q;
        self.q[tail] = q;
        self.p_out[head] = sub(self.p_out[head], q);
        self.link[head] = self.tail;
        self.link[tail] = self.head;
        self.order = 1;
    }

    pub fn clear_fake_order_one(&mut self) {
        assert_eq!(self.vertex_count(), 2);
        assert_eq!(self.order, 1);
        let head = self.head as usize;
        let tail = self.tail as usize;
        let q = self.q[head];
        self.p_out[head] = add(self.p_out[head], q);
        for slot in [head, tail] {
            self.q[slot] = [0.0; 3];
            self.link[slot] = NULL;
            self.phonons_above[slot] = 0;
        }
        self.order = 0;
    }

    pub fn set_vertex_tau(&mut self, slot: u32, tau: f64) {
        self.tau[slot as usize] = tau;
    }

    pub fn scale_taus(&mut self, scale: f64) {
        let mut slot = self.head;
        while slot != NULL {
            self.tau[slot as usize] *= scale;
            slot = self.next[slot as usize];
        }
    }

    pub fn update_arc_q(&mut self, left: u32, right: u32, new_q: Vec3) {
        assert_eq!(self.link[left as usize], right);
        let old_q = self.q[left as usize];
        let mut slot = left;
        while slot != right {
            self.p_out[slot as usize] = add(self.p_out[slot as usize], old_q);
            self.p_out[slot as usize] = sub(self.p_out[slot as usize], new_q);
            slot = self.next[slot as usize];
        }
        self.q[left as usize] = new_q;
        self.q[right as usize] = new_q;
    }

    pub(crate) fn swap_arc_connectivity(&mut self, i: u32, j: u32) {
        let link_i = self.link[i as usize];
        let link_j = self.link[j as usize];
        let q_i = self.q[i as usize];
        let q_j = self.q[j as usize];

        self.q[i as usize] = q_j;
        self.q[j as usize] = q_i;
        self.link[i as usize] = link_j;
        self.link[j as usize] = link_i;
        self.link[link_j as usize] = i;
        self.link[link_i as usize] = j;
    }

    fn splice_after(&mut self, left: u32, tau: f64, p_out: Vec3, q: Vec3) -> Option<u32> {
        let slot = self.alloc_slot()?;
        let right = self.next[left as usize];
        self.write_vertex(slot as usize, tau, p_out, q);
        self.prev[slot as usize] = left;
        self.next[slot as usize] = right;
        self.next[left as usize] = slot;
        if right == NULL {
            self.tail = slot;
        } else {
            self.prev[right as usize] = slot;
        }
        self.push_storage(slot);
        Some(slot)
    }

    fn unlink(&mut self, slot: u32) {
        let prev = self.prev[slot as usize];
        let next = self.next[slot as usize];
        if prev == NULL {
            self.head = next;
        } else {
            self.next[prev as usize] = next;
        }
        if next == NULL {
            self.tail = prev;
        } else {
            self.prev[next as usize] = prev;
        }
        self.swap_remove_storage(slot);
        self.clear_vertex(slot as usize);
    }

    fn alloc_slot(&self) -> Option<u32> {
        self.storage_idx
            .iter()
            .position(|&idx| idx == NULL)
            .map(|idx| idx as u32)
    }

    fn clear_storage(&mut self) {
        self.storage.clear();
        self.storage_idx.fill(NULL);
        for slot in 0..self.capacity() {
            self.clear_vertex(slot);
        }
    }

    fn write_vertex(&mut self, slot: usize, tau: f64, p_out: Vec3, q: Vec3) {
        self.tau[slot] = tau;
        self.p_out[slot] = p_out;
        self.q[slot] = q;
        self.link[slot] = NULL;
        self.prev[slot] = NULL;
        self.next[slot] = NULL;
        self.phonons_above[slot] = 0;
    }

    fn clear_vertex(&mut self, slot: usize) {
        self.write_vertex(slot, 0.0, [0.0; 3], [0.0; 3]);
    }

    fn push_storage(&mut self, slot: u32) {
        let idx = self.storage.len() as u32;
        self.storage.push(slot);
        self.storage_idx[slot as usize] = idx;
    }

    fn swap_remove_storage(&mut self, slot: u32) {
        let idx = self.storage_idx[slot as usize] as usize;
        self.storage.swap_remove(idx);
        self.storage_idx[slot as usize] = NULL;
        if idx < self.storage.len() {
            let moved = self.storage[idx];
            self.storage_idx[moved as usize] = idx as u32;
        }
    }
}

fn key_pos(keys: &[VKey], key: VKey) -> u32 {
    if key.is_null() {
        return NULL;
    }
    keys.iter()
        .position(|&candidate| candidate == key)
        .expect("linked vertex must be in ordered key set") as u32
}

fn v3(p: &nalgebra::Vector3<f64>) -> Vec3 {
    [p.x, p.y, p.z]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vec3, scalar: f64) -> Vec3 {
    [a[0] * scalar, a[1] * scalar, a[2] * scalar]
}

pub(crate) fn unit(a: Vec3) -> Vec3 {
    scale(a, 1.0 / physics::norm(a))
}

pub(crate) fn scale_vec(a: Vec3, scalar: f64) -> Vec3 {
    scale(a, scalar)
}

pub(crate) fn add_vec(a: Vec3, b: Vec3) -> Vec3 {
    add(a, b)
}

pub(crate) fn sub_vec(a: Vec3, b: Vec3) -> Vec3 {
    sub(a, b)
}

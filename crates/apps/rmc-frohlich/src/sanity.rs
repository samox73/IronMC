//! Invariant checks for [`Diagram`]: linked-list/storage consistency, reciprocal arc links,
//! momentum conservation at each vertex, time ordering, and `phonons_above` counts. Used from
//! `debug_assert!`s after each update's `accept`.

use nalgebra::Vector3;
use slotmap::Key;

use crate::diagram::{Diagram, VKey};

pub fn check_sanity(d: &Diagram) -> Result<(), String> {
    if !(d.back().tau > 0.0 && d.back().tau < d.max_tau) {
        return Err(format!(
            "last tau must be in (0, max_tau): tau={}, max_tau={}",
            d.back().tau,
            d.max_tau
        ));
    }
    check_list_and_storage(d)?;
    check_links(d)?;
    check_momentum_conservation(d)?;
    check_time_ordering(d)?;
    check_phonons_above(d)?;
    Ok(())
}

pub fn check_list_and_storage(d: &Diagram) -> Result<(), String> {
    if d.head.is_null() || d.tail.is_null() {
        return Err("head/tail must be set".to_string());
    }
    if d.storage.len() != d.arena.len() {
        return Err(format!(
            "storage size {} differs from arena size {}",
            d.storage.len(),
            d.arena.len()
        ));
    }
    for (idx, &k) in d.storage.iter().enumerate() {
        if !d.arena.contains_key(k) {
            return Err(format!("storage[{idx}] contains a stale key"));
        }
        if d.v(k).storage_idx != idx {
            return Err(format!(
                "storage index mismatch for vertex {:?}: got {}, expected {}",
                k,
                d.v(k).storage_idx,
                idx
            ));
        }
    }

    let mut count = 0;
    let mut prev = VKey::null();
    let mut k = d.head;
    while !k.is_null() {
        if !d.arena.contains_key(k) {
            return Err("list contains a stale key".to_string());
        }
        if d.v(k).prev != prev {
            return Err(format!("prev pointer mismatch at {:?}", k));
        }
        if d.v(k).next.is_null() && k != d.tail {
            return Err(format!("non-tail vertex {:?} points to null next", k));
        }
        prev = k;
        k = d.v(k).next;
        count += 1;
        if count > d.arena.len() {
            return Err("cycle detected in vertex list".to_string());
        }
    }
    if prev != d.tail {
        return Err("tail is not the last list vertex".to_string());
    }
    if count != d.arena.len() {
        return Err(format!(
            "list count {count} differs from arena size {}",
            d.arena.len()
        ));
    }
    Ok(())
}

pub fn check_links(d: &Diagram) -> Result<(), String> {
    let expected_vertices = if d.order == 0 { 2 } else { 2 * d.order };
    if d.vertex_count() != expected_vertices {
        return Err(format!(
            "vertex count {} does not match order {}",
            d.vertex_count(),
            d.order
        ));
    }

    let mut k = d.head;
    while !k.is_null() {
        let v = d.v(k);
        if d.order == 0 {
            if !v.link.is_null() {
                return Err(format!("order-0 vertex {:?} unexpectedly has a link", k));
            }
        } else {
            if v.link.is_null() || !d.arena.contains_key(v.link) {
                return Err(format!("vertex {:?} has invalid link {:?}", k, v.link));
            }
            if d.v(v.link).link != k {
                return Err(format!("vertex {:?} link is not reciprocal", k));
            }
        }
        k = v.next;
    }
    Ok(())
}

pub fn check_momentum_conservation(d: &Diagram) -> Result<(), String> {
    let precision = 1.0e-7;
    if d.order == 0 {
        if (d.front().p_out.norm() - d.momentum).abs() < precision {
            return Ok(());
        }
        return Err(format!(
            "order-0 momentum mismatch: p_out={}, momentum={}",
            d.front().p_out.norm(),
            d.momentum
        ));
    }

    let mut previous_p = Vector3::new(d.momentum, 0.0, 0.0);
    let mut k = d.head;
    while !k.is_null() {
        let v = d.v(k);
        if d.is_incoming(k) && (previous_p + v.q - v.p_out).norm() > precision {
            return Err(format!(
                "incoming momentum conservation violated at vertex {:?}",
                k
            ));
        }
        if d.is_outgoing(k) && (previous_p - (v.p_out + v.q)).norm() > precision {
            return Err(format!(
                "outgoing momentum conservation violated at vertex {:?}",
                k
            ));
        }
        previous_p = v.p_out;
        k = v.next;
    }

    if (d.back().p_out.norm() - d.momentum).abs() > precision {
        return Err(format!(
            "last vertex momentum mismatch: last={}, momentum={}",
            d.back().p_out.norm(),
            d.momentum
        ));
    }
    Ok(())
}

pub fn check_time_ordering(d: &Diagram) -> Result<(), String> {
    let mut k = d.head;
    while !k.is_null() {
        let next = d.next(k);
        if !next.is_null() && d.v(k).tau > d.v(next).tau {
            return Err(format!(
                "vertices are not time ordered: {} > {}",
                d.v(k).tau,
                d.v(next).tau
            ));
        }
        k = next;
    }
    Ok(())
}

pub fn check_phonons_above(d: &Diagram) -> Result<(), String> {
    if d.front().phonons_above > 0 {
        return Err("first vertex has phonons_above > 0".to_string());
    }
    if d.back().phonons_above > 0 {
        return Err("last vertex has phonons_above > 0".to_string());
    }

    let keys = d.ordered_keys();
    for &k in &keys {
        let expected = keys
            .iter()
            .copied()
            .filter(|&other| {
                let link = d.v(other).link;
                !link.is_null()
                    && d.v(other).tau < d.v(link).tau
                    && d.v(other).tau < d.v(k).tau
                    && d.v(k).tau < d.v(link).tau
                    && other != k
                    && link != k
            })
            .count();
        if d.v(k).phonons_above != expected {
            return Err(format!(
                "vertex {:?} phonons_above mismatch: got {}, expected {}",
                k,
                d.v(k).phonons_above,
                expected
            ));
        }
    }
    Ok(())
}

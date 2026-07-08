//! Per-update proposal/acceptance counters for [`PolaronKernel`], collected after a run and
//! rendered as the "UPDATE STATISTICS" table in run summaries.

use rmc_core::mc::UpdateSet;
use serde::{Deserialize, Serialize};

use crate::app::PolaronKernel;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UpdateStatEntry {
    pub name: String,
    pub weight: f64,
    pub proposed: u64,
    pub impossible: u64,
    pub accepted: u64,
    pub acc_ratio: f64,
}

impl UpdateStatEntry {
    fn new(name: String, weight: f64, proposed: u64, impossible: u64, accepted: u64) -> Self {
        Self {
            name,
            weight,
            proposed,
            impossible,
            accepted,
            acc_ratio: acc_ratio(accepted, proposed),
        }
    }
}

pub fn collect(kernel: &PolaronKernel) -> Vec<UpdateStatEntry> {
    kernel
        .updates()
        .entries()
        .iter()
        .zip(kernel.updates().stats())
        .map(|(entry, stats)| {
            UpdateStatEntry::new(
                entry.update().name().to_string(),
                entry.weight(),
                stats.nprops,
                stats.nimps,
                stats.naccs,
            )
        })
        .collect()
}

pub fn merge(rows: Vec<Vec<UpdateStatEntry>>) -> Vec<UpdateStatEntry> {
    let mut merged: Option<Vec<UpdateStatEntry>> = None;
    for chain in rows {
        merged = Some(match merged {
            Some(acc) => merge_pair(acc, chain),
            None => chain,
        });
    }
    merged.unwrap_or_default()
}

pub fn render(rows: &[UpdateStatEntry]) -> String {
    let mut out = String::new();
    out.push_str("============================\n");
    out.push_str("UPDATE STATISTICS:\n");
    out.push_str("============================\n");
    out.push_str(&format!(
        "{:<28}{:<20}{:<20}{:<20}{:<20}{}\n",
        "Update", "Weight", "Proposed", "Impossible", "Accepted", "Acc. ratio"
    ));
    out.push_str(&"-".repeat(128));
    out.push('\n');
    for row in rows {
        out.push_str(&format!(
            "{:<28}{:<20.6}{:<20}{:<20}{:<20}{:.6}\n",
            row.name, row.weight, row.proposed, row.impossible, row.accepted, row.acc_ratio
        ));
    }
    out
}

fn merge_pair(lhs: Vec<UpdateStatEntry>, rhs: Vec<UpdateStatEntry>) -> Vec<UpdateStatEntry> {
    assert_eq!(lhs.len(), rhs.len());
    lhs.into_iter()
        .zip(rhs)
        .map(|(mut lhs, rhs)| {
            assert_eq!(lhs.name, rhs.name);
            assert_eq!(lhs.weight, rhs.weight);
            lhs.proposed += rhs.proposed;
            lhs.impossible += rhs.impossible;
            lhs.accepted += rhs.accepted;
            lhs.acc_ratio = acc_ratio(lhs.accepted, lhs.proposed);
            lhs
        })
        .collect()
}

fn acc_ratio(accepted: u64, proposed: u64) -> f64 {
    if proposed == 0 {
        0.0
    } else {
        accepted as f64 / proposed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::updates::phonon::{AddPhonon, RemovePhonon};
    use crate::updates::{
        ChangeInternalTau, ChangeQDirection, ChangeQModulus, ChangeTau, ChangeTopology,
        PolaronUpdate, RescaleDiagram,
    };

    #[test]
    fn polaron_update_names_match_reference_order() {
        let updates = [
            PolaronUpdate::ChangeTau(ChangeTau::default()),
            PolaronUpdate::ChangeInternalTau(ChangeInternalTau::default()),
            PolaronUpdate::AddPhonon(AddPhonon::default()),
            PolaronUpdate::RemovePhonon(RemovePhonon::default()),
            PolaronUpdate::RescaleDiagram(RescaleDiagram::default()),
            PolaronUpdate::ChangeQModulus(ChangeQModulus::default()),
            PolaronUpdate::ChangeQDirection(ChangeQDirection::default()),
            PolaronUpdate::ChangeTopology(ChangeTopology::default()),
        ];
        let names = updates.map(|update| update.name());
        assert_eq!(
            names,
            [
                "ChangeTau",
                "ChangeInternalTau",
                "AddPhonon",
                "RemovePhonon",
                "RescaleDiagram",
                "ChangeQModulus",
                "ChangeQDirection",
                "ChangeTopology",
            ]
        );
    }

    #[test]
    fn merge_sums_counts_by_index_and_recomputes_ratio() {
        let rows = merge(vec![
            vec![UpdateStatEntry::new("change_tau".to_string(), 1.0, 2, 0, 1)],
            vec![UpdateStatEntry::new("change_tau".to_string(), 1.0, 3, 1, 2)],
        ]);

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].proposed, 5);
        assert_eq!(rows[0].impossible, 1);
        assert_eq!(rows[0].accepted, 3);
        assert_eq!(rows[0].acc_ratio, 0.6);
    }

    #[test]
    fn render_contains_banner_row_and_ratio() {
        let rendered = render(&[UpdateStatEntry::new("change_tau".to_string(), 1.0, 4, 0, 1)]);

        assert!(rendered.contains("UPDATE STATISTICS"));
        assert!(rendered.contains("change_tau"));
        assert!(rendered.contains("0.250000"));
    }

    #[test]
    fn zero_proposals_have_zero_acceptance_ratio() {
        let row = UpdateStatEntry::new("change_tau".to_string(), 1.0, 0, 0, 0);
        assert_eq!(row.acc_ratio, 0.0);
    }
}

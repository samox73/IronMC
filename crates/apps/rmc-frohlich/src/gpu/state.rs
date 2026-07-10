use crate::config::RunConfig;
use crate::flat::{FlatDiagram, NULL};

#[derive(Clone, Debug, PartialEq)]
pub struct GpuStateBuffers {
    pub chains: usize,
    pub capacity: usize,
    pub tau: Vec<f64>,
    pub p_out: Vec<[f64; 3]>,
    pub q: Vec<[f64; 3]>,
    pub link: Vec<u32>,
    pub prev: Vec<u32>,
    pub next: Vec<u32>,
    pub storage_idx: Vec<u32>,
    pub phonons_above: Vec<u32>,
    pub storage: Vec<u32>,
    pub storage_len: Vec<u32>,
    pub head: Vec<u32>,
    pub tail: Vec<u32>,
    pub order: Vec<u32>,
}

impl GpuStateBuffers {
    pub fn initial(cfg: &RunConfig, chains: usize) -> Self {
        let diagrams = (0..chains)
            .map(|_| {
                FlatDiagram::with_parameters(
                    cfg.alpha,
                    cfg.mu,
                    cfg.momentum,
                    cfg.max_tau,
                    cfg.start_tau,
                    cfg.min_order,
                    cfg.max_order,
                    cfg.max_order_gpu,
                )
            })
            .collect::<Vec<_>>();
        Self::from_flat_diagrams(&diagrams)
    }

    pub fn from_flat_diagrams(diagrams: &[FlatDiagram]) -> Self {
        assert!(!diagrams.is_empty(), "GPU state needs at least one chain");
        let chains = diagrams.len();
        let capacity = diagrams[0].capacity();
        let len = chains * capacity;
        let mut buffers = Self {
            chains,
            capacity,
            tau: vec![0.0; len],
            p_out: vec![[0.0; 3]; len],
            q: vec![[0.0; 3]; len],
            link: vec![NULL; len],
            prev: vec![NULL; len],
            next: vec![NULL; len],
            storage_idx: vec![NULL; len],
            phonons_above: vec![0; len],
            storage: vec![NULL; len],
            storage_len: vec![0; chains],
            head: vec![NULL; chains],
            tail: vec![NULL; chains],
            order: vec![0; chains],
        };
        for (chain, diagram) in diagrams.iter().enumerate() {
            assert_eq!(diagram.capacity(), capacity);
            buffers.write_chain(chain, diagram);
        }
        buffers
    }

    pub fn to_flat_diagram(&self, chain: usize, cfg: &RunConfig) -> FlatDiagram {
        assert!(chain < self.chains);
        let mut diagram = FlatDiagram::with_parameters(
            cfg.alpha,
            cfg.mu,
            cfg.momentum,
            cfg.max_tau,
            cfg.start_tau,
            cfg.min_order,
            cfg.max_order,
            cfg.max_order_gpu,
        );
        let base = self.base(chain);
        diagram
            .tau
            .copy_from_slice(&self.tau[base..base + self.capacity]);
        diagram
            .p_out
            .copy_from_slice(&self.p_out[base..base + self.capacity]);
        diagram
            .q
            .copy_from_slice(&self.q[base..base + self.capacity]);
        diagram
            .link
            .copy_from_slice(&self.link[base..base + self.capacity]);
        diagram
            .prev
            .copy_from_slice(&self.prev[base..base + self.capacity]);
        diagram
            .next
            .copy_from_slice(&self.next[base..base + self.capacity]);
        diagram
            .storage_idx
            .copy_from_slice(&self.storage_idx[base..base + self.capacity]);
        diagram
            .phonons_above
            .copy_from_slice(&self.phonons_above[base..base + self.capacity]);
        diagram.storage.clear();
        diagram
            .storage
            .extend_from_slice(&self.storage[base..base + self.storage_len[chain] as usize]);
        diagram.head = self.head[chain];
        diagram.tail = self.tail[chain];
        diagram.order = self.order[chain] as usize;
        diagram
    }

    fn write_chain(&mut self, chain: usize, diagram: &FlatDiagram) {
        let base = self.base(chain);
        self.tau[base..base + self.capacity].copy_from_slice(&diagram.tau);
        self.p_out[base..base + self.capacity].copy_from_slice(&diagram.p_out);
        self.q[base..base + self.capacity].copy_from_slice(&diagram.q);
        self.link[base..base + self.capacity].copy_from_slice(&diagram.link);
        self.prev[base..base + self.capacity].copy_from_slice(&diagram.prev);
        self.next[base..base + self.capacity].copy_from_slice(&diagram.next);
        self.storage_idx[base..base + self.capacity].copy_from_slice(&diagram.storage_idx);
        self.phonons_above[base..base + self.capacity].copy_from_slice(&diagram.phonons_above);
        self.storage[base..base + diagram.storage.len()].copy_from_slice(&diagram.storage);
        self.storage_len[chain] = diagram.storage.len() as u32;
        self.head[chain] = diagram.head;
        self.tail[chain] = diagram.tail;
        self.order[chain] = diagram.order as u32;
    }

    fn base(&self, chain: usize) -> usize {
        chain * self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_round_trips() {
        let cfg = RunConfig {
            chains: 2,
            max_order_gpu: 4,
            ..RunConfig::default()
        };
        let buffers = GpuStateBuffers::initial(&cfg, 2);
        let diagram = buffers.to_flat_diagram(1, &cfg);
        assert_eq!(diagram.capacity(), 10);
        assert_eq!(diagram.order, 0);
        assert_eq!(diagram.storage, vec![0, 1]);
        assert_eq!(diagram.tau(), cfg.start_tau);
    }
}

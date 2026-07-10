use rand::{Error, RngCore};

const M0: u32 = 0xd251_1f53;
const M1: u32 = 0xcd9e_8d57;
const W0: u32 = 0x9e37_79b9;
const W1: u32 = 0xbb67_ae85;

#[derive(Clone, Debug)]
pub struct PhiloxRng {
    seed: u64,
    chain_id: u64,
    step: u64,
    block: u32,
    words: [u32; 4],
    next_word: usize,
}

impl PhiloxRng {
    pub fn new(seed: u64, chain_id: u64, step: u64) -> Self {
        Self {
            seed,
            chain_id,
            step,
            block: 0,
            words: [0; 4],
            next_word: 4,
        }
    }

    fn refill(&mut self) {
        self.words = keyed_draw(self.seed, self.chain_id, self.step, self.block);
        self.block = self.block.wrapping_add(1);
        self.next_word = 0;
    }
}

impl RngCore for PhiloxRng {
    fn next_u32(&mut self) -> u32 {
        if self.next_word == 4 {
            self.refill();
        }
        let word = self.words[self.next_word];
        self.next_word += 1;
        word
    }

    fn next_u64(&mut self) -> u64 {
        u64::from(self.next_u32()) | (u64::from(self.next_u32()) << 32)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut chunks = dest.chunks_exact_mut(4);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next_u32().to_le_bytes());
        }
        let remainder = chunks.into_remainder();
        if !remainder.is_empty() {
            let word = self.next_u32().to_le_bytes();
            remainder.copy_from_slice(&word[..remainder.len()]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let product = u64::from(a) * u64::from(b);
    ((product >> 32) as u32, product as u32)
}

#[inline]
fn round(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let (hi0, lo0) = mulhilo(M0, ctr[0]);
    let (hi1, lo1) = mulhilo(M1, ctr[2]);
    [hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0]
}

pub fn philox4x32_10(ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    let mut ctr = round(ctr, key);
    for _ in 1..10 {
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
        ctr = round(ctr, key);
    }
    ctr
}

pub fn keyed_draw(run_seed: u64, chain_id: u64, step: u64, draw_index: u32) -> [u32; 4] {
    philox4x32_10(
        [
            chain_id as u32,
            (chain_id >> 32) as u32,
            step as u32,
            ((step >> 32) as u32) ^ draw_index,
        ],
        [run_seed as u32, (run_seed >> 32) as u32],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random123_known_answer_vectors() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox4x32_10([0xffff_ffff; 4], [0xffff_ffff, 0xffff_ffff]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0],
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn rng_stream_is_counter_addressable() {
        let mut rng = PhiloxRng::new(7, 12, 34);
        assert_eq!(rng.next_u32(), keyed_draw(7, 12, 34, 0)[0]);
        assert_eq!(rng.next_u32(), keyed_draw(7, 12, 34, 0)[1]);
        assert_eq!(rng.next_u32(), keyed_draw(7, 12, 34, 0)[2]);
        assert_eq!(rng.next_u32(), keyed_draw(7, 12, 34, 0)[3]);
        assert_eq!(rng.next_u32(), keyed_draw(7, 12, 34, 1)[0]);
    }
}

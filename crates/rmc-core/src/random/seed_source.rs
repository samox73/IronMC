use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub type DefaultRng = Xoshiro256PlusPlus;

/// Identifier of an independent Markov chain.
///
/// `SeedSource` combines this value with its master seed to derive a deterministic, chain-local RNG
/// seed. Chain IDs are stable data, not thread IDs; changing rayon scheduling does not change which
/// RNG stream a chain receives.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ChainId(pub u64);

/// Deterministic source of per-chain RNG streams.
///
/// The same `(master, ChainId)` pair always yields the same [`DefaultRng`] seed. Different chain IDs
/// are mixed through a SplitMix-style finalizer before seeding `rand_xoshiro`, giving stable,
/// well-diffused streams for independent-chain Monte Carlo.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SeedSource {
    master: u64,
}

impl SeedSource {
    pub const fn new(master: u64) -> Self {
        Self { master }
    }

    /// Return the master seed used to derive all chain seeds.
    pub const fn master(&self) -> u64 {
        self.master
    }

    /// Build the default RNG for `chain`.
    pub fn rng_for(&self, chain: ChainId) -> DefaultRng {
        Xoshiro256PlusPlus::from_seed(self.seed_for(chain))
    }

    /// Return the raw 256-bit seed used for `chain`.
    ///
    /// This is exposed primarily for testing, auditability, and checkpoint metadata.
    pub fn seed_for(&self, chain: ChainId) -> [u8; 32] {
        let mut seed = [0_u8; 32];
        let mut state = self.master ^ chain.0.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        for chunk in seed.chunks_exact_mut(8) {
            state = mix64(state);
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        seed
    }
}

fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

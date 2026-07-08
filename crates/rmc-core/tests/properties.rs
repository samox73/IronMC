use proptest::prelude::*;
use rand_core::RngCore;
use rmc_core::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, Runner, SimulationParams,
    SingleUpdateSet, Update,
};
use rmc_core::random::{ChainId, SeedSource};
use rmc_core::Merge;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn integer_merge_is_associative_and_commutative(a in 0_u64..1_000_000, b in 0_u64..1_000_000, c in 0_u64..1_000_000) {
        prop_assert_eq!(a.merge(b), b.merge(a));
        prop_assert_eq!(a.merge(b).merge(c), a.merge(b.merge(c)));
    }

    #[test]
    fn seed_source_is_reproducible_and_separates_streams(master in any::<u64>(), start in 0_u64..1_000_000) {
        let source = SeedSource::new(master);
        let first = source.seed_for(ChainId(start));
        let same = source.seed_for(ChainId(start));
        let next = source.seed_for(ChainId(start + 1));

        prop_assert_eq!(first, same);
        prop_assert_ne!(first, next);
    }

    #[test]
    fn derived_rng_stream_prefixes_are_distinct_and_bit_balanced(master in any::<u64>()) {
        const CHAINS: u64 = 128;
        const WORDS_PER_CHAIN: usize = 8;

        let source = SeedSource::new(master);
        let mut prefixes = Vec::with_capacity(CHAINS as usize);
        let mut ones = 0_u64;
        let mut words = 0_u64;
        let mut adjacent_hamming = 0_u64;
        let mut previous_prefix: Option<[u64; WORDS_PER_CHAIN]> = None;

        for chain in 0..CHAINS {
            let mut rng = source.rng_for(ChainId(chain));
            let mut prefix = [0_u64; WORDS_PER_CHAIN];
            for word in &mut prefix {
                *word = rng.next_u64();
                ones += u64::from(word.count_ones());
                words += 1;
            }

            if let Some(previous) = previous_prefix {
                adjacent_hamming += previous
                    .iter()
                    .zip(prefix.iter())
                    .map(|(left, right)| u64::from((left ^ right).count_ones()))
                    .sum::<u64>();
            }

            prefixes.push(prefix);
            previous_prefix = Some(prefix);
        }

        prefixes.sort_unstable();
        prefixes.dedup();
        prop_assert_eq!(prefixes.len(), CHAINS as usize);

        let total_bits = words * 64;
        let ones_fraction = ones as f64 / total_bits as f64;
        prop_assert!(
            (0.475..=0.525).contains(&ones_fraction),
            "ones fraction {ones_fraction}"
        );

        let compared_adjacent_bits = (CHAINS - 1) * WORDS_PER_CHAIN as u64 * 64;
        let hamming_fraction = adjacent_hamming as f64 / compared_adjacent_bits as f64;
        prop_assert!(
            (0.45..=0.55).contains(&hamming_fraction),
            "adjacent hamming fraction {hamming_fraction}"
        );
    }

    #[test]
    fn parallel_run_matches_manual_chain_reduction(seed in any::<u64>(), chains in 1_u64..16, steps in 1_u64..64, steps_per_cycle in 1_u64..16) {
        let params = SimulationParams {
            max_steps: steps,
            steps_per_cycle,
            cycles_per_check: 1,
        };
        let seed = SeedSource::new(seed);
        let parallel = Runner::new(seed, parity_chain).chains(chains).run(params).unwrap();
        let mut manual: Option<(rmc_core::mc::SimulationStats, u64)> = None;
        for chain in 0..chains {
            let mut rng = seed.rng_for(ChainId(chain));
            let (state, mut kernel, measurement) = parity_chain(ChainId(chain));
            let (_state, stats, output) =
                run_chain(state, &mut rng, &mut kernel, measurement, params, NoopCallbacks).unwrap();
            manual = Some(match manual {
                Some((acc_stats, acc_output)) => (acc_stats.merge(stats), acc_output.merge(output)),
                None => (stats, output),
            });
        }

        prop_assert_eq!((parallel.stats, parallel.output), manual.unwrap());
    }
}

#[derive(Clone, Copy)]
struct RandomParityUpdate;

impl Update<u64> for RandomParityUpdate {
    fn attempt<R: rmc_core::random::Rng + ?Sized>(&mut self, state: &mut u64, rng: &mut R) -> f64 {
        *state ^= rng.gen_range(0..=1);
        1.0
    }

    fn accept(&mut self, _state: &mut u64) {}
}

fn parity_chain(
    _chain: ChainId,
) -> (
    u64,
    MetropolisKernel<SingleUpdateSet<RandomParityUpdate>>,
    impl Measurement<u64, Output = u64>,
) {
    #[derive(Default)]
    struct FinalState {
        latest: u64,
    }

    impl Measurement<u64> for FinalState {
        type Output = u64;

        fn measure(&mut self, state: &u64) {
            self.latest = *state;
        }

        fn finish(self) -> Self::Output {
            self.latest
        }
    }

    (
        0,
        MetropolisKernel::new(SingleUpdateSet::new(RandomParityUpdate)),
        FinalState::default(),
    )
}

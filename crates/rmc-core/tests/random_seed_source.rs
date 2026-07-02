use rand_core::RngCore;
use rmc_core::random::{ChainId, SeedSource};

#[test]
fn seed_source_is_reproducible_per_chain() {
    let source = SeedSource::new(42);
    let mut first = source.rng_for(ChainId(7));
    let mut second = source.rng_for(ChainId(7));

    for _ in 0..16 {
        assert_eq!(first.next_u64(), second.next_u64());
    }
}

#[test]
fn seed_source_separates_chain_streams() {
    let source = SeedSource::new(42);
    let mut first = source.rng_for(ChainId(7));
    let mut second = source.rng_for(ChainId(8));

    let first_values = (0..8).map(|_| first.next_u64()).collect::<Vec<_>>();
    let second_values = (0..8).map(|_| second.next_u64()).collect::<Vec<_>>();

    assert_ne!(first_values, second_values);
    assert_ne!(source.seed_for(ChainId(7)), source.seed_for(ChainId(8)));
}

#[test]
fn seed_source_produces_unique_well_diffused_seeds_for_many_chains() {
    let source = SeedSource::new(0x5eed_cafe);
    let seeds = (0..512)
        .map(|chain| source.seed_for(ChainId(chain)))
        .collect::<Vec<_>>();

    for (idx, seed) in seeds.iter().enumerate() {
        assert_eq!(
            seeds.iter().filter(|candidate| *candidate == seed).count(),
            1,
            "duplicate seed for chain {idx}"
        );
    }

    let mut total_distance = 0_u64;
    let mut min_distance = u32::MAX;
    for pair in seeds.windows(2) {
        let distance = hamming_distance(&pair[0], &pair[1]);
        min_distance = min_distance.min(distance);
        total_distance += u64::from(distance);
    }

    let mean_distance = total_distance as f64 / (seeds.len() - 1) as f64;
    assert!(
        min_distance >= 96,
        "adjacent chain seeds diffuse poorly: min_distance={min_distance}"
    );
    assert!(
        (118.0..=138.0).contains(&mean_distance),
        "unexpected adjacent seed diffusion: mean_distance={mean_distance}"
    );
}

fn hamming_distance(first: &[u8; 32], second: &[u8; 32]) -> u32 {
    first
        .iter()
        .zip(second)
        .map(|(lhs, rhs)| (lhs ^ rhs).count_ones())
        .sum()
}

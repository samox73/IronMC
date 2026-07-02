use rmc_core::random::{
    cauchy_pdf, cauchy_sample, exclusive_uniform_int_pdf, exponential_pdf, exponential_pdf_bounded,
    exponential_sample, exponential_sample_bounded, normal_pdf, safe_exponential_pdf,
    safe_exponential_sample, uniform_index, uniform_int_pdf, uniform_pdf, uniform_sample, ChainId,
    SeedSource,
};

fn assert_close(actual: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= 1e-14 * scale,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn simple_closed_form_helpers_match_analytical_values() {
    assert_eq!(uniform_sample(0.5, 0.0, 1.0), 0.5);
    assert_eq!(uniform_pdf(0.0, 1.0), 1.0);
    assert_eq!(uniform_int_pdf(0, 9), 0.1);
    assert_eq!(exclusive_uniform_int_pdf(0, 10), 0.1);
}

#[test]
fn floating_sample_helpers_match_closed_forms() {
    assert_close(uniform_sample(0.5, -2.0, 2.0), 0.0);
    assert_close(uniform_pdf(-2.0, 2.0), 0.25);
    assert_close(exponential_sample(1.0, 2.0, 3.0), 3.0);
    assert_close(exponential_pdf(3.0, 2.0, 3.0), 2.0);
    assert_close(exponential_sample_bounded(0.0, 1.25, -1.0, 1.0), -1.0);
    assert_close(
        exponential_pdf_bounded(-1.0, 1.25, -1.0, 1.0),
        1.25 / (1.0 - (-2.5_f64).exp()),
    );
    assert_close(safe_exponential_sample(0.25, 0.0, 2.0, 6.0), 3.0);
    assert_close(safe_exponential_pdf(3.5, 0.0, 2.0, 6.0), 0.25);
    assert_close(
        normal_pdf(-2.0, -2.0, 1.25),
        1.0 / (1.25 * (2.0 * std::f64::consts::PI).sqrt()),
    );
    assert_close(cauchy_sample(0.5, -2.0, 1.25), -2.0);
    assert_close(
        cauchy_pdf(-2.0, -2.0, 1.25),
        1.0 / (std::f64::consts::PI * 1.25),
    );
}

#[test]
fn integer_pdfs_match_closed_forms() {
    assert_eq!(uniform_int_pdf(-2, 2), 0.2);
    assert_eq!(exclusive_uniform_int_pdf(-2, 2), 0.25);
}

#[test]
fn sample_ranges_follow_cpp_preconditions() {
    assert_eq!(uniform_sample(0.0, -2.0, 1.0), -2.0);
    assert!(exponential_sample(0.5, 2.0, 3.0) >= 3.0);

    for lambda in [-1.5, 0.0, 1.25] {
        let sample = safe_exponential_sample(0.625, lambda, -5.0, -2.0);
        assert!((-5.0..=-2.0).contains(&sample));
    }
}

#[test]
fn transform_sampler_midpoint_moments_match_analytical_values() {
    let n = 100_000;
    let mut uniform_sum = 0.0;
    let mut exponential_sum = 0.0;
    let lambda = 2.5;
    let offset = -1.25;

    for i in 0..n {
        let r = (i as f64 + 0.5) / n as f64;
        uniform_sum += uniform_sample(r, -3.0, 5.0);
        exponential_sum += exponential_sample(r, lambda, offset);
    }

    let uniform_mean = uniform_sum / n as f64;
    let exponential_mean = exponential_sum / n as f64;

    assert!(
        (uniform_mean - 1.0).abs() < 1e-12,
        "uniform mean={uniform_mean}"
    );
    assert!(
        (exponential_mean - (offset + 1.0 / lambda)).abs() < 1e-4,
        "exponential mean={exponential_mean}"
    );
}

#[test]
fn uniform_index_is_balanced_for_fixed_seed() {
    let mut rng = SeedSource::new(0x5150).rng_for(ChainId(0));
    let buckets = 7;
    let draws = 70_000;
    let mut counts = vec![0_usize; buckets];

    for _ in 0..draws {
        counts[uniform_index(&mut rng, buckets)] += 1;
    }

    let expected = draws / buckets;
    for (idx, count) in counts.into_iter().enumerate() {
        let diff = count.abs_diff(expected);
        assert!(
            diff < expected / 20,
            "bucket {idx} count {count} differs too much from {expected}"
        );
    }
}

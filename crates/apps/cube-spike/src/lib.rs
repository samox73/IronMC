//! Phase-0 CubeCL spike support code.

pub mod philox;

pub fn exp_reference(lambda: f64, dtau: &[f64]) -> Vec<f64> {
    dtau.iter().map(|&x| (-lambda * x).exp()).collect()
}

pub fn divergent_loop_reference(input: &[u32], iterations: u32) -> Vec<u32> {
    input
        .iter()
        .enumerate()
        .map(|(pos, &x)| {
            let mut acc = x;
            for i in 0..iterations + (pos as u32 % 3) {
                acc = if (i + pos as u32) % 2 == 0 {
                    acc.wrapping_add(3)
                } else {
                    acc.wrapping_add(1)
                };
            }
            acc
        })
        .collect()
}

#[cfg(any(
    feature = "cubecl-cpu",
    feature = "cubecl-hip",
    feature = "cubecl-cuda"
))]
pub mod cube;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_reference_matches_libm() {
        let dtau = [0.0, 1e-9, 0.1, 1.0, 12.5];
        let actual = exp_reference(0.7, &dtau);
        for (actual, expected) in actual.iter().zip(dtau.map(|x| f64::exp(-0.7 * x))) {
            assert!((actual - expected).abs() <= expected.abs() * 1e-14);
        }
    }

    #[test]
    fn divergent_loop_has_runtime_bound_and_branches() {
        assert_eq!(
            divergent_loop_reference(&[0, 10, 20, 30], 4),
            vec![8, 19, 32, 38]
        );
    }
}

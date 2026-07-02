use rmc_grids::{Grid1d, LinearGrid};
use rmc_numeric::{NumericError, PolynomialInterpolation};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn quartic(x: f64) -> f64 {
    1.25 - 0.5 * x + 2.0 * x * x - 0.75 * x * x * x + 0.125 * x * x * x * x
}

#[test]
fn polynomial_interpolation_reconstructs_polynomial_from_arbitrary_nodes() {
    let nodes = [-2.0, -0.75, 0.0, 1.25, 3.0];
    let values = nodes.map(quartic);
    let interpolation = PolynomialInterpolation::new(nodes, values).unwrap();

    for x in LinearGrid::new(-2.0, 3.0, 41).unwrap().points() {
        assert_close(interpolation.evaluate(x).unwrap(), quartic(x), 1.0e-10);
    }

    assert_eq!(interpolation.degree(), 4);
    assert_eq!(interpolation.nodes(), nodes);
    assert_eq!(interpolation.values(), values);
    assert_eq!(interpolation.weights().len(), nodes.len());
}

#[test]
fn polynomial_interpolation_returns_node_values_exactly() {
    let interpolation = PolynomialInterpolation::new([-1.0, 0.0, 2.0], [3.5, -2.0, 7.25]).unwrap();

    assert_eq!(interpolation.evaluate(-1.0).unwrap(), 3.5);
    assert_eq!(interpolation.evaluate(0.0).unwrap(), -2.0);
    assert_eq!(interpolation.evaluate(2.0).unwrap(), 7.25);
}

#[test]
fn polynomial_interpolation_can_be_built_from_grid_points() {
    let grid = LinearGrid::new(-1.0, 1.0, 5).unwrap();
    let values = grid.points().map(quartic).collect::<Vec<_>>();
    let interpolation = PolynomialInterpolation::from_grid(&grid, values).unwrap();

    assert_close(
        interpolation.evaluate(0.375).unwrap(),
        quartic(0.375),
        1.0e-12,
    );
}

#[test]
fn polynomial_interpolation_rejects_invalid_inputs() {
    assert_eq!(
        PolynomialInterpolation::new([], []).unwrap_err(),
        NumericError::NotEnoughInterpolationPoints { points: 0 }
    );
    assert_eq!(
        PolynomialInterpolation::new([0.0, 1.0], [0.0]).unwrap_err(),
        NumericError::ValueCountMismatch {
            grid_points: 2,
            values: 1
        }
    );
    assert_eq!(
        PolynomialInterpolation::new([0.0, 1.0, 1.0], [0.0, 1.0, 2.0]).unwrap_err(),
        NumericError::DuplicateInterpolationNode {
            first: 1,
            second: 2,
            value: 1.0
        }
    );

    let err = PolynomialInterpolation::new([0.0, f64::NAN], [0.0, 1.0]).unwrap_err();
    assert!(matches!(
        err,
        NumericError::NonFiniteInterpolationData {
            kind: "node",
            index: 1,
            value
        } if value.is_nan()
    ));

    let err = PolynomialInterpolation::new([0.0, 1.0], [0.0, f64::INFINITY]).unwrap_err();
    assert_eq!(
        err,
        NumericError::NonFiniteInterpolationData {
            kind: "value",
            index: 1,
            value: f64::INFINITY
        }
    );

    let interpolation = PolynomialInterpolation::new([0.0], [2.0]).unwrap();
    assert_eq!(
        interpolation.evaluate(f64::INFINITY).unwrap_err(),
        NumericError::NonFiniteEvaluationPoint { x: f64::INFINITY }
    );
}

use rmc_grids::{CustomGrid, Grid1d, LinearGrid, PowerGrid};
use rmc_numeric::{CubicSplineInterpolation, NumericError};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn cubic(x: f64) -> f64 {
    3.123 * x * x * x - 0.8146 * x * x + x + 5.91243
}

fn cubic_derivative(x: f64) -> f64 {
    3.0 * 3.123 * x * x - 2.0 * 0.8146 * x + 1.0
}

#[test]
fn natural_cubic_spline_interpolates_grid_values_and_boundaries() {
    let grid = LinearGrid::new(0.0, 4.0, 5).unwrap();
    let values = grid.points().map(|x| x * x).collect::<Vec<_>>();
    let spline = CubicSplineInterpolation::natural(grid, values.clone()).unwrap();

    assert_eq!(spline.grid().len(), 5);
    assert_eq!(spline.values(), values);
    assert_close(spline.second_derivatives()[0], 0.0, 1.0e-12);
    assert_close(spline.second_derivatives()[4], 0.0, 1.0e-12);

    for (index, expected) in values.iter().enumerate() {
        let x = spline.grid().point(index).unwrap();
        assert_close(spline.evaluate(x).unwrap(), *expected, 1.0e-12);
    }

    assert_close(spline.evaluate(0.5).unwrap(), 0.3392857142857143, 1.0e-12);
    assert_close(spline.evaluate(3.5).unwrap(), 12.339285714285714, 1.0e-12);
    assert_eq!(
        spline.evaluate(-0.1).unwrap_err(),
        NumericError::OutOfDomain
    );
}

#[test]
fn clamped_cubic_spline_reconstructs_cubic_polynomial_on_nonuniform_grid() {
    let grid = PowerGrid::new(-3.0, 1.0, 100, 3.0).unwrap();
    let values = grid.points().map(cubic).collect::<Vec<_>>();
    let spline = CubicSplineInterpolation::with_endpoint_derivatives(
        grid,
        values,
        cubic_derivative(-3.0),
        cubic_derivative(1.0),
    )
    .unwrap();

    for x in LinearGrid::new(-3.0, 1.0, 257).unwrap().points() {
        assert_close(spline.evaluate(x).unwrap(), cubic(x), 1.0e-9);
    }
}

#[test]
fn natural_cubic_spline_works_on_descending_custom_grid() {
    let grid = CustomGrid::new([4.0, 3.0, 2.0, 1.0, 0.0]).unwrap();
    let values = grid.points().map(|x| x.sin()).collect::<Vec<_>>();
    let spline = CubicSplineInterpolation::natural(grid, values.clone()).unwrap();

    for (index, expected) in values.iter().enumerate() {
        let x = spline.grid().point(index).unwrap();
        assert_close(spline.evaluate(x).unwrap(), *expected, 1.0e-12);
    }
    assert!(spline.evaluate(1.5).unwrap().is_finite());
}

#[test]
fn cubic_spline_rejects_invalid_inputs() {
    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    assert_eq!(
        CubicSplineInterpolation::natural(grid, [0.0, 1.0]).unwrap_err(),
        NumericError::ValueCountMismatch {
            grid_points: 3,
            values: 2
        }
    );

    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let err = CubicSplineInterpolation::natural(grid, [0.0, f64::NAN, 1.0]).unwrap_err();
    assert!(matches!(
        err,
        NumericError::NonFiniteInterpolationData {
            kind: "value",
            index: 1,
            value
        } if value.is_nan()
    ));

    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    assert_eq!(
        CubicSplineInterpolation::with_endpoint_derivatives(
            grid,
            [0.0, 0.5, 1.0],
            f64::INFINITY,
            1.0
        )
        .unwrap_err(),
        NumericError::NonFiniteSplineBoundaryDerivative {
            which: "left",
            value: f64::INFINITY
        }
    );

    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let spline = CubicSplineInterpolation::natural(grid, [0.0, 0.5, 1.0]).unwrap();
    let err = spline.evaluate(f64::NAN).unwrap_err();
    assert!(matches!(
        err,
        NumericError::NonFiniteEvaluationPoint { x } if x.is_nan()
    ));
}

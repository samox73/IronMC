use rmc_grids::{AxisGrid, CustomGrid, Grid1d, LinearGrid, NdGrid, PowerGrid};
use rmc_numeric::{
    LinearInterpolation, LinearInterpolationMixed, LinearInterpolationNd, NumericError,
};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-10;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn line(x: f64) -> f64 {
    2.5 - 0.75 * x
}

fn plane(x: f64, y: f64) -> f64 {
    3.125 + 1.25 * x - 0.875 * y
}

fn hyperplane(x: f64, y: f64, z: f64) -> f64 {
    -1.25 + 0.75 * x - 1.5 * y + 2.25 * z
}

#[test]
fn linear_interpolation_matches_affine_function_on_nonuniform_grid() {
    let grid = PowerGrid::new(-5.0, 5.0, 25, 2.0).unwrap();
    let values = grid.points().map(line).collect::<Vec<_>>();
    let interpolation = LinearInterpolation::new(grid, values).unwrap();
    let probes = LinearGrid::new(-5.0, 5.0, 101).unwrap();

    for x in probes.points() {
        assert_close(interpolation.evaluate(x).unwrap(), line(x));
    }

    assert_eq!(interpolation.grid().len(), 25);
    assert_eq!(interpolation.values().len(), 25);
    assert_eq!(
        interpolation.evaluate(-5.1).unwrap_err(),
        NumericError::OutOfDomain
    );
}

#[test]
fn linear_interpolation_rejects_value_count_mismatch() {
    let grid = LinearGrid::new(0.0, 1.0, 3).unwrap();
    let err = LinearInterpolation::new(grid, [0.0, 1.0]).unwrap_err();

    assert_eq!(
        err,
        NumericError::ValueCountMismatch {
            grid_points: 3,
            values: 2
        }
    );
}

#[test]
fn bilinear_interpolation_matches_affine_plane() {
    let x = LinearGrid::new(-1.0, 1.0, 11).unwrap();
    let y = LinearGrid::new(-2.0, 2.0, 13).unwrap();
    let grid = NdGrid::new([x, y]).unwrap();
    let values = grid.points().map(|[x, y]| plane(x, y)).collect::<Vec<_>>();
    let interpolation = LinearInterpolationNd::new(grid, values).unwrap();

    for x in LinearGrid::new(-1.0, 1.0, 21).unwrap().points() {
        for y in LinearGrid::new(-2.0, 2.0, 23).unwrap().points() {
            assert_close(interpolation.evaluate([x, y]).unwrap(), plane(x, y));
        }
    }

    assert_eq!(interpolation.grid().shape(), [11, 13]);
    assert_eq!(interpolation.values().len(), 11 * 13);
    assert_eq!(
        interpolation.evaluate([1.1, 0.0]).unwrap_err(),
        NumericError::OutOfDomain
    );
}

#[test]
fn trilinear_interpolation_matches_affine_hyperplane_on_mixed_axes() {
    let x = AxisGrid::from(LinearGrid::new(-2.0, 2.0, 9).unwrap());
    let y = AxisGrid::from(PowerGrid::new(1.0, 5.0, 8, 2.0).unwrap());
    let z = AxisGrid::from(CustomGrid::new([10.0, 7.0, 3.0, 1.0]).unwrap());
    let grid = NdGrid::new([x, y, z]).unwrap();
    let values = grid
        .points()
        .map(|[x, y, z]| hyperplane(x, y, z))
        .collect::<Vec<_>>();
    let interpolation: LinearInterpolationMixed<3> =
        LinearInterpolationMixed::new(grid, values).unwrap();

    let probes = [
        [-1.75, 1.25, 9.0],
        [0.0, 2.5, 6.0],
        [1.5, 4.25, 2.0],
        [2.0, 5.0, 1.0],
    ];
    for point in probes {
        assert_close(
            interpolation.evaluate(point).unwrap(),
            hyperplane(point[0], point[1], point[2]),
        );
    }
}

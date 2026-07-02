use rmc_grids::{
    index_subrange, integer_subrange, AxisGrid, CustomGrid, Grid1d, LinearGrid, PowerGrid,
    SymmetricPowerGrid,
};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_slice_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

fn assert_grid(
    grid: &impl Grid1d,
    expected_points: &[f64],
    expected_centers: &[f64],
    expected_widths: &[f64],
) {
    assert_eq!(grid.len(), expected_points.len());
    assert_eq!(grid.bin_count(), expected_widths.len());
    assert_close(grid.first(), expected_points[0]);
    assert_close(grid.last(), *expected_points.last().unwrap());

    let points = grid.points().collect::<Vec<_>>();
    let centers = grid.bin_centers().collect::<Vec<_>>();
    let widths = grid.bin_widths().collect::<Vec<_>>();

    assert_slice_close(&points, expected_points);
    assert_slice_close(&centers, expected_centers);
    assert_slice_close(&widths, expected_widths);

    for index in 0..grid.bin_count() {
        assert_close(grid.point(index).unwrap(), expected_points[index]);
        assert_close(grid.bin_center(index).unwrap(), expected_centers[index]);
        assert_close(grid.bin_width(index).unwrap(), expected_widths[index]);
        assert_eq!(grid.bin_index(expected_centers[index]), Some(index));
    }

    assert_eq!(grid.point(grid.len()), None);
    assert_eq!(grid.bin_center(grid.bin_count()), None);
    assert_eq!(grid.bin_width(grid.bin_count()), None);
    assert_eq!(grid.bin_bounds(grid.bin_count()), None);
    assert_eq!(grid.bin_index(grid.last()), Some(grid.bin_count() - 1));
}

#[test]
fn invalid_grid_inputs_are_rejected() {
    assert_eq!(
        LinearGrid::new(0.0, 1.0, 1).unwrap_err().to_string(),
        "grid size must be at least 2, got 1"
    );
    assert_eq!(
        LinearGrid::new(1.0, 1.0, 2).unwrap_err().to_string(),
        "grid endpoints must differ, got first=1 and last=1"
    );
    assert_eq!(
        PowerGrid::new(0.0, 1.0, 2, 0.0).unwrap_err().to_string(),
        "power must be > 0, got 0"
    );
    assert_eq!(
        SymmetricPowerGrid::new(0.0, 1.0, 4, 2.0)
            .unwrap_err()
            .to_string(),
        "symmetric power grid size must be odd and at least 3, got 4"
    );
    assert_eq!(
        CustomGrid::new([0.0]).unwrap_err().to_string(),
        "grid size must be at least 2, got 1"
    );
    assert_eq!(
        CustomGrid::new([0.0, 0.0]).unwrap_err().to_string(),
        "custom grid points must be strictly monotonic"
    );
    assert_eq!(
        CustomGrid::new([0.0, 2.0, 1.0]).unwrap_err().to_string(),
        "custom grid points must be strictly monotonic"
    );
    assert_eq!(
        CustomGrid::new([0.0, f64::NAN]).unwrap_err().to_string(),
        "custom grid points must be strictly monotonic"
    );
}

#[test]
fn linear_grid_supports_increasing_and_decreasing_ranges() {
    let increasing = LinearGrid::new(0.0, 10.0, 6).unwrap();
    assert_close(increasing.step(), 2.0);
    assert!(increasing.is_increasing());
    assert!(!increasing.is_decreasing());
    assert_eq!(increasing.domain(), (0.0, 10.0));
    assert_eq!(increasing.bin_bounds(2), Some((4.0, 6.0)));
    assert_grid(
        &increasing,
        &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        &[1.0, 3.0, 5.0, 7.0, 9.0],
        &[2.0; 5],
    );
    assert_eq!(increasing.bin_index(-1.0), None);
    assert_eq!(increasing.bin_index(2.0), Some(1));
    assert_eq!(increasing.bin_index(9.999), Some(4));
    assert_eq!(increasing.bin_index(11.0), None);

    let decreasing = LinearGrid::new(10.0, 0.0, 6).unwrap();
    assert_close(decreasing.step(), -2.0);
    assert!(!decreasing.is_increasing());
    assert!(decreasing.is_decreasing());
    assert_eq!(decreasing.domain(), (10.0, 0.0));
    assert_eq!(decreasing.bin_bounds(2), Some((6.0, 4.0)));
    assert_grid(
        &decreasing,
        &[10.0, 8.0, 6.0, 4.0, 2.0, 0.0],
        &[9.0, 7.0, 5.0, 3.0, 1.0],
        &[2.0; 5],
    );
    assert_eq!(decreasing.bin_index(11.0), None);
    assert_eq!(decreasing.bin_index(8.0), Some(1));
    assert_eq!(decreasing.bin_index(0.0), Some(4));
    assert_eq!(decreasing.bin_index(-1.0), None);
}

#[test]
fn custom_grid_uses_supplied_points_for_bins_and_geometry() {
    let increasing = CustomGrid::new([1.0, 2.3, 2.4, 5.7, 100.0]).unwrap();
    assert_grid(
        &increasing,
        &[1.0, 2.3, 2.4, 5.7, 100.0],
        &[1.65, 2.35, 4.05, 52.85],
        &[1.3, 0.1, 3.3, 94.3],
    );
    assert_slice_close(increasing.grid_points(), &[1.0, 2.3, 2.4, 5.7, 100.0]);
    assert_eq!(increasing.bin_index(0.99), None);
    assert_eq!(increasing.bin_index(1.0), Some(0));
    assert_eq!(increasing.bin_index(2.3), Some(1));
    assert_eq!(increasing.bin_index(99.0), Some(3));
    assert_eq!(increasing.bin_index(100.0), Some(3));
    assert_eq!(increasing.bin_index(100.01), None);

    let decreasing = CustomGrid::new([100.0, 5.7, 2.4, 2.3, 1.0]).unwrap();
    assert_grid(
        &decreasing,
        &[100.0, 5.7, 2.4, 2.3, 1.0],
        &[52.85, 4.05, 2.35, 1.65],
        &[94.3, 3.3, 0.1, 1.3],
    );
    assert_eq!(decreasing.bin_index(100.01), None);
    assert_eq!(decreasing.bin_index(100.0), Some(0));
    assert_eq!(decreasing.bin_index(99.0), Some(0));
    assert_eq!(decreasing.bin_index(2.4), Some(2));
    assert_eq!(decreasing.bin_index(1.0), Some(3));
    assert_eq!(decreasing.bin_index(0.99), None);
}

#[test]
fn axis_grid_wraps_all_1d_grid_variants() {
    let linear = AxisGrid::from(LinearGrid::new(0.0, 2.0, 3).unwrap());
    assert_eq!(linear.kind(), "linear");
    assert_eq!(linear.len(), 3);
    assert_eq!(linear.bin_index(1.5), Some(1));
    assert_close(linear.bin_center(1).unwrap(), 1.5);

    let power = AxisGrid::from(PowerGrid::new(1.0, 10.0, 4, 2.0).unwrap());
    assert_eq!(power.kind(), "power");
    assert_slice_close(&power.points().collect::<Vec<_>>(), &[1.0, 2.0, 5.0, 10.0]);
    assert_eq!(power.bin_index(9.9), Some(2));

    let symmetric = AxisGrid::from(SymmetricPowerGrid::new(0.0, 1.0, 5, 2.0).unwrap());
    assert_eq!(symmetric.kind(), "symmetric_power");
    assert_eq!(symmetric.bin_index(0.5), Some(2));

    let custom = AxisGrid::from(CustomGrid::new([10.0, 7.0, 1.0]).unwrap());
    assert_eq!(custom.kind(), "custom");
    assert_eq!(custom.bin_index(6.0), Some(1));
    assert_slice_close(&custom.bin_widths().collect::<Vec<_>>(), &[3.0, 6.0]);
}

#[test]
fn power_grid_matches_closed_form_points_and_bins() {
    let grid = PowerGrid::new(3.0, 21.0, 4, 2.0).unwrap();

    assert_close(grid.power(), 2.0);
    assert_close(grid.scale(), 2.0);
    assert_grid(
        &grid,
        &[3.0, 5.0, 11.0, 21.0],
        &[4.0, 8.0, 16.0],
        &[2.0, 6.0, 10.0],
    );
    assert_eq!(grid.bin_index(3.1), Some(0));
    assert_eq!(grid.bin_index(10.9), Some(1));
    assert_eq!(grid.bin_index(11.0), Some(2));
    assert_eq!(grid.bin_index(21.0), Some(2));

    let decreasing = PowerGrid::new(-3.0, -21.0, 4, 2.0).unwrap();
    assert_close(decreasing.scale(), -2.0);
    assert_grid(
        &decreasing,
        &[-3.0, -5.0, -11.0, -21.0],
        &[-4.0, -8.0, -16.0],
        &[2.0, 6.0, 10.0],
    );
    assert_eq!(decreasing.bin_index(-3.1), Some(0));
    assert_eq!(decreasing.bin_index(-11.0), Some(2));
    assert_eq!(decreasing.bin_index(-21.0), Some(2));
}

#[test]
fn symmetric_power_grid_mirrors_two_power_grids() {
    let grid = SymmetricPowerGrid::new(0.0, 1.0, 9, 2.0).unwrap();

    assert_close(grid.midpoint(), 0.5);
    assert_close(grid.power(), 2.0);
    assert_grid(
        &grid,
        &[
            0.0, 0.03125, 0.125, 0.28125, 0.5, 0.71875, 0.875, 0.96875, 1.0,
        ],
        &[
            0.015625, 0.078125, 0.203125, 0.390625, 0.609375, 0.796875, 0.921875, 0.984375,
        ],
        &[
            0.03125, 0.09375, 0.15625, 0.21875, 0.21875, 0.15625, 0.09375, 0.03125,
        ],
    );
    assert_eq!(grid.bin_index(0.015), Some(0));
    assert_eq!(grid.bin_index(0.5), Some(4));
    assert_eq!(grid.bin_index(0.99), Some(7));

    let decreasing = SymmetricPowerGrid::new(1.0, 0.0, 9, 2.0).unwrap();
    assert_grid(
        &decreasing,
        &[
            1.0, 0.96875, 0.875, 0.71875, 0.5, 0.28125, 0.125, 0.03125, 0.0,
        ],
        &[
            0.984375, 0.921875, 0.796875, 0.609375, 0.390625, 0.203125, 0.078125, 0.015625,
        ],
        &[
            0.03125, 0.09375, 0.15625, 0.21875, 0.21875, 0.15625, 0.09375, 0.03125,
        ],
    );
    assert_eq!(decreasing.bin_index(0.99), Some(0));
    assert_eq!(decreasing.bin_index(0.5), Some(4));
    assert_eq!(decreasing.bin_index(0.01), Some(7));
}

#[test]
fn integer_subrange_matches_cpp_boundary_cases() {
    assert_eq!(integer_subrange(4, 9, 1), Some(4));
    assert_eq!(integer_subrange(4, 9, 2), Some(4));
    assert_eq!(integer_subrange(4, 9, 3), Some(3));
    assert_eq!(integer_subrange(4, 9, 6), Some(2));
    assert_eq!(integer_subrange(4, 9, 8), Some(1));
    assert_eq!(integer_subrange(4, 9, 9), Some(0));
    assert_eq!(integer_subrange(7, 9, 1), Some(7));
    assert_eq!(integer_subrange(7, 9, 2), Some(7));
    assert_eq!(integer_subrange(7, 9, 3), Some(6));
    assert_eq!(integer_subrange(7, 9, 4), Some(5));
    assert_eq!(integer_subrange(7, 9, 7), Some(2));
    assert_eq!(integer_subrange(1, 9, 1), Some(1));
    assert_eq!(integer_subrange(1, 9, 2), Some(1));
    assert_eq!(integer_subrange(1, 9, 3), Some(0));
    assert_eq!(integer_subrange(1, 9, 4), Some(0));
    assert_eq!(integer_subrange(1, 9, 9), Some(0));

    assert_eq!(integer_subrange(9, 9, 1), None);
    assert_eq!(integer_subrange(0, 9, 0), None);
    assert_eq!(integer_subrange(0, 9, 10), None);
}

#[test]
fn index_subrange_centers_grid_subranges_around_values() {
    let increasing = LinearGrid::new(0.0, 8.0, 9).unwrap();
    assert_eq!(index_subrange(&increasing, 1, 4.2), Some(4));
    assert_eq!(index_subrange(&increasing, 2, 4.2), Some(4));
    assert_eq!(index_subrange(&increasing, 3, 4.2), Some(3));
    assert_eq!(index_subrange(&increasing, 6, 4.2), Some(2));
    assert_eq!(index_subrange(&increasing, 7, 7.5), Some(2));
    assert_eq!(index_subrange(&increasing, 9, 1.2), Some(0));
    assert_eq!(index_subrange(&increasing, 1, -0.1), None);
    assert_eq!(index_subrange(&increasing, 0, 4.2), None);
    assert_eq!(index_subrange(&increasing, 10, 4.2), None);

    let decreasing = LinearGrid::new(8.0, 0.0, 9).unwrap();
    assert_eq!(index_subrange(&decreasing, 3, 4.2), Some(2));
    assert_eq!(index_subrange(&decreasing, 7, 0.5), Some(2));
    assert_eq!(index_subrange(&decreasing, 9, 6.8), Some(0));
    assert_eq!(index_subrange(&decreasing, 1, -0.1), None);
}

#[test]
fn iterators_report_exact_remaining_lengths() {
    let grid = LinearGrid::new(0.0, 1.0, 5).unwrap();
    let mut points = grid.points();
    let mut centers = grid.bin_centers();
    let mut widths = grid.bin_widths();

    assert_eq!(points.len(), 5);
    assert_eq!(centers.len(), 4);
    assert_eq!(widths.len(), 4);
    assert_eq!(points.next(), Some(0.0));
    assert_eq!(centers.next(), Some(0.125));
    assert_eq!(widths.next(), Some(0.25));
    assert_eq!(points.len(), 4);
    assert_eq!(centers.len(), 3);
    assert_eq!(widths.len(), 3);
}

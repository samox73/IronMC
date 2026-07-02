use rmc_grids::{
    flat_index_row_major, nd_index_row_major, size_from_shape, AxisGrid, CustomGrid, Grid1d,
    LinearGrid, NdGrid, PowerGrid,
};

fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0e-12;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn assert_array_close<const N: usize>(actual: [f64; N], expected: [f64; N]) {
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_close(*actual, *expected);
    }
}

#[test]
fn row_major_index_helpers_round_trip_indices() {
    let shape = [2, 3, 4];
    assert_eq!(size_from_shape(shape), Some(24));
    assert_eq!(flat_index_row_major([0, 0, 0], shape), Some(0));
    assert_eq!(flat_index_row_major([0, 0, 1], shape), Some(1));
    assert_eq!(flat_index_row_major([0, 1, 0], shape), Some(4));
    assert_eq!(flat_index_row_major([1, 2, 3], shape), Some(23));
    assert_eq!(flat_index_row_major([2, 0, 0], shape), None);

    for flat_index in 0..24 {
        let indices = nd_index_row_major(flat_index, shape).unwrap();
        assert_eq!(flat_index_row_major(indices, shape), Some(flat_index));
    }

    assert_eq!(nd_index_row_major(24, shape), None);
    assert_eq!(size_from_shape::<0>([]), None);
    assert_eq!(size_from_shape([2, 0, 4]), None);
}

#[test]
fn nd_grid_composes_linear_axes_in_row_major_order() {
    let x = LinearGrid::new(0.0, 2.0, 3).unwrap();
    let y = LinearGrid::new(10.0, 13.0, 4).unwrap();
    let grid = NdGrid::new([x, y]).unwrap();

    assert_eq!(grid.dim(), 2);
    assert_eq!(grid.shape(), [3, 4]);
    assert_eq!(grid.bin_shape(), [2, 3]);
    assert_eq!(grid.len(), 12);
    assert_eq!(grid.bin_count(), 6);
    assert_array_close(grid.first(), [0.0, 10.0]);
    assert_array_close(grid.last(), [2.0, 13.0]);
    assert!(grid.contains([0.2, 10.7]));
    assert!(grid.contains([2.0, 13.0]));
    assert!(!grid.contains([2.1, 13.0]));

    assert_array_close(grid.point([0, 0]).unwrap(), [0.0, 10.0]);
    assert_array_close(grid.point([1, 2]).unwrap(), [1.0, 12.0]);
    assert_array_close(grid.point([2, 3]).unwrap(), [2.0, 13.0]);
    assert_eq!(grid.point([3, 0]), None);

    assert_eq!(grid.bin_index([0.2, 10.7]), Some([0, 0]));
    assert_eq!(grid.bin_index([1.2, 12.7]), Some([1, 2]));
    assert_eq!(grid.bin_index([2.0, 13.0]), Some([1, 2]));
    assert_eq!(grid.bin_index([2.1, 13.0]), None);

    assert_array_close(grid.bin_center([1, 2]).unwrap(), [1.5, 12.5]);
    let (lower, upper) = grid.bin_bounds([1, 2]).unwrap();
    assert_array_close(lower, [1.0, 12.0]);
    assert_array_close(upper, [2.0, 13.0]);
    assert_eq!(grid.bin_bounds([2, 0]), None);
    assert_close(grid.bin_volume([1, 2]).unwrap(), 1.0);
    assert_eq!(grid.bin_center([2, 0]), None);
    assert_eq!(grid.bin_volume([0, 3]), None);

    let point_indices = grid.point_indices().collect::<Vec<_>>();
    assert_eq!(point_indices.len(), 12);
    assert_eq!(point_indices[0], [0, 0]);
    assert_eq!(point_indices[1], [0, 1]);
    assert_eq!(point_indices[4], [1, 0]);
    assert_eq!(point_indices[11], [2, 3]);

    let bin_indices = grid.bin_indices().collect::<Vec<_>>();
    assert_eq!(
        bin_indices,
        vec![[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    );

    let points = grid.points().collect::<Vec<_>>();
    assert_eq!(points.len(), 12);
    assert_array_close(points[0], [0.0, 10.0]);
    assert_array_close(points[1], [0.0, 11.0]);
    assert_array_close(points[3], [0.0, 13.0]);
    assert_array_close(points[4], [1.0, 10.0]);
    assert_array_close(points[11], [2.0, 13.0]);

    let centers = grid.bin_centers().collect::<Vec<_>>();
    let volumes = grid.bin_volumes().collect::<Vec<_>>();
    assert_eq!(centers.len(), 6);
    assert_eq!(volumes, vec![1.0; 6]);
    assert_array_close(centers[0], [0.5, 10.5]);
    assert_array_close(centers[1], [0.5, 11.5]);
    assert_array_close(centers[3], [1.5, 10.5]);
    assert_array_close(centers[5], [1.5, 12.5]);
}

#[test]
fn nd_grid_flat_access_uses_grid_and_bin_shapes() {
    let axis = LinearGrid::new(0.0, 4.0, 5).unwrap();
    let grid = NdGrid::new([axis, axis, axis]).unwrap();

    assert_eq!(grid.flat_index([0, 0, 0]), Some(0));
    assert_eq!(grid.flat_index([0, 0, 1]), Some(1));
    assert_eq!(grid.flat_index([0, 1, 0]), Some(5));
    assert_eq!(grid.flat_index([4, 4, 4]), Some(124));
    assert_eq!(grid.flat_index([5, 0, 0]), None);
    assert_array_close(grid.point_at_flat(31).unwrap(), [1.0, 1.0, 1.0]);
    assert_eq!(grid.point_at_flat(125), None);

    assert_eq!(grid.flat_bin_index([0, 0, 0]), Some(0));
    assert_eq!(grid.flat_bin_index([0, 1, 0]), Some(4));
    assert_eq!(grid.flat_bin_index([3, 3, 3]), Some(63));
    assert_eq!(grid.flat_bin_index([4, 0, 0]), None);
    assert_array_close(grid.bin_center_at_flat(21).unwrap(), [1.5, 1.5, 1.5]);
    assert_close(grid.bin_volume_at_flat(21).unwrap(), 1.0);
    assert_eq!(grid.bin_center_at_flat(64), None);
}

#[test]
fn nd_grid_subrange_and_decreasing_axes_delegate_to_1d_grids() {
    let increasing = LinearGrid::new(0.0, 10.0, 11).unwrap();
    let decreasing = LinearGrid::new(10.0, 0.0, 11).unwrap();

    let inc_grid = NdGrid::new([increasing, increasing]).unwrap();
    assert_eq!(inc_grid.index_subrange(5, [2.2, 8.2]), Some([0, 6]));
    assert_eq!(inc_grid.index_subrange(2, [10.0, 9.5]), Some([9, 9]));
    assert_eq!(inc_grid.index_subrange(0, [2.2, 8.2]), None);
    assert_eq!(inc_grid.index_subrange(12, [2.2, 8.2]), None);
    assert_eq!(inc_grid.index_subrange(2, [-0.1, 8.2]), None);

    let dec_grid = NdGrid::new([decreasing, decreasing]).unwrap();
    assert_eq!(dec_grid.index_subrange(5, [2.2, 8.2]), Some([5, 0]));
    assert_eq!(dec_grid.index_subrange(2, [0.0, 0.5]), Some([9, 9]));
    assert_eq!(dec_grid.index_subrange(2, [10.1, 8.2]), None);
}

#[test]
fn nd_grid_validates_dimension_and_axis_sizes() {
    let err = NdGrid::<LinearGrid, 0>::new([]).unwrap_err();
    assert_eq!(err.to_string(), "grid dimension must be at least 1, got 0");

    #[derive(Clone, Debug)]
    struct EmptyAxis;

    impl Grid1d for EmptyAxis {
        fn len(&self) -> usize {
            0
        }

        fn first(&self) -> f64 {
            0.0
        }

        fn last(&self) -> f64 {
            0.0
        }

        fn point(&self, _index: usize) -> Option<f64> {
            None
        }

        fn bin_index(&self, _value: f64) -> Option<usize> {
            None
        }
    }

    let err = NdGrid::new([EmptyAxis]).unwrap_err();
    assert_eq!(err.to_string(), "axis 0 must have at least 2 points, got 0");
}

#[test]
fn nd_grid_works_with_custom_axes() {
    let axis = CustomGrid::new([1.0, 2.0, 4.0]).unwrap();
    let grid = NdGrid::new([axis.clone(), axis]).unwrap();

    assert_eq!(grid.shape(), [3, 3]);
    assert_array_close(grid.point([2, 1]).unwrap(), [4.0, 2.0]);
    assert_eq!(grid.bin_index([3.5, 1.5]), Some([1, 0]));
    assert_array_close(grid.bin_center([1, 0]).unwrap(), [3.0, 1.5]);
    assert_close(grid.bin_volume([1, 0]).unwrap(), 2.0);
}

#[test]
fn nd_grid_supports_mixed_axis_types_via_axis_grid() {
    let x = AxisGrid::from(LinearGrid::new(0.0, 2.0, 3).unwrap());
    let y = AxisGrid::from(PowerGrid::new(1.0, 10.0, 4, 2.0).unwrap());
    let z = AxisGrid::from(CustomGrid::new([10.0, 7.0, 1.0]).unwrap());
    let grid = NdGrid::new([x, y, z]).unwrap();

    assert_eq!(grid.shape(), [3, 4, 3]);
    assert_eq!(grid.bin_shape(), [2, 3, 2]);
    assert_array_close(grid.first(), [0.0, 1.0, 10.0]);
    assert_array_close(grid.last(), [2.0, 10.0, 1.0]);
    assert_eq!(grid.axis(0).unwrap().kind(), "linear");
    assert_eq!(grid.axis(1).unwrap().kind(), "power");
    assert_eq!(grid.axis(2).unwrap().kind(), "custom");

    assert_array_close(grid.point([2, 2, 1]).unwrap(), [2.0, 5.0, 7.0]);
    assert_eq!(grid.bin_index([1.2, 3.0, 6.0]), Some([1, 1, 1]));
    assert_array_close(grid.bin_center([1, 1, 1]).unwrap(), [1.5, 3.5, 4.0]);
    assert_close(grid.bin_volume([1, 1, 1]).unwrap(), 18.0);
    assert_eq!(grid.index_subrange(2, [1.2, 3.0, 6.0]), Some([1, 1, 1]));

    let points = grid.points().collect::<Vec<_>>();
    assert_eq!(points.len(), 36);
    assert_array_close(points[0], [0.0, 1.0, 10.0]);
    assert_array_close(points[1], [0.0, 1.0, 7.0]);
    assert_array_close(points[3], [0.0, 2.0, 10.0]);
    assert_array_close(points[35], [2.0, 10.0, 1.0]);
}

#![cfg(feature = "serde")]

use rmc_grids::{AxisGrid, CustomGrid, Grid1d, LinearGrid, NdGrid, PowerGrid, SymmetricPowerGrid};
use serde::de::DeserializeOwned;
use serde::Serialize;

fn round_trip<T>(value: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let encoded = serde_json::to_string(value).unwrap();
    serde_json::from_str(&encoded).unwrap()
}

#[test]
fn serde_round_trips_linear_grid() {
    let grid = LinearGrid::new(0.0, 10.0, 6).unwrap();
    let restored: LinearGrid = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.len(), 6);
}

#[test]
fn serde_round_trips_power_grid() {
    let grid = PowerGrid::new(3.0, 21.0, 4, 2.0).unwrap();
    let restored: PowerGrid = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.len(), 4);
    assert_eq!(restored.power(), 2.0);
}

#[test]
fn serde_round_trips_symmetric_power_grid() {
    let grid = SymmetricPowerGrid::new(0.0, 1.0, 9, 2.0).unwrap();
    let restored: SymmetricPowerGrid = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.len(), 9);
    assert_eq!(restored.bin_index(0.99), Some(7));
}

#[test]
fn serde_round_trips_custom_grid() {
    let grid = CustomGrid::new([1.0, 2.3, 2.4, 5.7, 100.0]).unwrap();
    let restored: CustomGrid = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.grid_points(), &[1.0, 2.3, 2.4, 5.7, 100.0]);
    assert_eq!(restored.bin_index(99.0), Some(3));
}

#[test]
fn serde_round_trips_nd_grid() {
    let axis = LinearGrid::new(0.0, 2.0, 3).unwrap();
    let grid = NdGrid::new([axis, axis]).unwrap();
    let restored: NdGrid<LinearGrid, 2> = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.shape(), [3, 3]);
    assert_eq!(restored.bin_index([1.2, 2.0]), Some([1, 1]));
}

#[test]
fn serde_round_trips_axis_grid_and_mixed_nd_grid() {
    let axis = AxisGrid::from(PowerGrid::new(1.0, 10.0, 4, 2.0).unwrap());
    let restored: AxisGrid = round_trip(&axis);

    assert_eq!(restored, axis);
    assert_eq!(restored.kind(), "power");
    assert_eq!(restored.bin_index(9.9), Some(2));

    let grid = NdGrid::new([
        AxisGrid::from(LinearGrid::new(0.0, 2.0, 3).unwrap()),
        AxisGrid::from(PowerGrid::new(1.0, 10.0, 4, 2.0).unwrap()),
        AxisGrid::from(CustomGrid::new([10.0, 7.0, 1.0]).unwrap()),
    ])
    .unwrap();
    let restored: NdGrid<AxisGrid, 3> = round_trip(&grid);

    assert_eq!(restored, grid);
    assert_eq!(restored.shape(), [3, 4, 3]);
    assert_eq!(restored.bin_index([1.2, 3.0, 6.0]), Some([1, 1, 1]));
}

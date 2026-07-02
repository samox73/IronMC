//! Numerical utilities for interpolation, quadrature, and later lattice helpers.
//!
//! The first MVP slices provide checked linear interpolation on `rmc-grids` grids and basic
//! one-dimensional quadrature for smooth finite intervals.

use rmc_grids::{index_subrange, AxisGrid, Grid1d, NdGrid};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, NumericError>;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum NumericError {
    #[error("number of function values ({values}) does not match grid points ({grid_points})")]
    ValueCountMismatch { grid_points: usize, values: usize },
    #[error("point lies outside the interpolation domain")]
    OutOfDomain,
    #[error("interpolation cell has zero width on axis {axis}")]
    DegenerateCell { axis: usize },
    #[error("N-D interpolation supports at most {max} dimensions, got {actual}")]
    DimensionTooLarge { max: usize, actual: usize },
    #[error("polynomial interpolation requires at least one point, got {points}")]
    NotEnoughInterpolationPoints { points: usize },
    #[error("non-finite interpolation {kind} at index {index}: {value}")]
    NonFiniteInterpolationData {
        kind: &'static str,
        index: usize,
        value: f64,
    },
    #[error("duplicate interpolation node at indices {first} and {second}: {value}")]
    DuplicateInterpolationNode {
        first: usize,
        second: usize,
        value: f64,
    },
    #[error("evaluation point must be finite, got {x}")]
    NonFiniteEvaluationPoint { x: f64 },
    #[error("cubic spline interpolation requires at least two points, got {points}")]
    NotEnoughSplinePoints { points: usize },
    #[error("non-finite spline boundary derivative {which}: {value}")]
    NonFiniteSplineBoundaryDerivative { which: &'static str, value: f64 },
    #[error("singular spline system at row {row}")]
    SingularSplineSystem { row: usize },
    #[error("quadrature interval bounds must be finite, got [{start}, {end}]")]
    NonFiniteInterval { start: f64, end: f64 },
    #[error("{rule} quadrature requires {required}, got {actual}")]
    InvalidPanelCount {
        rule: &'static str,
        required: &'static str,
        actual: usize,
    },
    #[error("quadrature tolerances must be finite and non-negative, got abs={abs}, rel={rel}")]
    InvalidTolerance { abs: f64, rel: f64 },
    #[error("integrand returned non-finite value {value} at x={x}")]
    NonFiniteFunctionValue { x: f64, value: f64 },
    #[error("adaptive Simpson quadrature exceeded maximum recursion depth {max_depth}")]
    MaxDepthExceeded { max_depth: u32 },
}

/// One-dimensional piecewise-linear interpolation over a `Grid1d`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearInterpolation<G> {
    grid: G,
    values: Vec<f64>,
}

impl<G: Grid1d> LinearInterpolation<G> {
    pub fn new(grid: G, values: impl Into<Vec<f64>>) -> Result<Self> {
        let values = values.into();
        validate_value_count(grid.len(), values.len())?;
        Ok(Self { grid, values })
    }

    pub fn grid(&self) -> &G {
        &self.grid
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn evaluate(&self, x: f64) -> Result<f64> {
        let index = index_subrange(&self.grid, 2, x).ok_or(NumericError::OutOfDomain)?;
        let x0 = self.grid.point(index).ok_or(NumericError::OutOfDomain)?;
        let x1 = self
            .grid
            .point(index + 1)
            .ok_or(NumericError::OutOfDomain)?;
        let denominator = x1 - x0;
        if denominator == 0.0 {
            return Err(NumericError::DegenerateCell { axis: 0 });
        }

        Ok(interp_linear_1d(
            (x - x0) / denominator,
            self.values[index],
            self.values[index + 1],
        ))
    }
}

pub type LinearInterpolation1d = LinearInterpolation<AxisGrid>;

/// Const-generic multilinear interpolation over an `NdGrid`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "G: serde::Serialize",
        deserialize = "G: Grid1d + serde::Deserialize<'de>"
    ))
)]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearInterpolationNd<G, const N: usize> {
    grid: NdGrid<G, N>,
    values: Vec<f64>,
}

impl<G: Grid1d, const N: usize> LinearInterpolationNd<G, N> {
    pub fn new(grid: NdGrid<G, N>, values: impl Into<Vec<f64>>) -> Result<Self> {
        let values = values.into();
        validate_value_count(grid.len(), values.len())?;
        Ok(Self { grid, values })
    }

    pub fn grid(&self) -> &NdGrid<G, N> {
        &self.grid
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn evaluate(&self, point: [f64; N]) -> Result<f64> {
        if N >= usize::BITS as usize {
            return Err(NumericError::DimensionTooLarge {
                max: usize::BITS as usize - 1,
                actual: N,
            });
        }

        let base = self
            .grid
            .index_subrange(2, point)
            .ok_or(NumericError::OutOfDomain)?;
        let ratios = self.distance_ratios(base, point)?;

        let mut interpolated = 0.0;
        for corner in 0..(1usize << N) {
            let mut indices = base;
            let mut weight = 1.0;
            for axis in 0..N {
                if (corner >> axis) & 1 == 0 {
                    weight *= 1.0 - ratios[axis];
                } else {
                    indices[axis] += 1;
                    weight *= ratios[axis];
                }
            }
            let flat_index = self
                .grid
                .flat_index(indices)
                .ok_or(NumericError::OutOfDomain)?;
            interpolated += weight * self.values[flat_index];
        }

        Ok(interpolated)
    }

    fn distance_ratios(&self, base: [usize; N], point: [f64; N]) -> Result<[f64; N]> {
        let mut ratios = [0.0; N];
        for axis in 0..N {
            let grid = self.grid.axis(axis).ok_or(NumericError::OutOfDomain)?;
            let x0 = grid.point(base[axis]).ok_or(NumericError::OutOfDomain)?;
            let x1 = grid
                .point(base[axis] + 1)
                .ok_or(NumericError::OutOfDomain)?;
            let denominator = x1 - x0;
            if denominator == 0.0 {
                return Err(NumericError::DegenerateCell { axis });
            }
            ratios[axis] = (point[axis] - x0) / denominator;
        }
        Ok(ratios)
    }
}

pub type LinearInterpolationMixed<const N: usize> = LinearInterpolationNd<AxisGrid, N>;

/// One-dimensional cubic spline interpolation over a `Grid1d`.
///
/// The spline stores the grid values and the solved second derivatives at each grid point. Natural
/// boundary conditions set the endpoint second derivatives to zero; clamped boundary conditions
/// prescribe the first derivatives at the two endpoints.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct CubicSplineInterpolation<G> {
    grid: G,
    values: Vec<f64>,
    second_derivatives: Vec<f64>,
}

impl<G: Grid1d> CubicSplineInterpolation<G> {
    pub fn natural(grid: G, values: impl Into<Vec<f64>>) -> Result<Self> {
        let values = values.into();
        validate_spline_inputs(&grid, &values)?;
        let second_derivatives = natural_spline_second_derivatives(&grid, &values)?;

        Ok(Self {
            grid,
            values,
            second_derivatives,
        })
    }

    pub fn with_endpoint_derivatives(
        grid: G,
        values: impl Into<Vec<f64>>,
        left_derivative: f64,
        right_derivative: f64,
    ) -> Result<Self> {
        let values = values.into();
        validate_spline_inputs(&grid, &values)?;
        validate_boundary_derivative("left", left_derivative)?;
        validate_boundary_derivative("right", right_derivative)?;
        let second_derivatives =
            clamped_spline_second_derivatives(&grid, &values, left_derivative, right_derivative)?;

        Ok(Self {
            grid,
            values,
            second_derivatives,
        })
    }

    pub fn grid(&self) -> &G {
        &self.grid
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn second_derivatives(&self) -> &[f64] {
        &self.second_derivatives
    }

    pub fn evaluate(&self, x: f64) -> Result<f64> {
        if !x.is_finite() {
            return Err(NumericError::NonFiniteEvaluationPoint { x });
        }

        let index = index_subrange(&self.grid, 2, x).ok_or(NumericError::OutOfDomain)?;
        let x0 = self.grid.point(index).ok_or(NumericError::OutOfDomain)?;
        let x1 = self
            .grid
            .point(index + 1)
            .ok_or(NumericError::OutOfDomain)?;
        let width = x1 - x0;
        if width == 0.0 {
            return Err(NumericError::DegenerateCell { axis: 0 });
        }

        let left_weight = (x1 - x) / width;
        let right_weight = (x - x0) / width;
        Ok(left_weight * self.values[index]
            + right_weight * self.values[index + 1]
            + ((left_weight * left_weight * left_weight - left_weight)
                * self.second_derivatives[index]
                + (right_weight * right_weight * right_weight - right_weight)
                    * self.second_derivatives[index + 1])
                * width
                * width
                / 6.0)
    }
}

pub type CubicSplineInterpolation1d = CubicSplineInterpolation<AxisGrid>;

/// Global polynomial interpolation in first-form barycentric coordinates.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialInterpolation {
    nodes: Vec<f64>,
    values: Vec<f64>,
    weights: Vec<f64>,
}

impl PolynomialInterpolation {
    pub fn new(nodes: impl Into<Vec<f64>>, values: impl Into<Vec<f64>>) -> Result<Self> {
        let nodes = nodes.into();
        let values = values.into();
        validate_polynomial_inputs(&nodes, &values)?;
        let weights = barycentric_weights(&nodes)?;

        Ok(Self {
            nodes,
            values,
            weights,
        })
    }

    pub fn from_grid<G: Grid1d>(grid: &G, values: impl Into<Vec<f64>>) -> Result<Self> {
        Self::new(grid.points().collect::<Vec<_>>(), values)
    }

    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn degree(&self) -> usize {
        self.nodes.len() - 1
    }

    pub fn evaluate(&self, x: f64) -> Result<f64> {
        if !x.is_finite() {
            return Err(NumericError::NonFiniteEvaluationPoint { x });
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for ((&node, &value), &weight) in self.nodes.iter().zip(&self.values).zip(&self.weights) {
            if x == node {
                return Ok(value);
            }

            let scaled = weight / (x - node);
            numerator += scaled * value;
            denominator += scaled;
        }

        Ok(numerator / denominator)
    }
}

/// Composite trapezoidal-rule integration over a finite interval.
///
/// `panels` is the number of equal-width subintervals. Decreasing intervals are supported and
/// return the signed integral.
pub fn integrate_trapezoid<F>(mut f: F, start: f64, end: f64, panels: usize) -> Result<f64>
where
    F: FnMut(f64) -> f64,
{
    validate_interval(start, end)?;
    if panels == 0 {
        return Err(NumericError::InvalidPanelCount {
            rule: "trapezoid",
            required: "at least one panel",
            actual: panels,
        });
    }

    let step = (end - start) / panels as f64;
    let mut sum = 0.5 * checked_eval(&mut f, start)? + 0.5 * checked_eval(&mut f, end)?;
    for index in 1..panels {
        sum += checked_eval(&mut f, start + step * index as f64)?;
    }

    Ok(sum * step)
}

/// Composite Simpson-rule integration over a finite interval.
///
/// `panels` is the number of equal-width subintervals and must be positive and even. Decreasing
/// intervals are supported and return the signed integral.
pub fn integrate_simpson<F>(mut f: F, start: f64, end: f64, panels: usize) -> Result<f64>
where
    F: FnMut(f64) -> f64,
{
    validate_interval(start, end)?;
    if panels == 0 || panels % 2 != 0 {
        return Err(NumericError::InvalidPanelCount {
            rule: "simpson",
            required: "a positive even panel count",
            actual: panels,
        });
    }

    let step = (end - start) / panels as f64;
    let mut sum = checked_eval(&mut f, start)? + checked_eval(&mut f, end)?;
    for index in 1..panels {
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        sum += weight * checked_eval(&mut f, start + step * index as f64)?;
    }

    Ok(sum * step / 3.0)
}

/// Configuration for adaptive Simpson quadrature over a finite interval.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdaptiveSimpson {
    abs_tolerance: f64,
    rel_tolerance: f64,
    max_depth: u32,
}

impl AdaptiveSimpson {
    pub fn new(abs_tolerance: f64, rel_tolerance: f64, max_depth: u32) -> Result<Self> {
        if !abs_tolerance.is_finite()
            || !rel_tolerance.is_finite()
            || abs_tolerance < 0.0
            || rel_tolerance < 0.0
        {
            return Err(NumericError::InvalidTolerance {
                abs: abs_tolerance,
                rel: rel_tolerance,
            });
        }

        Ok(Self {
            abs_tolerance,
            rel_tolerance,
            max_depth,
        })
    }

    pub fn abs_tolerance(&self) -> f64 {
        self.abs_tolerance
    }

    pub fn rel_tolerance(&self) -> f64 {
        self.rel_tolerance
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    pub fn integrate<F>(&self, mut f: F, start: f64, end: f64) -> Result<f64>
    where
        F: FnMut(f64) -> f64,
    {
        validate_interval(start, end)?;
        if start == end {
            return Ok(0.0);
        }

        let initial = SimpsonPanel::new(&mut f, start, end)?;
        let scale = initial.whole.abs().max(1.0);
        let tolerance = self.abs_tolerance.max(self.rel_tolerance * scale);
        adaptive_simpson(&mut f, initial, tolerance, self.max_depth).map_err(|err| match err {
            NumericError::MaxDepthExceeded { .. } => NumericError::MaxDepthExceeded {
                max_depth: self.max_depth,
            },
            other => other,
        })
    }
}

impl Default for AdaptiveSimpson {
    fn default() -> Self {
        Self {
            abs_tolerance: 1.0e-10,
            rel_tolerance: 1.0e-10,
            max_depth: 32,
        }
    }
}

fn validate_value_count(grid_points: usize, values: usize) -> Result<()> {
    if grid_points != values {
        return Err(NumericError::ValueCountMismatch {
            grid_points,
            values,
        });
    }
    Ok(())
}

fn validate_spline_inputs<G: Grid1d>(grid: &G, values: &[f64]) -> Result<()> {
    if grid.len() < 2 {
        return Err(NumericError::NotEnoughSplinePoints { points: grid.len() });
    }
    validate_value_count(grid.len(), values.len())?;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(NumericError::NonFiniteInterpolationData {
                kind: "value",
                index,
                value,
            });
        }
    }
    Ok(())
}

fn validate_boundary_derivative(which: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(NumericError::NonFiniteSplineBoundaryDerivative { which, value });
    }
    Ok(())
}

fn natural_spline_second_derivatives<G: Grid1d>(grid: &G, values: &[f64]) -> Result<Vec<f64>> {
    let n = grid.len();
    let mut lower = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    diagonal[0] = 1.0;
    diagonal[n - 1] = 1.0;
    for i in 1..n - 1 {
        let x_prev = grid.point(i - 1).ok_or(NumericError::OutOfDomain)?;
        let x = grid.point(i).ok_or(NumericError::OutOfDomain)?;
        let x_next = grid.point(i + 1).ok_or(NumericError::OutOfDomain)?;
        let left_width = x - x_prev;
        let right_width = x_next - x;
        if left_width == 0.0 || right_width == 0.0 {
            return Err(NumericError::DegenerateCell { axis: 0 });
        }

        lower[i] = left_width;
        diagonal[i] = 2.0 * (x_next - x_prev);
        upper[i] = right_width;
        rhs[i] = 6.0
            * ((values[i + 1] - values[i]) / right_width
                - (values[i] - values[i - 1]) / left_width);
    }

    solve_tridiagonal(lower, diagonal, upper, rhs)
}

fn clamped_spline_second_derivatives<G: Grid1d>(
    grid: &G,
    values: &[f64],
    left_derivative: f64,
    right_derivative: f64,
) -> Result<Vec<f64>> {
    let n = grid.len();
    let mut lower = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    let x0 = grid.point(0).ok_or(NumericError::OutOfDomain)?;
    let x1 = grid.point(1).ok_or(NumericError::OutOfDomain)?;
    let left_width = x1 - x0;
    if left_width == 0.0 {
        return Err(NumericError::DegenerateCell { axis: 0 });
    }
    diagonal[0] = 2.0 * left_width;
    upper[0] = left_width;
    rhs[0] = 6.0 * ((values[1] - values[0]) / left_width - left_derivative);

    for i in 1..n - 1 {
        let x_prev = grid.point(i - 1).ok_or(NumericError::OutOfDomain)?;
        let x = grid.point(i).ok_or(NumericError::OutOfDomain)?;
        let x_next = grid.point(i + 1).ok_or(NumericError::OutOfDomain)?;
        let left_width = x - x_prev;
        let right_width = x_next - x;
        if left_width == 0.0 || right_width == 0.0 {
            return Err(NumericError::DegenerateCell { axis: 0 });
        }

        lower[i] = left_width;
        diagonal[i] = 2.0 * (x_next - x_prev);
        upper[i] = right_width;
        rhs[i] = 6.0
            * ((values[i + 1] - values[i]) / right_width
                - (values[i] - values[i - 1]) / left_width);
    }

    let x_prev = grid.point(n - 2).ok_or(NumericError::OutOfDomain)?;
    let x_last = grid.point(n - 1).ok_or(NumericError::OutOfDomain)?;
    let right_width = x_last - x_prev;
    if right_width == 0.0 {
        return Err(NumericError::DegenerateCell { axis: 0 });
    }
    lower[n - 1] = right_width;
    diagonal[n - 1] = 2.0 * right_width;
    rhs[n - 1] = 6.0 * (right_derivative - (values[n - 1] - values[n - 2]) / right_width);

    solve_tridiagonal(lower, diagonal, upper, rhs)
}

fn solve_tridiagonal(
    lower: Vec<f64>,
    mut diagonal: Vec<f64>,
    upper: Vec<f64>,
    mut rhs: Vec<f64>,
) -> Result<Vec<f64>> {
    let n = diagonal.len();
    for i in 1..n {
        let pivot = diagonal[i - 1];
        if pivot == 0.0 {
            return Err(NumericError::SingularSplineSystem { row: i - 1 });
        }
        let multiplier = lower[i] / pivot;
        diagonal[i] -= multiplier * upper[i - 1];
        rhs[i] -= multiplier * rhs[i - 1];
    }

    let last = n - 1;
    if diagonal[last] == 0.0 {
        return Err(NumericError::SingularSplineSystem { row: last });
    }
    let mut solution = vec![0.0; n];
    solution[last] = rhs[last] / diagonal[last];
    for i in (0..last).rev() {
        if diagonal[i] == 0.0 {
            return Err(NumericError::SingularSplineSystem { row: i });
        }
        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / diagonal[i];
    }

    Ok(solution)
}

fn interp_linear_1d(ratio: f64, f0: f64, f1: f64) -> f64 {
    f0 * (1.0 - ratio) + f1 * ratio
}

fn validate_polynomial_inputs(nodes: &[f64], values: &[f64]) -> Result<()> {
    if nodes.is_empty() {
        return Err(NumericError::NotEnoughInterpolationPoints { points: 0 });
    }
    validate_value_count(nodes.len(), values.len())?;

    for (index, &node) in nodes.iter().enumerate() {
        if !node.is_finite() {
            return Err(NumericError::NonFiniteInterpolationData {
                kind: "node",
                index,
                value: node,
            });
        }
    }
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(NumericError::NonFiniteInterpolationData {
                kind: "value",
                index,
                value,
            });
        }
    }

    Ok(())
}

fn barycentric_weights(nodes: &[f64]) -> Result<Vec<f64>> {
    let mut weights = Vec::with_capacity(nodes.len());
    for (i, &node_i) in nodes.iter().enumerate() {
        let mut weight = 1.0;
        for (j, &node_j) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }
            let difference = node_i - node_j;
            if difference == 0.0 {
                return Err(NumericError::DuplicateInterpolationNode {
                    first: i.min(j),
                    second: i.max(j),
                    value: node_i,
                });
            }
            weight /= difference;
        }
        weights.push(weight);
    }
    Ok(weights)
}

fn validate_interval(start: f64, end: f64) -> Result<()> {
    if !start.is_finite() || !end.is_finite() {
        return Err(NumericError::NonFiniteInterval { start, end });
    }
    Ok(())
}

fn checked_eval<F>(f: &mut F, x: f64) -> Result<f64>
where
    F: FnMut(f64) -> f64,
{
    let value = f(x);
    if !value.is_finite() {
        return Err(NumericError::NonFiniteFunctionValue { x, value });
    }
    Ok(value)
}

#[derive(Clone, Copy, Debug)]
struct SimpsonPanel {
    start: f64,
    mid: f64,
    end: f64,
    f_start: f64,
    f_mid: f64,
    f_end: f64,
    whole: f64,
}

impl SimpsonPanel {
    fn new<F>(f: &mut F, start: f64, end: f64) -> Result<Self>
    where
        F: FnMut(f64) -> f64,
    {
        let mid = midpoint(start, end);
        let f_start = checked_eval(f, start)?;
        let f_mid = checked_eval(f, mid)?;
        let f_end = checked_eval(f, end)?;
        Ok(Self {
            start,
            mid,
            end,
            f_start,
            f_mid,
            f_end,
            whole: simpson_area(start, end, f_start, f_mid, f_end),
        })
    }

    fn bisect<F>(self, f: &mut F) -> Result<(Self, Self)>
    where
        F: FnMut(f64) -> f64,
    {
        let left_mid = midpoint(self.start, self.mid);
        let right_mid = midpoint(self.mid, self.end);
        let f_left_mid = checked_eval(f, left_mid)?;
        let f_right_mid = checked_eval(f, right_mid)?;

        let left = Self {
            start: self.start,
            mid: left_mid,
            end: self.mid,
            f_start: self.f_start,
            f_mid: f_left_mid,
            f_end: self.f_mid,
            whole: simpson_area(self.start, self.mid, self.f_start, f_left_mid, self.f_mid),
        };
        let right = Self {
            start: self.mid,
            mid: right_mid,
            end: self.end,
            f_start: self.f_mid,
            f_mid: f_right_mid,
            f_end: self.f_end,
            whole: simpson_area(self.mid, self.end, self.f_mid, f_right_mid, self.f_end),
        };
        Ok((left, right))
    }
}

fn adaptive_simpson<F>(f: &mut F, panel: SimpsonPanel, tolerance: f64, depth: u32) -> Result<f64>
where
    F: FnMut(f64) -> f64,
{
    let (left, right) = panel.bisect(f)?;
    let refined = left.whole + right.whole;
    let error_estimate = (refined - panel.whole).abs() / 15.0;

    if error_estimate <= tolerance {
        return Ok(refined + (refined - panel.whole) / 15.0);
    }
    if depth == 0 {
        return Err(NumericError::MaxDepthExceeded { max_depth: 0 });
    }

    Ok(adaptive_simpson(f, left, tolerance * 0.5, depth - 1)?
        + adaptive_simpson(f, right, tolerance * 0.5, depth - 1)?)
}

fn midpoint(start: f64, end: f64) -> f64 {
    start + (end - start) * 0.5
}

fn simpson_area(start: f64, end: f64, f_start: f64, f_mid: f64, f_end: f64) -> f64 {
    (end - start) * (f_start + 4.0 * f_mid + f_end) / 6.0
}

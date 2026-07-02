use rmc_numeric::{integrate_simpson, integrate_trapezoid, AdaptiveSimpson, NumericError};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

#[test]
fn trapezoid_integrates_affine_functions_exactly() {
    let integral = integrate_trapezoid(|x| 2.0 + 3.0 * x, -2.0, 4.0, 7).unwrap();
    let expected = 30.0;

    assert_close(integral, expected, 1.0e-12);
}

#[test]
fn simpson_integrates_cubic_polynomials_exactly() {
    let cubic = |x: f64| 1.0 - 2.0 * x + 3.0 * x * x - 0.5 * x * x * x;
    let antiderivative = |x: f64| x - x * x + x * x * x - 0.125 * x * x * x * x;

    let integral = integrate_simpson(cubic, -1.5, 2.0, 12).unwrap();
    let expected = antiderivative(2.0) - antiderivative(-1.5);

    assert_close(integral, expected, 1.0e-12);
}

#[test]
fn quadrature_preserves_signed_decreasing_intervals() {
    let increasing = integrate_simpson(|x| x * x, 0.0, 3.0, 20).unwrap();
    let decreasing = integrate_simpson(|x| x * x, 3.0, 0.0, 20).unwrap();

    assert_close(increasing, 9.0, 1.0e-12);
    assert_close(decreasing, -9.0, 1.0e-12);
}

#[test]
fn quadrature_rejects_invalid_panel_counts_and_nonfinite_values() {
    assert_eq!(
        integrate_trapezoid(|x| x, 0.0, 1.0, 0).unwrap_err(),
        NumericError::InvalidPanelCount {
            rule: "trapezoid",
            required: "at least one panel",
            actual: 0
        }
    );
    assert_eq!(
        integrate_simpson(|x| x, 0.0, 1.0, 3).unwrap_err(),
        NumericError::InvalidPanelCount {
            rule: "simpson",
            required: "a positive even panel count",
            actual: 3
        }
    );
    assert_eq!(
        integrate_simpson(|x| x, 0.0, f64::INFINITY, 2).unwrap_err(),
        NumericError::NonFiniteInterval {
            start: 0.0,
            end: f64::INFINITY
        }
    );

    let err =
        integrate_trapezoid(|x| if x == 0.5 { f64::NAN } else { x }, 0.0, 1.0, 2).unwrap_err();
    assert!(matches!(
        err,
        NumericError::NonFiniteFunctionValue { x: 0.5, value } if value.is_nan()
    ));
}

#[test]
fn adaptive_simpson_integrates_smooth_functions_to_requested_tolerance() {
    let quadrature = AdaptiveSimpson::new(1.0e-12, 1.0e-12, 32).unwrap();

    let sine = quadrature
        .integrate(|x| x.sin(), 0.0, std::f64::consts::PI)
        .unwrap();
    let gaussian_like = quadrature.integrate(|x| (-x * x).exp(), -1.0, 1.0).unwrap();

    assert_close(sine, 2.0, 1.0e-11);
    assert_close(gaussian_like, 1.493648265624854, 1.0e-11);
    assert_eq!(quadrature.integrate(|x| x.cos(), 1.25, 1.25).unwrap(), 0.0);
}

#[test]
fn adaptive_simpson_reports_tolerance_and_depth_errors() {
    let err = AdaptiveSimpson::new(f64::NAN, 0.0, 8).unwrap_err();
    assert!(matches!(
        err,
        NumericError::InvalidTolerance { abs, rel: 0.0 } if abs.is_nan()
    ));
    assert_eq!(
        AdaptiveSimpson::new(0.0, -1.0, 8).unwrap_err(),
        NumericError::InvalidTolerance {
            abs: 0.0,
            rel: -1.0
        }
    );

    let err = AdaptiveSimpson::new(1.0e-15, 0.0, 0)
        .unwrap()
        .integrate(|x| x.sin(), 0.0, std::f64::consts::PI)
        .unwrap_err();
    assert_eq!(err, NumericError::MaxDepthExceeded { max_depth: 0 });
}

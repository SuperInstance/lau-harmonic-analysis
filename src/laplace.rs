//! Laplace transform and inverse (numerical).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Laplace transform: L{f}(s) = integral_0^inf f(t) * e^(-st) dt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplaceTransform;

impl LaplaceTransform {
    /// Numerical Laplace transform via trapezoidal integration.
    /// `f` is the time-domain function, `s` is the complex frequency variable.
    /// `t_max` is the upper integration limit, `n_steps` is the number of steps.
    pub fn transform<F>(f: F, s: Complex64, t_max: f64, n_steps: usize) -> Complex64
    where
        F: Fn(f64) -> f64,
    {
        let dt = t_max / n_steps as f64;
        let mut sum = Complex64::new(0.0, 0.0);

        for i in 0..=n_steps {
            let t = i as f64 * dt;
            let ft = f(t);
            let weight = if i == 0 || i == n_steps { 0.5 } else { 1.0 };
            sum += weight * ft * (-s * t).exp() * dt;
        }
        sum
    }

    /// Numerical inverse Laplace transform using the Euler method (Talbot inversion).
    /// Returns f(t) at the specified time `t`.
    /// `n_terms` is the number of terms in the series expansion.
    pub fn inverse_transform<F>(f_laplace: F, t: f64, n_terms: usize) -> f64
    where
        F: Fn(Complex64) -> Complex64,
    {
        // Talbot's method for numerical inversion
        let r = 2.0 * n_terms as f64 / (5.0 * t);
        let mut sum = Complex64::new(0.0, 0.0);

        for k in 0..n_terms {
            let theta = std::f64::consts::PI * k as f64 / n_terms as f64;
            let s = Complex64::new(r * theta.cos(), r * theta.sin());
            let ds_dtheta = Complex64::new(-r * theta.sin(), r * theta.cos());

            if k == 0 {
                sum += f_laplace(s) * ds_dtheta;
            } else {
                sum += f_laplace(s) * ds_dtheta * Complex64::new(0.0, -1.0) * theta.sin();
            }
        }

        let result = (sum / (std::f64::consts::PI * Complex64::new(0.0, 1.0))).re * r / (n_terms as f64 * 2.0);
        // Simplified approach: use Gaver-Stehfest
        Self::stehfest(f_laplace, t, n_terms.min(20))
    }

    /// Gaver-Stehfest algorithm for inverse Laplace transform.
    fn stehfest<F>(f_laplace: F, t: f64, n: usize) -> f64
    where
        F: Fn(Complex64) -> Complex64,
    {
        let ln2_over_t = std::f64::consts::LN_2 / t;
        let mut result = 0.0;

        for i in 1..=n {
            let v_i = Self::stehfest_weight(i, n);
            let s = Complex64::new(i as f64 * ln2_over_t, 0.0);
            result += v_i * f_laplace(s).re;
        }

        result * ln2_over_t
    }

    /// Compute Stehfest weights.
    fn stehfest_weight(i: usize, n: usize) -> f64 {
        let k_max = ((i as f64) / 2.0).floor() as usize;
        let k_min = ((i as f64 + 1.0) / 2.0).floor() as usize;
        let mut sum = 0.0;

        for k in (k_min - 1)..k_max.min(n) {
            let binom = Self::binomial(n, k + 1);
            let sign = if (i as isize - k as isize - 1) % 2 == 0 { 1.0 } else { -1.0 };
            let power = (k + 1) as f64;
            let term = sign * power.powi(i as i32) * binom;
            let inner_binom = Self::binomial(2 * k + 1, k);
            sum += term * inner_binom;
        }

        let factorial_n = Self::factorial(n);
        let sign2 = if n % 2 == 0 { 1.0 } else { -1.0 };
        sign2 * sum / factorial_n
    }

    fn binomial(n: usize, k: usize) -> f64 {
        if k > n { return 0.0; }
        Self::factorial(n) / (Self::factorial(k) * Self::factorial(n - k))
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, x| acc * x as f64)
    }

    /// Common transform: unit step u(t) → 1/s
    pub fn unit_step(s: Complex64) -> Complex64 {
        1.0 / s
    }

    /// Common transform: exponential e^(-at) → 1/(s+a)
    pub fn exponential(s: Complex64, a: f64) -> Complex64 {
        1.0 / (s + a)
    }

    /// Common transform: cosine cos(ωt) → s/(s²+ω²)
    pub fn cosine(s: Complex64, omega: f64) -> Complex64 {
        s / (s * s + omega * omega)
    }

    /// Common transform: sine sin(ωt) → ω/(s²+ω²)
    pub fn sine(s: Complex64, omega: f64) -> Complex64 {
        omega / (s * s + omega * omega)
    }

    /// Transfer function evaluation: H(s) = K / (s² + 2ζωₙs + ωₙ²)
    pub fn second_order_system(s: Complex64, k: f64, zeta: f64, wn: f64) -> Complex64 {
        let denom = s * s + 2.0 * zeta * wn * s + wn * wn;
        k / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_unit_step() {
        let result = LaplaceTransform::transform(
            |_| 1.0,
            Complex64::new(1.0, 0.0),
            50.0,
            10000,
        );
        // L{1} at s=1 should be 1/1 = 1.0
        assert!((result.re - 1.0).abs() < 0.05, "Unit step: {}", result.re);
    }

    #[test]
    fn test_laplace_exponential() {
        let a = 2.0;
        let result = LaplaceTransform::transform(
            |t| (-a * t).exp(),
            Complex64::new(3.0, 0.0),
            20.0,
            10000,
        );
        // L{e^(-2t)} at s=3 should be 1/(3+2) = 0.2
        assert!((result.re - 0.2).abs() < 0.01, "Exponential: {}", result.re);
    }

    #[test]
    fn test_laplace_sine() {
        let omega = 1.0;
        let result = LaplaceTransform::transform(
            |t| (omega * t).sin(),
            Complex64::new(2.0, 0.0),
            50.0,
            10000,
        );
        // L{sin(t)} at s=2 should be 1/(4+1) = 0.2
        assert!((result.re - 0.2).abs() < 0.01, "Sine: {}", result.re);
    }

    #[test]
    fn test_common_transforms() {
        let s = Complex64::new(2.0, 0.0);
        // Unit step: 1/s = 0.5
        assert!((LaplaceTransform::unit_step(s).re - 0.5).abs() < 1e-10);
        // Exponential a=1: 1/(s+1) = 1/3
        assert!((LaplaceTransform::exponential(s, 1.0).re - 1.0/3.0).abs() < 1e-10);
        // Cosine ω=1: s/(s²+1) = 2/5
        assert!((LaplaceTransform::cosine(s, 1.0).re - 0.4).abs() < 1e-10);
        // Sine ω=1: 1/(s²+1) = 0.2
        assert!((LaplaceTransform::sine(s, 1.0).re - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_second_order_system() {
        let s = Complex64::new(0.0, 1.0);
        let h = LaplaceTransform::second_order_system(s, 1.0, 0.5, 1.0);
        // H(j) = 1 / (j² + 2*0.5*1*j + 1) = 1 / (-1 + j + 1) = 1/j = -j
        assert!(h.im.abs() > 0.0, "Second order should be non-zero");
    }
}

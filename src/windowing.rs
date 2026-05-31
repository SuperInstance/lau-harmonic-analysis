//! Windowing functions: Hamming, Hann, Blackman, Gaussian.

use serde::{Deserialize, Serialize};

/// Windowing functions for spectral analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Windowing;

impl Windowing {
    /// Apply a window function to a signal.
    pub fn apply(signal: &[f64], window: &[f64]) -> Vec<f64> {
        signal.iter().zip(window.iter()).map(|(s, w)| s * w).collect()
    }

    /// Hamming window: w(n) = 0.54 - 0.46 * cos(2πn/(N-1))
    pub fn hamming(n: usize) -> Vec<f64> {
        Self::symmetric_window(n, |alpha, beta, _, n_norm| {
            alpha - beta * (2.0 * std::f64::consts::PI * n_norm).cos()
        }, 0.54, 0.46)
    }

    /// Hann (Hanning) window: w(n) = 0.5 * (1 - cos(2πn/(N-1)))
    pub fn hann(n: usize) -> Vec<f64> {
        (0..n).map(|i| {
            0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        }).collect()
    }

    /// Blackman window: w(n) = a0 - a1*cos(2πn/(N-1)) + a2*cos(4πn/(N-1))
    pub fn blackman(n: usize) -> Vec<f64> {
        let a0 = 0.42;
        let a1 = 0.5;
        let a2 = 0.08;
        (0..n).map(|i| {
            let norm = i as f64 / (n - 1) as f64;
            a0 - a1 * (2.0 * std::f64::consts::PI * norm).cos()
                 + a2 * (4.0 * std::f64::consts::PI * norm).cos()
        }).collect()
    }

    /// Gaussian window: w(n) = exp(-0.5 * ((n - (N-1)/2) / (σ*(N-1)/2))^2)
    pub fn gaussian(n: usize, sigma: f64) -> Vec<f64> {
        let center = (n - 1) as f64 / 2.0;
        let denom = sigma * center;
        (0..n).map(|i| {
            let x = (i as f64 - center) / denom;
            (-0.5 * x * x).exp()
        }).collect()
    }

    /// Rectangular (no) window.
    pub fn rectangular(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    /// Compute the coherent gain of a window (mean value).
    pub fn coherent_gain(window: &[f64]) -> f64 {
        window.iter().sum::<f64>() / window.len() as f64
    }

    /// Compute the equivalent noise bandwidth (ENBW) of a window.
    /// ENBW = N * sum(w^2) / (sum(w))^2
    pub fn enbw(window: &[f64]) -> f64 {
        let n = window.len() as f64;
        let sum_w: f64 = window.iter().sum();
        let sum_w2: f64 = window.iter().map(|w| w * w).sum();
        n * sum_w2 / (sum_w * sum_w)
    }

    /// Compute the scalloping loss of a window (worst-case processing loss).
    pub fn scalloping_loss(window: &[f64]) -> f64 {
        let sum_w: f64 = window.iter().sum();
        let n = window.len();
        // Evaluate DTFT at half-bin offset
        let half_bin: f64 = window.iter().enumerate().map(|(i, &w)| {
            w * (std::f64::consts::PI * (i as f64 - (n - 1) as f64 / 2.0) / n as f64).cos()
        }).sum();
        (half_bin / sum_w).abs()
    }

    fn symmetric_window<F: Fn(f64, f64, f64, f64) -> f64>(
        n: usize, f: F, alpha: f64, beta: f64,
    ) -> Vec<f64> {
        (0..n).map(|i| {
            let n_norm = i as f64 / (n - 1) as f64;
            f(alpha, beta, n as f64, n_norm)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_symmetry() {
        let w = Windowing::hamming(32);
        for i in 0..16 {
            assert!((w[i] - w[31 - i]).abs() < 1e-12, "Hamming not symmetric at {i}");
        }
    }

    #[test]
    fn test_hamming_endpoints() {
        let w = Windowing::hamming(64);
        assert!((w[0] - 0.08).abs() < 0.01, "Hamming start: {}", w[0]);
        assert!((w[63] - 0.08).abs() < 0.01, "Hamming end: {}", w[63]);
    }

    #[test]
    fn test_hann_symmetry() {
        let w = Windowing::hann(32);
        for i in 0..16 {
            assert!((w[i] - w[31 - i]).abs() < 1e-12, "Hann not symmetric at {i}");
        }
    }

    #[test]
    fn test_hann_endpoints() {
        let w = Windowing::hann(64);
        assert!(w[0].abs() < 0.01, "Hann start should be ~0: {}", w[0]);
        assert!(w[63].abs() < 0.01, "Hann end should be ~0: {}", w[63]);
    }

    #[test]
    fn test_hann_peak() {
        let w = Windowing::hann(33);
        let mid = 16;
        assert!((w[mid] - 1.0).abs() < 1e-10, "Hann peak should be 1.0: {}", w[mid]);
    }

    #[test]
    fn test_blackman_symmetry() {
        let w = Windowing::blackman(32);
        for i in 0..16 {
            assert!((w[i] - w[31 - i]).abs() < 1e-12, "Blackman not symmetric at {i}");
        }
    }

    #[test]
    fn test_blackman_endpoints() {
        let w = Windowing::blackman(64);
        // Blackman endpoints should be near 0
        assert!(w[0].abs() < 0.01, "Blackman start: {}", w[0]);
        assert!(w[63].abs() < 0.01, "Blackman end: {}", w[63]);
    }

    #[test]
    fn test_gaussian_symmetry() {
        let w = Windowing::gaussian(32, 0.5);
        for i in 0..16 {
            assert!((w[i] - w[31 - i]).abs() < 1e-12, "Gaussian not symmetric at {i}");
        }
    }

    #[test]
    fn test_gaussian_peak() {
        let w = Windowing::gaussian(33, 0.5);
        assert!((w[16] - 1.0).abs() < 1e-10, "Gaussian peak should be 1.0: {}", w[16]);
    }

    #[test]
    fn test_coherent_gain() {
        let rect = Windowing::rectangular(100);
        assert!((Windowing::coherent_gain(&rect) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_enbw_rectangular() {
        let rect = Windowing::rectangular(100);
        let enbw = Windowing::enbw(&rect);
        assert!((enbw - 1.0).abs() < 1e-10, "Rectangular ENBW should be 1.0: {enbw}");
    }

    #[test]
    fn test_enbw_hann_greater_than_rect() {
        let rect = Windowing::rectangular(64);
        let hann = Windowing::hann(64);
        assert!(Windowing::enbw(&hann) > Windowing::enbw(&rect));
    }

    #[test]
    fn test_apply_window() {
        let signal = vec![2.0; 4];
        let window = vec![0.5; 4];
        let result = Windowing::apply(&signal, &window);
        for v in &result {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }
}

//! Z-transform for discrete systems.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Z-transform: X(z) = sum_{n=0}^{N-1} x[n] * z^{-n}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZTransform;

impl ZTransform {
    /// Compute the Z-transform of a discrete signal at a given complex point z.
    pub fn transform(signal: &[f64], z: Complex64) -> Complex64 {
        signal.iter().enumerate().map(|(n, &x)| {
            x * z.powi(-(n as i32))
        }).sum()
    }

    /// Evaluate the Z-transform on the unit circle (z = e^{jω}), equivalent to the DTFT.
    pub fn dtft(signal: &[f64], omega: f64) -> Complex64 {
        let z = Complex64::from_polar(1.0, omega);
        Self::transform(signal, z)
    }

    /// Compute the frequency response at multiple frequencies.
    pub fn frequency_response(signal: &[f64], frequencies: &[f64]) -> Vec<Complex64> {
        frequencies.iter().map(|&omega| Self::dtft(signal, omega)).collect()
    }

    /// Compute the magnitude response in dB.
    pub fn magnitude_response_db(signal: &[f64], frequencies: &[f64]) -> Vec<f64> {
        Self::frequency_response(signal, frequencies)
            .iter()
            .map(|h| 20.0 * h.norm().log10())
            .collect()
    }

    /// Compute the phase response in radians.
    pub fn phase_response(signal: &[f64], frequencies: &[f64]) -> Vec<f64> {
        Self::frequency_response(signal, frequencies)
            .iter()
            .map(|h| h.arg())
            .collect()
    }

    /// Check stability of an FIR filter (all zeros inside unit circle).
    /// For IIR, you'd need pole analysis.
    pub fn zeros(signal: &[f64]) -> Vec<Complex64> {
        let n = signal.len();
        if n <= 1 {
            return vec![];
        }
        // Find roots of the polynomial sum a_k * z^{-k} = 0
        // Equivalently, sum a_k * z^{N-1-k} = 0
        let mut coeffs: Vec<f64> = signal.iter().rev().cloned().collect();
        Self::find_roots(&coeffs)
    }

    /// Simple root finding via companion matrix eigenvalues (using nalgebra).
    fn find_roots(coeffs: &[f64]) -> Vec<Complex64> {
        let n = coeffs.len();
        if n <= 1 {
            return vec![];
        }

        // Normalize by leading coefficient
        let a0 = coeffs[0];
        if a0.abs() < 1e-30 {
            return vec![];
        }

        // Build companion matrix
        let size = n - 1;
        use nalgebra::DMatrix;
        let mut companion = DMatrix::<f64>::zeros(size, size);

        // Last column: -a_n/a_0, -a_{n-1}/a_0, ...
        for i in 0..size {
            companion[(i, size - 1)] = -coeffs[n - 1 - i] / a0;
        }
        // Sub-diagonal: 1s
        for i in 1..size {
            companion[(i, i - 1)] = 1.0;
        }

        let eigenvalues = companion.complex_eigenvalues();
        eigenvalues.iter().map(|c| Complex64::new(c.re, c.im)).collect()
    }

    /// Transfer function of a first-order system: H(z) = b0 + b1*z^{-1} / (1 + a1*z^{-1})
    pub fn first_order_transfer(z: Complex64, b0: f64, b1: f64, a1: f64) -> Complex64 {
        let num = b0 + b1 * z.powi(-1);
        let den = 1.0 + a1 * z.powi(-1);
        num / den
    }

    /// Compute group delay at a given frequency.
    pub fn group_delay(signal: &[f64], omega: f64, d_omega: f64) -> f64 {
        let h1 = Self::dtft(signal, omega - d_omega / 2.0);
        let h2 = Self::dtft(signal, omega + d_omega / 2.0);
        let phase_diff = h2.arg() - h1.arg();
        -phase_diff / d_omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ztransform_delta() {
        // δ[n]: X(z) = 1
        let signal = vec![1.0];
        let z = Complex64::new(2.0, 0.0);
        let result = ZTransform::transform(&signal, z);
        assert!((result.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ztransform_unit_step() {
        // u[n] for N=4: X(z) = 1 + z^{-1} + z^{-2} + z^{-3}
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        let z = Complex64::new(2.0, 0.0);
        let result = ZTransform::transform(&signal, z);
        let expected = 1.0 + 0.5 + 0.25 + 0.125;
        assert!((result.re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_ztransform_geometric() {
        // a^n for N=3: X(z) = a + a²/z + a³/z²
        let a = 0.5;
        let signal = vec![a, a * a, a * a * a];
        let z = Complex64::new(1.0, 0.0);
        let result = ZTransform::transform(&signal, z);
        let expected = a + a * a + a * a * a;
        assert!((result.re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_dtft_dc() {
        let signal = vec![1.0, 1.0, 1.0, 1.0];
        let result = ZTransform::dtft(&signal, 0.0);
        assert!((result.re - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dtft_nyquist() {
        let signal = vec![1.0, -1.0, 1.0, -1.0];
        let result = ZTransform::dtft(&signal, std::f64::consts::PI);
        assert!((result.re - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_response_db() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let freqs = vec![0.0, std::f64::consts::PI / 4.0];
        let db = ZTransform::magnitude_response_db(&signal, &freqs);
        // Delta has flat magnitude response (0 dB)
        assert!((db[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_response() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let freqs = vec![0.0];
        let phase = ZTransform::phase_response(&signal, &freqs);
        assert!(phase[0].abs() < 1e-10);
    }

    #[test]
    fn test_first_order_transfer() {
        let z = Complex64::new(1.0, 0.0);
        let h = ZTransform::first_order_transfer(z, 1.0, 0.5, -0.3);
        let expected = (1.0 + 0.5) / (1.0 - 0.3);
        assert!((h.re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_group_delay() {
        let signal = vec![1.0, 0.0, 0.0, 0.0];
        let gd = ZTransform::group_delay(&signal, 0.5, 0.01);
        // Delta has 0 group delay everywhere
        assert!(gd.abs() < 0.5, "Group delay of delta should be ~0: {gd}");
    }
}

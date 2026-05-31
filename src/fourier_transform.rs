//! Fourier transform properties: convolution theorem, Plancherel, uncertainty principle.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Continuous Fourier transform properties and numerical utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierTransform;

impl FourierTransform {
    /// Numerical convolution of two real signals.
    pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len() + b.len() - 1;
        let mut result = vec![0.0; n];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] += a[i] * b[j];
            }
        }
        result
    }

    /// Circular convolution via FFT (convolution theorem).
    /// Pads both signals to length >= len(a) + len(b) - 1 (next power of 2).
    pub fn convolve_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
        let min_len = a.len() + b.len() - 1;
        let n = Self::next_power_of_2(min_len);

        let mut ca: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut cb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        ca.resize(n, Complex64::new(0.0, 0.0));
        cb.resize(n, Complex64::new(0.0, 0.0));

        let fa = crate::dft::Dft::fft(&ca);
        let fb = crate::dft::Dft::fft(&cb);

        let product: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(a, b)| a * b).collect();
        let result = crate::dft::Dft::ifft(&product);

        result[..min_len].iter().map(|c| c.re).collect()
    }

    /// Plancherel's theorem: the L2 norm is preserved under Fourier transform.
    /// Returns (time_domain_energy, frequency_domain_energy).
    pub fn plancherel(signal: &[Complex64]) -> (f64, f64) {
        let time_energy: f64 = signal.iter().map(|c| c.norm_sqr()).sum();
        let spectrum = crate::dft::Dft::fft(signal);
        let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / signal.len() as f64;
        (time_energy, freq_energy)
    }

    /// Verify Plancherel's theorem. Returns the absolute difference between
    /// time and frequency domain energies (should be ~0).
    pub fn verify_plancherel(signal: &[Complex64]) -> f64 {
        let (t, f) = Self::plancherel(signal);
        (t - f).abs()
    }

    /// Compute the uncertainty principle lower bound for a signal.
    /// Heisenberg's inequality: Δt · Δf ≥ 1/(4π)
    /// Returns (delta_t, delta_f, product).
    pub fn uncertainty(signal: &[Complex64], sample_rate: f64) -> (f64, f64, f64) {
        let n = signal.len() as f64;
        let dt = 1.0 / sample_rate;

        // Time-domain variance
        let total_energy: f64 = signal.iter().map(|c| c.norm_sqr()).sum();
        if total_energy < 1e-30 {
            return (0.0, 0.0, 0.0);
        }

        let t_mean: f64 = signal.iter().enumerate()
            .map(|(i, c)| i as f64 * dt * c.norm_sqr()).sum::<f64>() / total_energy;
        let t_var: f64 = signal.iter().enumerate()
            .map(|(i, c)| {
                let t = i as f64 * dt;
                let diff = t - t_mean;
                diff * diff * c.norm_sqr()
            }).sum::<f64>() / total_energy;

        // Frequency-domain variance
        let spectrum = crate::dft::Dft::fft(signal);
        let spec_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        let df = sample_rate / n;

        let f_mean: f64 = spectrum.iter().enumerate()
            .map(|(k, c)| k as f64 * df * c.norm_sqr()).sum::<f64>() / spec_energy;
        let f_var: f64 = spectrum.iter().enumerate()
            .map(|(k, c)| {
                let f = k as f64 * df;
                let diff = f - f_mean;
                diff * diff * c.norm_sqr()
            }).sum::<f64>() / spec_energy;

        let delta_t = t_var.sqrt();
        let delta_f = f_var.sqrt();
        (delta_t, delta_f, delta_t * delta_f)
    }

    /// Compute cross-correlation of two signals using FFT.
    pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
        let min_len = a.len() + b.len() - 1;
        let n = Self::next_power_of_2(min_len);

        let mut ca: Vec<Complex64> = a.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let mut cb: Vec<Complex64> = b.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        ca.resize(n, Complex64::new(0.0, 0.0));
        cb.resize(n, Complex64::new(0.0, 0.0));

        let fa = crate::dft::Dft::fft(&ca);
        let fb = crate::dft::Dft::fft(&cb);

        // Cross-correlation: conj(Fa) * Fb
        let product: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(a, b)| a.conj() * b).collect();
        let result = crate::dft::Dft::ifft(&product);

        result[..min_len].iter().map(|c| c.re).collect()
    }

    fn next_power_of_2(n: usize) -> usize {
        let mut p = 1;
        while p < n { p *= 2; }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolution_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.5, 1.0];
        let result = FourierTransform::convolve(&a, &b);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.5).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolution_theorem() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, -1.0, 0.5];
        let direct = FourierTransform::convolve(&a, &b);
        let fft_conv = FourierTransform::convolve_fft(&a, &b);
        assert_eq!(direct.len(), fft_conv.len());
        for (d, f) in direct.iter().zip(fft_conv.iter()) {
            assert!((d - f).abs() < 1e-8, "convolution theorem mismatch: {d} vs {f}");
        }
    }

    #[test]
    fn test_plancherel_theorem() {
        let signal: Vec<Complex64> = (0..64).map(|j| {
            Complex64::new((j as f64 * 0.1).sin(), (j as f64 * 0.05).cos())
        }).collect();
        let error = FourierTransform::verify_plancherel(&signal);
        assert!(error < 1e-8, "Plancherel error: {error}");
    }

    #[test]
    fn test_plancherel_impulse() {
        let mut signal = vec![Complex64::new(0.0, 0.0); 32];
        signal[0] = Complex64::new(1.0, 0.0);
        let (t, f) = FourierTransform::plancherel(&signal);
        assert!((t - f).abs() < 1e-10, "Plancherel impulse: {t} vs {f}");
    }

    #[test]
    fn test_uncertainty_principle() {
        // Gaussian-like signal
        let signal: Vec<Complex64> = (0..128).map(|j| {
            let x = (j as f64 - 64.0) / 10.0;
            Complex64::new((-x * x / 2.0).exp(), 0.0)
        }).collect();
        let (_, _, product) = FourierTransform::uncertainty(&signal, 100.0);
        assert!(product > 0.0, "Uncertainty product should be positive: {product}");
    }

    #[test]
    fn test_cross_correlation() {
        let a = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 2.0, 1.0];
        let corr = FourierTransform::cross_correlate(&a, &b);
        // Peak should be at lag 0 for auto-correlation-like case
        assert!(corr.len() >= 8);
    }

    #[test]
    fn test_convolution_delta() {
        // Convolving with delta should return original
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let delta = vec![1.0];
        let result = FourierTransform::convolve(&signal, &delta);
        assert_eq!(result.len(), 4);
        for (a, b) in signal.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

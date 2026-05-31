//! Discrete Fourier Transform: naive DFT, Cooley-Tukey FFT, and inverse.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// DFT/FFT operations on complex-valued signals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dft;

impl Dft {
    /// Naive O(n^2) DFT.
    pub fn dft(signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        let mut result = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for (j, &x) in signal.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                sum += x * Complex64::from_polar(1.0, angle);
            }
            result.push(sum);
        }
        result
    }

    /// Inverse DFT (naive).
    pub fn idft(spectrum: &[Complex64]) -> Vec<Complex64> {
        let n = spectrum.len();
        let conj: Vec<Complex64> = spectrum.iter().map(|c| c.conj()).collect();
        let result = Self::dft(&conj);
        result.iter().map(|c| c.conj() / n as f64).collect()
    }

    /// Cooley-Tukey radix-2 FFT. Input length must be a power of 2.
    /// Returns the DFT of the input.
    pub fn fft(signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        if n == 1 {
            return signal.to_vec();
        }
        if n == 2 {
            return vec![
                signal[0] + signal[1],
                signal[0] - signal[1],
            ];
        }
        // Bit-reversal permutation + iterative butterfly
        let mut x = Self::bit_reverse_copy(signal);
        let mut m = 1usize;
        while m < n {
            let wm = Complex64::from_polar(1.0, -std::f64::consts::PI / m as f64);
            for k in (0..n).step_by(2 * m) {
                let mut w = Complex64::new(1.0, 0.0);
                for j in 0..m {
                    let t = w * x[k + j + m];
                    let u = x[k + j];
                    x[k + j] = u + t;
                    x[k + j + m] = u - t;
                    w *= wm;
                }
            }
            m *= 2;
        }
        x
    }

    /// Inverse FFT using the conjugate trick: IFFT(X) = conj(FFT(conj(X))) / N.
    pub fn ifft(spectrum: &[Complex64]) -> Vec<Complex64> {
        let n = spectrum.len();
        let conj: Vec<Complex64> = spectrum.iter().map(|c| c.conj()).collect();
        let result = Self::fft(&conj);
        result.iter().map(|c| c.conj() / n as f64).collect()
    }

    /// Bit-reversal permutation.
    fn bit_reverse_copy(signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        let bits = (n as f64).log2() as usize;
        let mut result = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..n {
            let j = Self::reverse_bits(i, bits);
            result[j] = signal[i];
        }
        result
    }

    /// Reverse the lower `bits` bits of `n`.
    fn reverse_bits(mut n: usize, bits: usize) -> usize {
        let mut result = 0usize;
        for _ in 0..bits {
            result = (result << 1) | (n & 1);
            n >>= 1;
        }
        result
    }

    /// Compute the power spectrum (magnitude squared) of the FFT.
    pub fn power_spectrum(signal: &[Complex64]) -> Vec<f64> {
        let spectrum = Self::fft(signal);
        spectrum.iter().map(|c| c.norm_sqr()).collect()
    }

    /// Compute the frequency bins for a signal of length `n` with sample rate `fs`.
    pub fn frequency_bins(n: usize, fs: f64) -> Vec<f64> {
        (0..n).map(|k| k as f64 * fs / n as f64).collect()
    }

    /// Check if n is a power of 2.
    pub fn is_power_of_2(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Pad signal to next power of 2 with zeros.
    pub fn zero_pad_to_power_of_2(signal: &[Complex64]) -> Vec<Complex64> {
        let n = signal.len();
        if Self::is_power_of_2(n) {
            return signal.to_vec();
        }
        let next_pow2 = 1 << ((n as f64).log2().ceil() as usize);
        let mut padded = signal.to_vec();
        padded.resize(next_pow2, Complex64::new(0.0, 0.0));
        padded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex_approx_eq(a: Complex64, b: Complex64, tol: f64) -> bool {
        (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol
    }

    #[test]
    fn test_dft_dc_signal() {
        let signal = vec![Complex64::new(3.0, 0.0); 8];
        let result = Dft::dft(&signal);
        assert!(complex_approx_eq(result[0], Complex64::new(24.0, 0.0), 1e-10));
        for k in 1..8 {
            assert!(result[k].norm() < 1e-10, "Non-DC bin {k} should be ~0");
        }
    }

    #[test]
    fn test_dft_single_tone() {
        let n = 64;
        let signal: Vec<Complex64> = (0..n).map(|j| {
            let t = j as f64 / n as f64;
            Complex64::new((2.0 * std::f64::consts::PI * 4.0 * t).cos(), 0.0)
        }).collect();
        let result = Dft::dft(&signal);
        // Bin 4 and bin 60 (= 64-4) should have large magnitude
        assert!(result[4].norm() > 20.0, "Bin 4 magnitude: {}", result[4].norm());
        assert!(result[60].norm() > 20.0, "Bin 60 magnitude: {}", result[60].norm());
    }

    #[test]
    fn test_idft_roundtrip() {
        let signal: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.0, -1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 1.0),
            Complex64::new(0.5, -0.5),
        ];
        let spectrum = Dft::dft(&signal);
        let recovered = Dft::idft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10), "Roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_fft_matches_dft() {
        let signal: Vec<Complex64> = (0..16).map(|j| {
            Complex64::new((j as f64 * 0.5).sin(), (j as f64 * 0.3).cos())
        }).collect();
        let dft_result = Dft::dft(&signal);
        let fft_result = Dft::fft(&signal);
        for (a, b) in dft_result.iter().zip(fft_result.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10), "FFT != DFT: {a} vs {b}");
        }
    }

    #[test]
    fn test_ifft_roundtrip() {
        let signal: Vec<Complex64> = (0..32).map(|j| {
            Complex64::new((j as f64).cos(), (j as f64 * 0.7).sin())
        }).collect();
        let spectrum = Dft::fft(&signal);
        let recovered = Dft::ifft(&spectrum);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!(complex_approx_eq(*a, *b, 1e-10), "IFFT roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_fft_dc() {
        let signal = vec![Complex64::new(5.0, 0.0); 8];
        let result = Dft::fft(&signal);
        assert!(complex_approx_eq(result[0], Complex64::new(40.0, 0.0), 1e-10));
        for k in 1..8 {
            assert!(result[k].norm() < 1e-10);
        }
    }

    #[test]
    fn test_power_spectrum() {
        let signal = vec![Complex64::new(1.0, 0.0); 4];
        let ps = Dft::power_spectrum(&signal);
        assert!((ps[0] - 16.0).abs() < 1e-10);
        assert!(ps[1] < 1e-10);
    }

    #[test]
    fn test_frequency_bins() {
        let bins = Dft::frequency_bins(8, 1000.0);
        assert_eq!(bins.len(), 8);
        assert!((bins[0] - 0.0).abs() < 1e-10);
        assert!((bins[1] - 125.0).abs() < 1e-10);
        assert!((bins[4] - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(Dft::reverse_bits(0, 3), 0);
        assert_eq!(Dft::reverse_bits(1, 3), 4);
        assert_eq!(Dft::reverse_bits(3, 3), 6);
        assert_eq!(Dft::reverse_bits(6, 3), 3);
    }

    #[test]
    fn test_is_power_of_2() {
        assert!(Dft::is_power_of_2(1));
        assert!(Dft::is_power_of_2(256));
        assert!(!Dft::is_power_of_2(3));
        assert!(!Dft::is_power_of_2(0));
    }

    #[test]
    fn test_zero_pad() {
        let signal = vec![Complex64::new(1.0, 0.0); 5];
        let padded = Dft::zero_pad_to_power_of_2(&signal);
        assert_eq!(padded.len(), 8);
        assert!(Dft::is_power_of_2(padded.len()));
    }
}

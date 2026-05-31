//! Wavelet transform basics: Haar wavelets and multi-resolution analysis.

use serde::{Deserialize, Serialize};

/// Wavelet transform operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wavelet;

impl Wavelet {
    /// Haar wavelet mother function ψ(t):
    ///   1 for 0 ≤ t < 0.5
    ///  -1 for 0.5 ≤ t < 1
    ///   0 otherwise
    pub fn haar_wavelet(t: f64) -> f64 {
        if t >= 0.0 && t < 0.5 { 1.0 }
        else if t >= 0.5 && t < 1.0 { -1.0 }
        else { 0.0 }
    }

    /// Haar scaling function φ(t):
    ///   1 for 0 ≤ t < 1
    ///   0 otherwise
    pub fn haar_scaling(t: f64) -> f64 {
        if t >= 0.0 && t < 1.0 { 1.0 } else { 0.0 }
    }

    /// Discrete Haar wavelet transform.
    /// Signal length must be a power of 2.
    /// Returns (approximation_coeffs, detail_coeffs) at each level.
    pub fn haar_dwt(signal: &[f64]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut levels = Vec::new();
        let mut current = signal.to_vec();
        let n = current.len();

        if n < 2 || !Self::is_power_of_2(n) {
            return levels;
        }

        while current.len() >= 2 {
            let half = current.len() / 2;
            let mut approx = Vec::with_capacity(half);
            let mut detail = Vec::with_capacity(half);

            for i in 0..half {
                let a = (current[2 * i] + current[2 * i + 1]) / std::f64::consts::SQRT_2;
                let d = (current[2 * i] - current[2 * i + 1]) / std::f64::consts::SQRT_2;
                approx.push(a);
                detail.push(d);
            }

            levels.push((approx.clone(), detail));
            current = approx;
        }

        levels
    }

    /// Inverse discrete Haar wavelet transform.
    /// Takes the final approximation and all detail coefficients.
    pub fn haar_idwt(approx: &[f64], details: &[Vec<f64>]) -> Vec<f64> {
        let mut current = approx.to_vec();

        for detail in details.iter().rev() {
            let mut next = Vec::with_capacity(current.len() * 2);
            for i in 0..current.len() {
                let a = current[i];
                let d = detail[i];
                next.push((a + d) / std::f64::consts::SQRT_2);
                next.push((a - d) / std::f64::consts::SQRT_2);
            }
            current = next;
        }

        current
    }

    /// Multi-resolution analysis: decompose signal into levels and reconstruct
    /// each level contribution.
    pub fn multiresolution(signal: &[f64]) -> Vec<Vec<f64>> {
        let n = signal.len();
        let levels = Self::haar_dwt(signal);

        let mut contributions = Vec::new();
        let mut base_approx = levels.last().unwrap().0.clone();

        // Reconstruct level by level
        for level_idx in 0..levels.len() {
            let details_to_use: Vec<Vec<f64>> = levels[..=level_idx]
                .iter().map(|(_, d)| d.clone()).collect();
            let reconstructed = Self::haar_idwt(&base_approx, &details_to_use);
            // Trim or pad to original length
            let contrib: Vec<f64> = reconstructed.iter().take(n).cloned().collect();
            let mut padded = contrib;
            padded.resize(n, 0.0);
            contributions.push(padded);

            // For next level, we use the approximation from this level
            if level_idx + 1 < levels.len() {
                base_approx = levels[level_idx].0.clone();
            }
        }

        contributions
    }

    /// Compute wavelet energy at each decomposition level.
    pub fn wavelet_energy(signal: &[f64]) -> Vec<f64> {
        let levels = Self::haar_dwt(signal);
        levels.iter().map(|(_, detail)| {
            detail.iter().map(|d| d * d).sum()
        }).collect()
    }

    /// Denoise signal by thresholding wavelet coefficients.
    /// Uses soft thresholding: sign(x) * max(|x| - threshold, 0)
    pub fn denoise(signal: &[f64], threshold: f64) -> Vec<f64> {
        let levels = Self::haar_dwt(signal);
        if levels.is_empty() {
            return signal.to_vec();
        }

        let final_approx = levels.last().unwrap().0.clone();
        let details: Vec<Vec<f64>> = levels.iter().map(|(_, detail)| {
            detail.iter().map(|&d| {
                if d.abs() <= threshold { 0.0 }
                else { d.signum() * (d.abs() - threshold) }
            }).collect()
        }).collect();

        Self::haar_idwt(&final_approx, &details)
    }

    fn is_power_of_2(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haar_wavelet_values() {
        assert_eq!(Wavelet::haar_wavelet(0.25), 1.0);
        assert_eq!(Wavelet::haar_wavelet(0.75), -1.0);
        assert_eq!(Wavelet::haar_wavelet(-0.5), 0.0);
        assert_eq!(Wavelet::haar_wavelet(1.5), 0.0);
    }

    #[test]
    fn test_haar_scaling_values() {
        assert_eq!(Wavelet::haar_scaling(0.5), 1.0);
        assert_eq!(Wavelet::haar_scaling(1.5), 0.0);
    }

    #[test]
    fn test_haar_dwt_constant() {
        let signal = vec![4.0; 8];
        let levels = Wavelet::haar_dwt(&signal);
        assert_eq!(levels.len(), 3); // log2(8) = 3 levels
        // All detail coefficients should be ~0
        for (_, detail) in &levels {
            for &d in detail {
                assert!(d.abs() < 1e-10, "Detail should be 0: {d}");
            }
        }
    }

    #[test]
    fn test_haar_dwt_level_sizes() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let levels = Wavelet::haar_dwt(&signal);
        assert_eq!(levels[0].0.len(), 4); // Level 1: 4 approx, 4 detail
        assert_eq!(levels[0].1.len(), 4);
        assert_eq!(levels[1].0.len(), 2); // Level 2: 2 approx, 2 detail
        assert_eq!(levels[2].0.len(), 1); // Level 3: 1 approx, 1 detail
    }

    #[test]
    fn test_haar_idwt_roundtrip() {
        let signal = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let levels = Wavelet::haar_dwt(&signal);
        let final_approx = levels.last().unwrap().0.clone();
        let details: Vec<Vec<f64>> = levels.iter().map(|(_, d)| d.clone()).collect();
        let reconstructed = Wavelet::haar_idwt(&final_approx, &details);
        for (a, b) in signal.iter().zip(reconstructed.iter()) {
            assert!((a - b).abs() < 1e-10, "IDWT roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_haar_dwt_step_signal() {
        // Step: [0,0,0,0, 1,1,1,1] — detail at level 3 should capture this
        let signal = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let levels = Wavelet::haar_dwt(&signal);
        // Level 1 details should be all zero (constant in each pair)
        for &d in &levels[0].1 {
            assert!(d.abs() < 1e-10, "Level 1 detail should be 0: {d}");
        }
    }

    #[test]
    fn test_wavelet_energy_constant() {
        let signal = vec![5.0; 8];
        let energies = Wavelet::wavelet_energy(&signal);
        for &e in &energies {
            assert!(e < 1e-10, "Constant signal should have 0 detail energy: {e}");
        }
    }

    #[test]
    fn test_wavelet_energy_nonzero() {
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let energies = Wavelet::wavelet_energy(&signal);
        // High frequency signal should have energy
        let total: f64 = energies.iter().sum();
        assert!(total > 0.0, "Should have energy: {total}");
    }

    #[test]
    fn test_denoise_preserves_smooth() {
        let mut signal = vec![0.0; 16];
        for i in 0..16 {
            signal[i] = (i as f64 / 16.0 * std::f64::consts::PI).sin();
        }
        // Add small noise
        let mut noisy = signal.clone();
        noisy[3] += 10.0;
        noisy[10] -= 10.0;
        let denoised = Wavelet::denoise(&noisy, 1.0);
        // Denoised should be closer to original than noisy
        let noisy_error: f64 = signal.iter().zip(noisy.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        let denoised_error: f64 = signal.iter().zip(denoised.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
        assert!(denoised_error < noisy_error, "Denoised should be closer: {denoised_error} vs {noisy_error}");
    }
}

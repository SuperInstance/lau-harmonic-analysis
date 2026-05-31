//! Agent signal analysis: detect periodic patterns in agent behavior.

use serde::{Deserialize, Serialize};

use crate::dft::Dft;
use crate::spectral::SpectralEstimation;
use crate::wavelet::Wavelet;
use num_complex::Complex64;

/// Detected periodic pattern in agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicPattern {
    /// Dominant frequency (Hz or cycles per sample)
    pub frequency: f64,
    /// Period in samples
    pub period: usize,
    /// Relative strength (0-1)
    pub strength: f64,
    /// Phase offset in radians
    pub phase: f64,
}

/// Analysis result for agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorAnalysis {
    /// Detected periodic patterns, sorted by strength descending.
    pub patterns: Vec<PeriodicPattern>,
    /// Dominant period in samples (if any pattern detected).
    pub dominant_period: Option<usize>,
    /// Periodicity score (0 = random, 1 = perfectly periodic).
    pub periodicity_score: f64,
    /// Wavelet energy at each level.
    pub wavelet_energies: Vec<f64>,
    /// Spectral centroid.
    pub spectral_centroid: f64,
}

/// Signal analysis for detecting periodic patterns in agent behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalAnalysis;

impl SignalAnalysis {
    /// Analyze agent behavior signal for periodic patterns.
    /// `signal` is a time series of agent activity metrics.
    /// `sample_rate` is the sampling rate (e.g., 1.0 for per-second metrics).
    pub fn analyze(signal: &[f64], sample_rate: f64) -> BehaviorAnalysis {
        // 1. Spectral analysis
        let (freqs, psd) = SpectralEstimation::periodogram(signal, sample_rate);

        // Find peaks
        let max_psd = psd.iter().cloned().fold(0.0f64, f64::max);
        let min_peak = max_psd * 0.05; // 5% of max
        let peaks = SpectralEstimation::find_peaks(&freqs, &psd, min_peak);

        // Convert peaks to patterns
        let patterns: Vec<PeriodicPattern> = peaks.iter().map(|&(freq, power)| {
            let period = if freq > 0.0 { (sample_rate / freq).round() as usize } else { usize::MAX };
            let strength = if max_psd > 0.0 { power / max_psd } else { 0.0 };
            // Estimate phase from DFT
            let complex_signal: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
            let spectrum = Dft::fft(&complex_signal);
            let bin = (freq * signal.len() as f64 / sample_rate).round() as usize;
            let phase = if bin < spectrum.len() { spectrum[bin].arg() } else { 0.0 };

            PeriodicPattern { frequency: freq, period, strength, phase }
        }).collect();

        // 2. Dominant period
        let dominant_period = patterns.first().map(|p| p.period);

        // 3. Periodicity score: ratio of peak power to total power
        let total_power: f64 = psd.iter().sum();
        let peak_power: f64 = patterns.iter().map(|p| p.strength * max_psd).sum();
        let periodicity_score = if total_power > 0.0 {
            (peak_power / total_power).min(1.0)
        } else {
            0.0
        };

        // 4. Wavelet energies (need power-of-2 length)
        let n = signal.len();
        let next_pow2 = {
            let mut p = 1;
            while p < n { p *= 2; }
            p
        };
        let mut padded = signal.to_vec();
        padded.resize(next_pow2, 0.0);
        let wavelet_energies = Wavelet::wavelet_energy(&padded);

        // 5. Spectral centroid
        let spectral_centroid = SpectralEstimation::spectral_centroid(&freqs, &psd);

        BehaviorAnalysis {
            patterns,
            dominant_period,
            periodicity_score,
            wavelet_energies,
            spectral_centroid,
        }
    }

    /// Generate a synthetic agent behavior signal with known periodicity.
    pub fn synthetic_behavior(n: usize, period: usize, noise_level: f64) -> Vec<f64> {
        (0..n).map(|i| {
            let base = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            let noise = if noise_level > 0.0 {
                ((i * 7919 + 1234) as f64 * 0.0001 % 1.0 - 0.5) * 2.0 * noise_level
            } else {
                0.0
            };
            base + noise
        }).collect()
    }

    /// Compute autocorrelation to detect periodicity.
    pub fn autocorrelation(signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let mean = signal.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = signal.iter().map(|x| x - mean).collect();
        let var: f64 = centered.iter().map(|x| x * x).sum();

        if var < 1e-30 {
            return vec![1.0; n];
        }

        (0..n).map(|lag| {
            let sum: f64 = (0..n - lag).map(|i| centered[i] * centered[i + lag]).sum();
            sum / var
        }).collect()
    }

    /// Find the period from autocorrelation (first significant peak after lag 0).
    pub fn detect_period_autocorrelation(autocorr: &[f64], threshold: f64) -> Option<usize> {
        let n = autocorr.len();
        if n < 3 { return None; }

        // Skip first few samples, find first peak above threshold
        let min_lag = 2;
        for i in min_lag..n.saturating_sub(1) {
            if autocorr[i] > threshold
                && autocorr[i] > autocorr[i - 1]
                && autocorr[i] >= autocorr[i + 1]
            {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_sinusoidal_behavior() {
        let signal = SignalAnalysis::synthetic_behavior(512, 32, 0.0);
        let analysis = SignalAnalysis::analyze(&signal, 1.0);
        assert!(analysis.periodicity_score > 0.5, "Should detect periodicity: {}", analysis.periodicity_score);
        assert!(analysis.dominant_period.is_some());
        if let Some(period) = analysis.dominant_period {
            assert!((period as i64 - 32).abs() <= 2, "Period should be ~32: {period}");
        }
    }

    #[test]
    fn test_analyze_noisy_behavior() {
        let signal = SignalAnalysis::synthetic_behavior(512, 64, 0.3);
        let analysis = SignalAnalysis::analyze(&signal, 1.0);
        assert!(!analysis.patterns.is_empty() || analysis.periodicity_score > 0.0);
    }

    #[test]
    fn test_synthetic_behavior_values() {
        let signal = SignalAnalysis::synthetic_behavior(100, 10, 0.0);
        assert_eq!(signal.len(), 100);
        // Check periodicity
        for i in 0..90 {
            assert!((signal[i] - signal[i + 10]).abs() < 1e-10, "Should be periodic at period 10");
        }
    }

    #[test]
    fn test_autocorrelation_periodic() {
        let signal = SignalAnalysis::synthetic_behavior(256, 16, 0.0);
        let autocorr = SignalAnalysis::autocorrelation(&signal);
        assert!((autocorr[0] - 1.0).abs() < 1e-10, "ACF at lag 0 should be 1.0");
        // ACF at period should be high
        assert!(autocorr[16] > 0.5, "ACF at period should be high: {}", autocorr[16]);
    }

    #[test]
    fn test_detect_period() {
        let signal = SignalAnalysis::synthetic_behavior(256, 20, 0.0);
        let autocorr = SignalAnalysis::autocorrelation(&signal);
        let period = SignalAnalysis::detect_period_autocorrelation(&autocorr, 0.3);
        assert!(period.is_some(), "Should detect period");
        if let Some(p) = period {
            assert!((p as i64 - 20).abs() <= 2, "Detected period should be ~20: {p}");
        }
    }

    #[test]
    fn test_analyze_wavelet_energies() {
        let signal = SignalAnalysis::synthetic_behavior(256, 32, 0.0);
        let analysis = SignalAnalysis::analyze(&signal, 1.0);
        assert!(!analysis.wavelet_energies.is_empty());
        let total: f64 = analysis.wavelet_energies.iter().sum();
        assert!(total > 0.0, "Wavelet energies should be positive");
    }

    #[test]
    fn test_analysis_serializable() {
        let signal = SignalAnalysis::synthetic_behavior(64, 8, 0.1);
        let analysis = SignalAnalysis::analyze(&signal, 1.0);
        let json = serde_json::to_string(&analysis).expect("should serialize");
        assert!(json.contains("periodicity_score"));
        let recovered: BehaviorAnalysis = serde_json::from_str(&json).expect("should deserialize");
        assert!((recovered.periodicity_score - analysis.periodicity_score).abs() < 1e-10);
    }
}

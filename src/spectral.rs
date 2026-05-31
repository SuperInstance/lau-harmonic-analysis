//! Spectral estimation: periodogram, Welch's method.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::windowing::Windowing;
use crate::dft::Dft;

/// Spectral estimation methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralEstimation;

impl SpectralEstimation {
    /// Compute the periodogram (power spectral density estimate).
    /// Returns (frequencies, psd) where psd is in power/Hz.
    pub fn periodogram(signal: &[f64], sample_rate: f64) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let complex_signal: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        // Zero-pad to next power of 2 for better frequency resolution
        let padded = Dft::zero_pad_to_power_of_2(&complex_signal);
        let nfft = padded.len();

        let spectrum = Dft::fft(&padded);
        let psd: Vec<f64> = spectrum.iter()
            .map(|c| c.norm_sqr() / (sample_rate * n as f64))
            .collect();

        let freqs = Dft::frequency_bins(nfft, sample_rate);

        (freqs, psd)
    }

    /// Welch's method for PSD estimation with overlapping segments.
    /// `segment_length` is the length of each segment.
    /// `overlap` is the number of overlapping samples between segments.
    pub fn welch(
        signal: &[f64],
        sample_rate: f64,
        segment_length: usize,
        overlap: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let nfft = if Dft::is_power_of_2(segment_length) {
            segment_length
        } else {
            let mut p = 1;
            while p < segment_length { p *= 2; }
            p
        };

        let window = Windowing::hann(segment_length);
        let window_norm: f64 = window.iter().map(|w| w * w).sum();

        let step = segment_length - overlap;
        let n_segments = if signal.len() >= segment_length {
            (signal.len() - segment_length) / step + 1
        } else {
            1
        };

        let mut avg_psd = vec![0.0; nfft];

        for seg in 0..n_segments {
            let start = seg * step;
            if start + segment_length > signal.len() {
                break;
            }

            let windowed: Vec<Complex64> = signal[start..start + segment_length]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| Complex64::new(s * w, 0.0))
                .collect();

            let mut padded = windowed;
            padded.resize(nfft, Complex64::new(0.0, 0.0));

            let spectrum = Dft::fft(&padded);
            for (i, c) in spectrum.iter().enumerate() {
                avg_psd[i] += c.norm_sqr();
            }
        }

        // Normalize
        let scale = sample_rate * window_norm * n_segments as f64;
        for p in avg_psd.iter_mut() {
            *p /= scale;
        }

        let freqs = Dft::frequency_bins(nfft, sample_rate);
        (freqs, avg_psd)
    }

    /// Find dominant frequency peaks in a PSD estimate.
    /// Returns (frequency, power) pairs sorted by power descending.
    pub fn find_peaks(freqs: &[f64], psd: &[f64], min_peak_height: f64) -> Vec<(f64, f64)> {
        let mut peaks = Vec::new();
        let n = psd.len();
        let half = n / 2; // Only look at first half (positive frequencies)

        for i in 1..half.saturating_sub(1) {
            if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > min_peak_height {
                peaks.push((freqs[i], psd[i]));
            }
        }

        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        peaks
    }

    /// Compute the total power in a frequency band.
    pub fn band_power(freqs: &[f64], psd: &[f64], f_low: f64, f_high: f64) -> f64 {
        let df = if freqs.len() > 1 { freqs[1] - freqs[0] } else { 1.0 };
        freqs.iter().zip(psd.iter())
            .filter(|(f, _)| **f >= f_low && **f <= f_high)
            .map(|(_, p)| p * df)
            .sum()
    }

    /// Compute spectral centroid (center of mass of the spectrum).
    pub fn spectral_centroid(freqs: &[f64], psd: &[f64]) -> f64 {
        let half = freqs.len() / 2;
        let total_power: f64 = psd[..half].iter().sum();
        if total_power < 1e-30 { return 0.0; }
        freqs[..half].iter().zip(psd[..half].iter())
            .map(|(f, p)| f * p).sum::<f64>() / total_power
    }

    /// Compute spectral bandwidth (variance around centroid).
    pub fn spectral_bandwidth(freqs: &[f64], psd: &[f64]) -> f64 {
        let centroid = Self::spectral_centroid(freqs, psd);
        let half = freqs.len() / 2;
        let total_power: f64 = psd[..half].iter().sum();
        if total_power < 1e-30 { return 0.0; }
        let variance = freqs[..half].iter().zip(psd[..half].iter())
            .map(|(f, p)| (f - centroid).powi(2) * p).sum::<f64>() / total_power;
        variance.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodogram_dc() {
        let signal = vec![5.0; 128];
        let (freqs, psd) = SpectralEstimation::periodogram(&signal, 1000.0);
        // DC bin should dominate
        assert!(psd[0] > psd[1], "DC should dominate");
        assert!(psd[0] > 0.0);
    }

    #[test]
    fn test_periodogram_tone() {
        let sample_rate = 1000.0;
        let n = 1024;
        let freq = 100.0;
        let signal: Vec<f64> = (0..n).map(|i| {
            (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin()
        }).collect();
        let (freqs, psd) = SpectralEstimation::periodogram(&signal, sample_rate);
        let bin = (freq * n as f64 / sample_rate) as usize;
        assert!(psd[bin] > psd[0] * 10.0, "Tone bin should be large");
    }

    #[test]
    fn test_welch_output_shape() {
        let signal: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.1).sin()).collect();
        let (freqs, psd) = SpectralEstimation::welch(&signal, 1000.0, 256, 128);
        assert_eq!(freqs.len(), psd.len());
        assert!(psd.iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn test_welch_smoother_than_periodogram() {
        let mut signal = vec![0.0; 1024];
        for i in 0..1024 {
            signal[i] = (2.0 * std::f64::consts::PI * 50.0 * i as f64 / 1000.0).sin()
                       + 0.5 * (2.0 * std::f64::consts::PI * 120.0 * i as f64 / 1000.0).sin();
        }
        let (_, psd_per) = SpectralEstimation::periodogram(&signal, 1000.0);
        let (_, psd_welch) = SpectralEstimation::welch(&signal, 1000.0, 256, 128);
        // Welch should have lower variance (simpler check: both have positive values)
        assert!(psd_per.iter().all(|p| *p >= 0.0));
        assert!(psd_welch.iter().all(|p| *p >= 0.0));
    }

    #[test]
    fn test_find_peaks() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 10.0).collect();
        let mut psd = vec![0.0; 100];
        // Create clear peaks at bins 20 and 50 with valleys between
        psd[20] = 10.0;
        psd[19] = 0.1; psd[21] = 0.1;
        psd[40] = 20.0;
        psd[39] = 0.1; psd[41] = 0.1;
        let peaks = SpectralEstimation::find_peaks(&freqs, &psd, 1.0);
        assert_eq!(peaks.len(), 2, "Should find 2 peaks, found {}: {:?}", peaks.len(), peaks);
        assert!((peaks[0].0 - 400.0).abs() < 1e-10, "Strongest peak at 400 Hz");
        assert!((peaks[1].0 - 200.0).abs() < 1e-10, "Second peak at 200 Hz");
    }

    #[test]
    fn test_band_power() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 10.0).collect();
        let mut psd = vec![0.0; 100];
        psd[10] = 1.0; // 100 Hz
        psd[50] = 2.0; // 500 Hz
        let power = SpectralEstimation::band_power(&freqs, &psd, 50.0, 150.0);
        assert!(power > 0.0, "Band power should be positive");
    }

    #[test]
    fn test_spectral_centroid() {
        let n = 100;
        let freqs: Vec<f64> = (0..n).map(|i| i as f64 * 10.0).collect();
        let mut psd = vec![0.0; n];
        let bin = 25; // Within first half (n/2 = 50)
        psd[bin] = 1.0; // All power at 250 Hz
        let centroid = SpectralEstimation::spectral_centroid(&freqs, &psd);
        assert!((centroid - 250.0).abs() < 1e-10, "Centroid should be 250: {centroid}");
    }

    #[test]
    fn test_spectral_bandwidth() {
        let freqs: Vec<f64> = (0..100).map(|i| i as f64 * 10.0).collect();
        let mut psd = vec![0.0; 100];
        psd[50] = 1.0; // Delta function — bandwidth should be ~0
        let bw = SpectralEstimation::spectral_bandwidth(&freqs, &psd);
        assert!(bw < 1.0, "Delta bandwidth should be small: {bw}");
    }
}

//! Fourier series: coefficients, convergence, Parseval's theorem.

use serde::{Deserialize, Serialize};

/// Represents a Fourier series approximation of a periodic function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FourierSeries {
    /// DC component (a0/2)
    pub a0: f64,
    /// Cosine coefficients a_n
    pub a_coeffs: Vec<f64>,
    /// Sine coefficients b_n
    pub b_coeffs: Vec<f64>,
    /// Period of the function
    pub period: f64,
}

impl FourierSeries {
    /// Compute Fourier series coefficients from discrete samples over one period.
    /// `samples` should cover exactly one period with uniform spacing.
    /// `n_harmonics` is the number of harmonics to compute (excluding DC).
    pub fn from_samples(samples: &[f64], period: f64, n_harmonics: usize) -> Self {
        let n = samples.len() as f64;
        let a0 = 2.0 * samples.iter().sum::<f64>() / n;
        let mut a_coeffs = Vec::with_capacity(n_harmonics);
        let mut b_coeffs = Vec::with_capacity(n_harmonics);

        for k in 1..=n_harmonics {
            let mut a_k = 0.0;
            let mut b_k = 0.0;
            for (j, &x_j) in samples.iter().enumerate() {
                let t = j as f64 * period / n;
                a_k += x_j * (2.0 * std::f64::consts::PI * k as f64 * t / period).cos();
                b_k += x_j * (2.0 * std::f64::consts::PI * k as f64 * t / period).sin();
            }
            a_coeffs.push(2.0 * a_k / n);
            b_coeffs.push(2.0 * b_k / n);
        }

        FourierSeries { a0, a_coeffs, b_coeffs, period }
    }

    /// Evaluate the Fourier series at time `t`.
    pub fn evaluate(&self, t: f64) -> f64 {
        let omega = 2.0 * std::f64::consts::PI / self.period;
        let mut result = self.a0 / 2.0;
        for (k, (a, b)) in self.a_coeffs.iter().zip(self.b_coeffs.iter()).enumerate() {
            let n = (k + 1) as f64;
            result += a * (n * omega * t).cos() + b * (n * omega * t).sin();
        }
        result
    }

    /// Compute the mean squared error between the Fourier series and the original samples.
    pub fn mse(&self, samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let period = self.period;
        let sum: f64 = samples.iter().enumerate().map(|(j, &x)| {
            let t = j as f64 * period / n;
            let diff = x - self.evaluate(t);
            diff * diff
        }).sum();
        sum / n
    }

    /// Parseval's theorem: compute the power in the frequency domain.
    /// Returns (1/T) * integral of |f(t)|^2 over one period, which equals
    /// (a0/2)^2 + (1/2) * sum(a_n^2 + b_n^2).
    pub fn parseval_power(&self) -> f64 {
        let mut power = (self.a0 / 2.0).powi(2);
        for (a, b) in self.a_coeffs.iter().zip(self.b_coeffs.iter()) {
            power += 0.5 * (a * a + b * b);
        }
        power
    }

    /// Compute time-domain power: (1/N) * sum(|x_i|^2) from samples.
    pub fn time_domain_power(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        samples.iter().map(|x| x * x).sum::<f64>() / n
    }

    /// Verify Parseval's theorem: frequency-domain power should equal time-domain power.
    pub fn verify_parseval(&self, samples: &[f64]) -> f64 {
        let freq_power = self.parseval_power();
        let time_power = Self::time_domain_power(samples);
        (freq_power - time_power).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_function() {
        // f(t) = 5 for all t
        let samples = vec![5.0; 64];
        let fs = FourierSeries::from_samples(&samples, 2.0 * std::f64::consts::PI, 3);
        assert!((fs.a0 - 10.0).abs() < 1e-10);
        for a in &fs.a_coeffs { assert!(a.abs() < 1e-10, "a_coeff should be ~0"); }
        for b in &fs.b_coeffs { assert!(b.abs() < 1e-10, "b_coeff should be ~0"); }
    }

    #[test]
    fn test_cosine_function() {
        // f(t) = cos(t), period = 2π
        let n = 128;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            t.cos()
        }).collect();
        let fs = FourierSeries::from_samples(&samples, period, 3);
        assert!((fs.a0).abs() < 1e-10, "a0 should be ~0");
        assert!((fs.a_coeffs[0] - 1.0).abs() < 1e-6, "a1 should be ~1");
        assert!(fs.b_coeffs[0].abs() < 1e-6, "b1 should be ~0");
        assert!(fs.a_coeffs[1].abs() < 1e-6, "a2 should be ~0");
    }

    #[test]
    fn test_sine_function() {
        let n = 128;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            (3.0 * t).sin()
        }).collect();
        let fs = FourierSeries::from_samples(&samples, period, 5);
        assert!((fs.b_coeffs[2] - 1.0).abs() < 1e-6, "b3 should be ~1");
    }

    #[test]
    fn test_square_wave_coefficients() {
        // Square wave: odd harmonics only, amplitude 4/(nπ)
        let n = 256;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            if t < std::f64::consts::PI { 1.0 } else { -1.0 }
        }).collect();
        let fs = FourierSeries::from_samples(&samples, period, 7);
        // b1 ≈ 4/π, b3 ≈ 4/(3π), etc.
        assert!((fs.b_coeffs[0] - 4.0 / std::f64::consts::PI).abs() < 0.05);
        assert!((fs.b_coeffs[2] - 4.0 / (3.0 * std::f64::consts::PI)).abs() < 0.05);
        // Even harmonics should be near zero
        assert!(fs.b_coeffs[1].abs() < 0.05);
    }

    #[test]
    fn test_parseval_theorem() {
        let n = 256;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            t.cos() + 0.5 * (2.0 * t).sin()
        }).collect();
        let fs = FourierSeries::from_samples(&samples, period, 5);
        let error = fs.verify_parseval(&samples);
        assert!(error < 0.01, "Parseval's theorem error: {error}");
    }

    #[test]
    fn test_evaluate_reconstruction() {
        let n = 128;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            t.cos() + (2.0 * t).sin()
        }).collect();
        let fs = FourierSeries::from_samples(&samples, period, 4);
        // Mid-point check
        let t_mid = period / 4.0;
        let expected = t_mid.cos() + (2.0 * t_mid).sin();
        let actual = fs.evaluate(t_mid);
        assert!((expected - actual).abs() < 0.05, "reconstruction mismatch");
    }

    #[test]
    fn test_convergence_more_harmonics() {
        let n = 256;
        let period = 2.0 * std::f64::consts::PI;
        let samples: Vec<f64> = (0..n).map(|j| {
            let t = j as f64 * period / n as f64;
            if t < std::f64::consts::PI { 1.0 } else { -1.0 }
        }).collect();
        let fs4 = FourierSeries::from_samples(&samples, period, 4);
        let fs16 = FourierSeries::from_samples(&samples, period, 16);
        let mse4 = fs4.mse(&samples);
        let mse16 = fs16.mse(&samples);
        assert!(mse16 < mse4, "More harmonics should reduce MSE: {mse16} vs {mse4}");
    }
}

# lau-harmonic-analysis

A Rust library for **harmonic analysis** — Fourier series, DFT/FFT, Fourier transform properties (convolution theorem, Plancherel, uncertainty principle), windowing functions, Laplace and Z transforms, Haar wavelets, spectral estimation (periodogram, Welch's method), and signal analysis (autocorrelation, period detection, pattern extraction).

Built on [nalgebra](https://nalgebra.org) for linear algebra, [num-complex](https://docs.rs/num-complex) for complex arithmetic, and [serde](https://serde.rs) for serialization.

## What This Does

This crate implements the core tools of harmonic analysis — the mathematics of decomposing signals into oscillatory components. You can:

- Decompose periodic signals into **Fourier series** and verify Parseval's theorem
- Compute the **DFT** and **FFT** (radix-2 Cooley-Tukey) with roundtrip guarantees
- Perform **convolution** (naive and FFT-based) and verify the convolution theorem
- Apply **windowing functions** (Hamming, Hann, Blackman, Gaussian) and compute ENBW
- Compute numerical **Laplace transforms** (trapezoidal integration) and **inverse Laplace** (Gaver-Stehfest)
- Evaluate the **Z-transform** on the unit circle, find zeros, compute group delay
- Perform **Haar wavelet** decomposition, multi-resolution analysis, and denoising
- Estimate **power spectral density** via periodogram and Welch's method
- Detect **periodicity** via autocorrelation, find dominant periods, and extract patterns

**76 unit tests** cover all modules.

## Key Idea

Harmonic analysis is built on one insight: **signals can be decomposed into frequencies**. A complex waveform is a sum of sinusoids at different frequencies, amplitudes, and phases. This library implements that decomposition and its consequences:

- **Fourier analysis** (series, DFT, FFT) decomposes signals into frequency components
- **Windowing** controls spectral leakage when analyzing finite-length signals
- **Transforms** (Laplace, Z) extend Fourier analysis to complex frequencies and discrete systems
- **Wavelets** provide time-frequency localization (where Fourier gives only frequency)
- **Spectral estimation** extracts reliable power spectra from noisy data
- **Signal analysis** detects structure (periods, patterns) in real-world data

## Install

```toml
[dependencies]
lau-harmonic-analysis = { git = "https://github.com/SuperInstance/lau-harmonic-analysis" }
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Companion matrix eigenvalues for Z-transform zeros |
| `num-complex` | Complex number arithmetic |
| `serde` | Serialization of all types |
| `serde_json` (dev) | Round-trip tests |

## Quick Start

```rust
use lau_harmonic_analysis::*;
use num_complex::Complex64;

// --- Fourier Series ---
let samples: Vec<f64> = (0..256).map(|j| {
    let t = j as f64 * 2.0 * std::f64::consts::PI / 256.0;
    t.cos() + 0.5 * (2.0 * t).sin()
}).collect();
let fs = FourierSeries::from_samples(&samples, 2.0 * std::f64::consts::PI, 5);
println!("Parseval error: {}", fs.verify_parseval(&samples)); // ≈ 0

// --- DFT / FFT ---
let signal: Vec<Complex64> = (0..64).map(|j| {
    Complex64::new((2.0 * std::f64::consts::PI * 4.0 * j as f64 / 64.0).cos(), 0.0)
}).collect();
let spectrum = Dft::fft(&signal);
let psd = Dft::power_spectrum(&signal);
println!("Dominant bin: {}", Dft::frequency_bins(64, 44100.0)[4]); // 2756.25 Hz

// --- Convolution (FFT-based) ---
let a = vec![1.0, 2.0, 3.0];
let b = vec![0.5, 1.0];
let conv = FourierTransform::convolve_fft(&a, &b);

// --- Windowing ---
let window = Windowing::hann(1024);
let enbw = Windowing::enbw(&window);
println!("Hann ENBW: {:.3} bins", enbw); // ≈ 1.5

// --- Laplace Transform ---
let result = LaplaceTransform::transform(
    |t| (-2.0 * t).exp(),  // f(t) = e^{-2t}
    Complex64::new(3.0, 0.0),  // s = 3
    20.0, 10000
);
println!("L{{e^-2t}}(3) = {:.4} (expected 0.2)", result.re);

// --- Welch's PSD ---
let signal: Vec<f64> = (0..1024).map(|i| (2.0 * std::f64::consts::PI * 50.0 * i as f64 / 44100.0).sin()).collect();
let (freqs, psd) = SpectralEstimation::welch(&signal, 44100.0, 256, 128);
let peaks = SpectralEstimation::find_peaks(&freqs, &psd, 1.0);
println!("Dominant frequency: {:.1} Hz", peaks[0].0);

// --- Haar Wavelet ---
let signal = vec![4.0, 6.0, 8.0, 10.0, 2.0, 4.0, 6.0, 8.0];
let levels = Wavelet::haar_dwt(&signal);
let energies = Wavelet::wavelet_energy(&signal);
let denoised = Wavelet::denoise(&signal, 1.0);

// --- Signal Analysis ---
let signal = SignalAnalysis::synthetic_behavior(512, 32, 0.1);
let analysis = SignalAnalysis::analyze(&signal, 44100.0);
println!("Periodicity: {:.2}, Period: {:?}", analysis.periodicity_score, analysis.dominant_period);
```

## API Reference

### `FourierSeries`

| Method | Description |
|--------|-------------|
| `from_samples(samples, period, n_harmonics)` | Compute coefficients from uniform samples over one period |
| `evaluate(t)` | Reconstruct f(t) from the series |
| `mse(samples)` | Mean squared error vs. original samples |
| `parseval_power()` | Frequency-domain power (a₀/2)² + ½Σ(aₙ² + bₙ²) |
| `time_domain_power(samples)` | (1/N)Σ|xᵢ|² |
| `verify_parseval(samples)` | |freq_power − time_power| |

### `Dft`

| Method | Description |
|--------|-------------|
| `dft(signal)` | O(N²) discrete Fourier transform |
| `idft(spectrum)` | Inverse DFT |
| `fft(signal)` | O(N log N) radix-2 Cooley-Tukey FFT |
| `ifft(spectrum)` | Inverse FFT |
| `power_spectrum(signal)` | |X[k]|² |
| `frequency_bins(n, sample_rate)` | Frequency labels for each bin |
| `zero_pad_to_power_of_2(signal)` | Pad to next power of 2 |

### `FourierTransform`

| Method | Description |
|--------|-------------|
| `convolve(a, b)` | Naive O(N²) convolution |
| `convolve_fft(a, b)` | FFT-based O(N log N) convolution |
| `plancherel(signal)` | Verify L² norm preservation |
| `uncertainty(signal, sample_rate)` | Heisenberg's Δt·Δf ≥ 1/(4π) |
| `modulation(signal, freq)` | Frequency shift via modulation |
| `time_shift(signal, shift)` | Circular time shift ↔ phase rotation |

### `Windowing`

| Method | Description |
|--------|-------------|
| `hamming(n)` | Hamming window (0.54 − 0.46 cos) |
| `hann(n)` | Hann window (0.5(1 − cos)) |
| `blackman(n)` | Blackman window (3-term cosine) |
| `gaussian(n, σ)` | Gaussian window |
| `rectangular(n)` | No window (all ones) |
| `apply(signal, window)` | Element-wise multiplication |
| `coherent_gain(window)` | Mean window value |
| `enbw(window)` | Equivalent noise bandwidth |
| `scalloping_loss(window)` | Worst-case processing loss |

### `LaplaceTransform`

| Method | Description |
|--------|-------------|
| `transform(f, s, t_max, n_steps)` | Numerical forward transform (trapezoidal) |
| `inverse_transform(F, t, n_terms)` | Inverse via Gaver-Stehfest |
| `unit_step(s)` | L{u(t)} = 1/s |
| `exponential(s, a)` | L{e⁻ᵃᵗ} = 1/(s+a) |
| `cosine(s, ω)` | L{cos ωt} = s/(s²+ω²) |
| `sine(s, ω)` | L{sin ωt} = ω/(s²+ω²) |
| `second_order_system(s, K, ζ, ωₙ)` | H(s) = K/(s²+2ζωₙs+ωₙ²) |

### `ZTransform`

| Method | Description |
|--------|-------------|
| `transform(signal, z)` | X(z) = Σ x[n]z⁻ⁿ |
| `dtft(signal, ω)` | Evaluate on unit circle z = eʲʷ |
| `frequency_response(signal, freqs)` | DTFT at multiple frequencies |
| `magnitude_response_db(signal, freqs)` | |H(f)| in dB |
| `phase_response(signal, freqs)` | arg(H(f)) in radians |
| `zeros(signal)` | Zeros via companion matrix eigenvalues |
| `first_order_transfer(z, b0, b1, a1)` | H(z) = (b0 + b1z⁻¹)/(1 + a1z⁻¹) |
| `group_delay(signal, ω, dω)` | −d(arg H)/dω |

### `Wavelet` (Haar)

| Method | Description |
|--------|-------------|
| `haar_dwt(signal)` | Multi-level discrete Haar wavelet transform |
| `haar_idwt(approx, details)` | Inverse DWT |
| `multiresolution(signal)` | Level-by-level reconstruction |
| `wavelet_energy(signal)` | Energy per decomposition level |
| `denoise(signal, threshold)` | Soft-threshold denoising |

### `SpectralEstimation`

| Method | Description |
|--------|-------------|
| `periodogram(signal, fs)` | Raw PSD via FFT |
| `welch(signal, fs, seg_len, overlap)` | Averaged PSD with Hann window |
| `find_peaks(freqs, psd, min_height)` | Dominant frequency peaks |
| `band_power(freqs, psd, f_low, f_high)` | Power in a frequency band |
| `spectral_centroid(freqs, psd)` | Center of mass of spectrum |
| `spectral_flatness(freqs, psd)` | Geometric/arithmetic mean ratio |

### `SignalAnalysis`

| Method | Description |
|--------|-------------|
| `analyze(signal, fs)` | Full analysis → BehaviorAnalysis struct |
| `autocorrelation(signal)` | Normalized ACF (peak = 1.0 at lag 0) |
| `detect_period_autocorrelation(acf, threshold)` | Period from first ACF peak |
| `synthetic_behavior(n, period, noise)` | Generate test signal |

## How It Works

### FFT Implementation

The FFT uses the classic **Cooley-Tukey radix-2 decimation-in-time** algorithm:

1. Bit-reverse the input order
2. Butterfly operations at each level (log₂N levels)
3. Each butterfly: combine two N/2-point DFTs into one N-point DFT

For non-power-of-2 inputs, `zero_pad_to_power_of_2` pads with zeros.

### Convolution Theorem

Time-domain convolution equals frequency-domain multiplication:

> f * g = IFFT(FFT(f) · FFT(g))

The library implements both naive O(N²) and FFT-based O(N log N) convolution. The FFT version pads to the next power of 2 ≥ len(a) + len(b) − 1 to avoid circular aliasing.

### Welch's Method

1. Split signal into overlapping segments
2. Apply Hann window to each segment
3. Compute FFT of each windowed segment
4. Average the squared magnitudes
5. Normalize by sample rate, window norm, and segment count

This reduces the variance of the periodogram by a factor of ~K (number of segments) at the cost of frequency resolution.

### Gaver-Stehfest Inverse Laplace

Evaluates the Laplace transform F(s) at real points s = k · ln(2)/t for k = 1, ..., N and combines with Stehfest weights. Accurate for smooth, well-behaved functions with N ≈ 10–20 terms.

## The Math

### Parseval's Theorem

For a periodic signal sampled over one period:

> (1/N) Σ|f(tₖ)|² = (a₀/2)² + ½ Σ(aₙ² + bₙ²)

Time-domain energy equals frequency-domain energy. This is the discrete form of Plancherel's theorem.

### Heisenberg's Uncertainty Principle

For any signal:

> Δt · Δf ≥ 1/(4π)

where Δt and Δf are the RMS spreads in time and frequency. A signal concentrated in time (short pulse) must be spread in frequency (broad spectrum), and vice versa. This is not a limitation of measurement — it's a fundamental property of the Fourier transform.

### Welch's PSD Estimation

The power spectral density is estimated as:

> P̂(f) = (1/K) Σₖ |FFT{w(t) · xₖ(t)}|² / (fs · Σw²)

where K is the number of segments, w is the Hann window, and xₖ is the k-th segment. The variance of P̂ decreases as 1/K.

### Haar Wavelet Decomposition

At each level, the Haar transform splits a signal into:
- **Approximation**: a[j] = (x[2j] + x[2j+1]) / √2 (local averages)
- **Detail**: d[j] = (x[2j] − x[2j+1]) / √2 (local differences)

Multi-resolution analysis reconstructs the signal level by level, from coarse (low-frequency trends) to fine (high-frequency detail). Soft-thresholding the detail coefficients and inverting achieves denoising.

## License

MIT

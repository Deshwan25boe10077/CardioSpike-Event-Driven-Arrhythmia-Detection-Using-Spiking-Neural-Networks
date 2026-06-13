import numpy as np
from scipy.signal import find_peaks

def generate_synthetic_ecg(num_beats=200, fs=250, duration_per_beat=0.8):
    """Generate synthetic ECG-like signal"""
    samples_per_beat = int(fs * duration_per_beat)
    total_samples = num_beats * samples_per_beat
    
    t = np.linspace(0, num_beats * duration_per_beat, total_samples)
    
    # Synthetic ECG: P wave + QRS complex + T wave (add this as reference digram in the readme too)
    ecg = (
        0.2 * np.sin(2 * np.pi * 1 * t) +      # P wave
        0.8 * np.sin(2 * np.pi * 4 * t) +      # QRS (stronger)
        0.3 * np.sin(2 * np.pi * 1.5 * t) +    # T wave
        0.05 * np.random.randn(total_samples)  # Noise
    )
    
    # Shift to positive range ( I might fix this sometime)
    ecg = ecg - ecg.min() + 1
    return ecg, fs

def extract_beats(signal, fs=250, window_ms=800):
    """Extract beats from signal"""
    # Find peaks (R waves) with lower threshold
    peaks, _ = find_peaks(signal, height=0.5, distance=int(fs*0.6))
    
    if len(peaks) == 0:
        print("⚠ No peaks detected, returning full signal as single beat")
        return np.array([signal[:int(fs * window_ms / 1000)]])
    
    window_samples = int(fs * window_ms / 1000)
    beats = []
    
    for peak in peaks:
        start = max(0, peak - window_samples // 2)
        end = min(len(signal), peak + window_samples // 2)
        if end - start == window_samples:
            beats.append(signal[start:end])
    
    return np.array(beats)

# I just removed the past save section because of certain fallback errors

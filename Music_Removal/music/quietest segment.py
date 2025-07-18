import librosa
import numpy as np
import soundfile as sf

def rms_db(signal):
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms + 1e-10)

def match_amplitude(y, sr, target_db, ref_len=5.0):
    """Scale full track so first ref_len seconds match target RMS (dB)."""
    ref_samples = int(ref_len * sr)
    ref_segment = y[:ref_samples]
    current_db = rms_db(ref_segment)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    return y * gain

# Input files (6 tracks)
files = [
    "track1.wav",
    "track2.wav",
    "track3.wav",
    "track4.wav",
    "track5.wav",
    "track6.wav"
]

signals = []
sr = None

# Load and normalize each track based on Track 1's first 5 seconds RMS
for f in files:
    y, sr_ = librosa.load(f, sr=None, mono=True)
    sr = sr or sr_
    if sr_ != sr:
        raise ValueError("All tracks must share the same sample rate")
    signals.append(y)

target_db = rms_db(signals[0][:int(5 * sr)])
signals = [match_amplitude(y, sr, target_db) for y in signals]

# Find the longest track
max_len = max(len(y) for y in signals)

# Pad tracks with NaNs to handle length differences
padded = []
for y in signals:
    pad_len = max_len - len(y)
    if pad_len > 0:
        y = np.concatenate([y, np.full(pad_len, np.nan)])
    padded.append(y)

stacked = np.stack(padded, axis=0)  # shape: (tracks, samples)

# Define window size (1/10th second)
window_size = int(sr / 10)
output_segments = []

# Process each window
for start in range(0, max_len, window_size):
    end = min(start + window_size, max_len)
    window = stacked[:, start:end]
    
    # Compute RMS for each track in this window (ignoring NaNs)
    rms_vals = []
    for i in range(window.shape[0]):
        seg = window[i]
        seg = seg[~np.isnan(seg)]
        if len(seg) == 0:
            rms_vals.append(np.inf)  # no data, exclude
        else:
            rms_vals.append(np.sqrt(np.mean(seg**2)))
    
    # Pick track with lowest RMS
    chosen_idx = int(np.argmin(rms_vals))
    
    # Use that entire segment
    chosen_seg = window[chosen_idx]
    # Replace NaNs with 0s in case it's shorter
    chosen_seg = np.nan_to_num(chosen_seg)
    output_segments.append(chosen_seg)

# Concatenate all chosen segments
output = np.concatenate(output_segments)

# Normalize output to avoid clipping
output = output / (np.max(np.abs(output)) + 1e-10) * 0.99

sf.write("quietest_mix_windows.wav", output, sr)
print("Saved: quietest_mix_windows.wav")

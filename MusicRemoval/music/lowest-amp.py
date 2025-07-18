import librosa
import numpy as np
import soundfile as sf

def rms_db(signal):
    """Return RMS loudness in dB for a signal."""
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

# List your 6 input files here
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

# Load and normalize all tracks
for f in files:
    y, sr_ = librosa.load(f, sr=None, mono=True)
    sr = sr or sr_
    if sr_ != sr:
        raise ValueError("All tracks must share the same sample rate")
    signals.append(y)

# Determine RMS target from the first 5 seconds of track1
target_db = rms_db(signals[0][:int(5 * sr)])
signals = [match_amplitude(y, sr, target_db) for y in signals]

# Find the longest track (so we know final output length)
max_len = max(len(y) for y in signals)

# Pad tracks with NaNs (not zeros) to handle different lengths safely
padded = []
for y in signals:
    pad_len = max_len - len(y)
    if pad_len > 0:
        y = np.concatenate([y, np.full(pad_len, np.nan)])
    padded.append(y)

stacked = np.stack(padded, axis=0)  # shape: (tracks, samples)

# For each sample, ignore NaNs and pick value with lowest absolute amplitude
output = []
for i in range(max_len):
    col = stacked[:, i]
    valid = col[~np.isnan(col)]
    if len(valid) == 0:
        # No tracks have data here — skip this sample entirely
        continue
    # Pick the value with smallest absolute amplitude
    idx = np.argmin(np.abs(valid))
    output.append(valid[idx])

output = np.array(output)

# Normalize final output to avoid clipping
output = output / (np.max(np.abs(output)) + 1e-10) * 0.99

sf.write("quietest_mix_6tracks.wav", output, sr)
print("Saved: quietest_mix_6tracks.wav")

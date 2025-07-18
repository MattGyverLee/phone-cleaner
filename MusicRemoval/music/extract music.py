import librosa
import numpy as np
import soundfile as sf

# Threshold for deciding if a frequency "exists" in a track (relative to max magnitude)
PRESENCE_THRESHOLD = 0.01  # adjust as needed (1% of max)

def rms_db(signal):
    rms = np.sqrt(np.mean(signal**2))
    return 20 * np.log10(rms + 1e-10)

def match_amplitude(y, sr, target_db, ref_len=5.0):
    ref_samples = int(ref_len * sr)
    ref_segment = y[:ref_samples]
    current_db = rms_db(ref_segment)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    return y * gain

# Load 3 mono tracks
files = ["track1.wav", "track2.wav", "track3.wav"]
signals = []
sr = None
for f in files:
    y, sr_ = librosa.load(f, sr=None, mono=True)
    sr = sr or sr_
    if sr_ != sr:
        raise ValueError("All tracks must share the same sample rate")
    signals.append(y)

# Trim to shortest length
min_len = min(len(y) for y in signals)
signals = [y[:min_len] for y in signals]

# Normalize all based on first 5 seconds of track 1
target_db = rms_db(signals[0][:int(5 * sr)])
signals = [match_amplitude(y, sr, target_db) for y in signals]

# Compute STFTs
spectrograms = [librosa.stft(y, n_fft=2048, hop_length=512) for y in signals]
magnitudes = [np.abs(S) for S in spectrograms]

# Build a strict intersection mask:
# Only keep bins where ALL tracks exceed the threshold relative to their own max
presence_masks = []
for mag in magnitudes:
    max_val = np.max(mag)
    presence_masks.append(mag > (PRESENCE_THRESHOLD * max_val))

# Combine masks (logical AND)
common_mask = presence_masks[0] & presence_masks[1] & presence_masks[2]

# Combine magnitudes: use the minimum value where all present, else 0
common_mag = np.where(common_mask, np.minimum.reduce(magnitudes), 0.0)

# Use phase from the first track
phase = np.angle(spectrograms[0])
S_common = common_mag * np.exp(1j * phase)

# Inverse STFT to reconstruct
y_common = librosa.istft(S_common, hop_length=512)

# Normalize output to avoid clipping
y_common = y_common / (np.max(np.abs(y_common)) + 1e-10) * 0.99

sf.write("common_strict.wav", y_common, sr)
print("Saved: common_strict.wav")

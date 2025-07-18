import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter

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

files = ["track1.wav", "track2.wav", "track3.wav"]
signals = []
sr = None
for f in files:
    y, sr_ = librosa.load(f, sr=None, mono=True)
    sr = sr or sr_
    if sr_ != sr:
        raise ValueError("All tracks must have same sample rate")
    signals.append(y)

min_len = min(len(y) for y in signals)
signals = [y[:min_len] for y in signals]

target_db = rms_db(signals[0][:int(5 * sr)])
signals = [match_amplitude(y, sr, target_db) for y in signals]

# STFT
spectrograms = [librosa.stft(y, n_fft=2048, hop_length=512) for y in signals]
magnitudes = [np.abs(S) for S in spectrograms]

# Soft overlap: geometric mean keeps shared energy without zeroing everything
geom_mean = np.exp(np.mean([np.log(m + 1e-10) for m in magnitudes], axis=0))

# Optional smoothing (reduce clickiness)
smoothed = gaussian_filter(geom_mean, sigma=1)

# Use phase from first track
phase = np.angle(spectrograms[0])
S_common = smoothed * np.exp(1j * phase)

y_common = librosa.istft(S_common, hop_length=512)
y_common = y_common / (np.max(np.abs(y_common)) + 1e-10) * 0.99

sf.write("common_music-soft.wav", y_common, sr)
print("Saved: common_music-soft.wav")

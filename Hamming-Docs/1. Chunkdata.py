import librosa
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import parselmouth # Needs Praat installed for parselmouth to work
from praatio import tgio
from collections import Counter
import os

def guess_phonetic_segments(audio_path, n_clusters=5, frame_duration_ms=25, hop_duration_ms=10, smoothing_window_frames=5):
    """
    Performs unsupervised clustering to guess phonetic segments (Silence, Vowel, Consonant)
    and exports them to a Praat TextGrid for manual approval.

    Args:
        audio_path (str): Path to the input audio file.
        n_clusters (int): Number of clusters for K-Means. Experiment with this (e.g., 3, 5, 7).
        frame_duration_ms (int): Duration of analysis frames in milliseconds.
        hop_duration_ms (int): Hop duration between frames in milliseconds.
        smoothing_window_frames (int): Number of frames for majority-vote smoothing.
    Returns:
        str: Path to the generated TextGrid file.
    """
    print(f"Processing audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None) # Load at original sample rate

    frame_length_samples = int(sr * frame_duration_ms / 1000)
    hop_length_samples = int(sr * hop_duration_ms / 1000)

    # Ensure consistent framing for all features
    num_frames = librosa.util.frame(y, frame_length=frame_length_samples, hop_length=hop_length_samples).shape[1]
    times = librosa.frames_to_time(np.arange(num_frames), sr=sr, hop_length=hop_length_samples)

    # --- Feature Extraction ---
    print("Extracting acoustic features...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length_samples, hop_length=hop_length_samples)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, frame_length=frame_length_samples, hop_length=hop_length_samples).T

    # Parselmouth features for pitch and formants
    sound_parsel = parselmouth.Sound(audio_path)
    pitch = sound_parsel.to_pitch(time_step=hop_duration_ms/1000)
    formants = sound_parsel.to_formant_burg(time_step=hop_duration_ms/1000, max_number_of_formants=5)

    f0_values = np.array([pitch.get_value_at_time(t) if pitch.get_value_at_time(t) != 0 else np.nan for t in times])
    f1_values = np.array([formants.get_value_at_time(1, t) for t in times])
    f2_values = np.array([formants.get_value_at_time(2, t) for t in times])

    # Combine all features
    features = np.vstack([
        rms,
        zcr,
        mfcc.T, # Ensure MFCCs are correctly shaped
        f0_values,
        f1_values,
        f2_values
    ]).T

    # Handle NaNs (e.g., from unvoiced frames in pitch/formants). Imputation or 0.
    # For now, let's fill NaNs with 0, but a more sophisticated strategy might be needed.
    features[np.isnan(features)] = 0

    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    print(f"Features extracted. Shape: {scaled_features.shape}")

    # --- Clustering ---
    print(f"Performing K-Means clustering with {n_clusters} clusters...")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    print("Clustering complete.")

    # --- Interpret Clusters (Crucial Manual Step) ---
    print("\n--- Cluster Feature Averages (for manual interpretation) ---")
    cluster_mapping_suggestions = {}
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        if len(cluster_indices) > 0:
            avg_rms = np.mean(rms[cluster_indices])
            avg_zcr = np.mean(zcr[cluster_indices])
            avg_f0 = np.nanmean(f0_values[cluster_indices]) # Use nanmean to ignore 0/NaNs
            avg_f1 = np.nanmean(f1_values[cluster_indices])
            avg_f2 = np.nanmean(f2_values[cluster_indices])

            print(f"Cluster {i}:")
            print(f"  Avg RMS: {avg_rms:.4f}")
            print(f"  Avg ZCR: {avg_zcr:.4f}")
            print(f"  Avg F0 (avg, incl 0): {np.mean(f0_values[cluster_indices]):.1f} Hz") # Show overall F0 including 0s
            print(f"  Avg F0 (voiced only): {avg_f0:.1f} Hz (NaN if all unvoiced)")
            print(f"  Avg F1 (clear F): {avg_f1:.1f} Hz (NaN if no clear formants)")
            print(f"  Avg F2 (clear F): {avg_f2:.1f} Hz (NaN if no clear formants)")
            print("-" * 20)
            # Based on these values, you'll suggest a label for the cluster.
            # E.g., if avg_rms is very low and avg_f0 is 0: likely "Silence"
            # If high rms, clear F0, specific F1/F2: likely "Vowel"
            # If high ZCR, low F0: likely "Unvoiced Consonant"
            # If low rms but clear F0, no clear formants: likely "Voiced Consonant"
            cluster_mapping_suggestions[i] = "UNKNOWN" # Placeholder
    print("--- End Cluster Analysis ---")
    print("\nBased on the above, you will map cluster IDs to 'Silence', 'Vowel', 'Consonant', etc.")
    print("Example: cluster_to_label_map = {0: 'Silence', 1: 'Vowel', 2: 'Consonant_Noisy', 3: 'Consonant_Voiced', 4: 'Vowel_Reduced'}")

    # You MUST fill this manually after inspecting the printed cluster averages:
    cluster_to_label_map = {
        # EXAMPLE MAPPING - YOU NEED TO DETERMINE THIS BASED ON YOUR DATA!
        # 0: "Silence",
        # 1: "Vowel",
        # 2: "Consonant_Voiced",
        # 3: "Consonant_Unvoiced",
        # 4: "Noise"
        # Placeholder for demonstration, replace with your actual mapping
        i: f"Cluster_{i}" for i in range(n_clusters)
    }
    print(f"\nUsing assumed cluster mapping: {cluster_to_label_map}")
    predicted_labels = [cluster_to_label_map[label] for label in cluster_labels]

    # --- Smoothing Labels ---
    print(f"Smoothing labels with window size: {smoothing_window_frames} frames...")
    smoothed_labels = []
    for i in range(len(predicted_labels)):
        start_idx = max(0, i - smoothing_window_frames // 2)
        end_idx = min(len(predicted_labels), i + smoothing_window_frames // 2 + 1)
        segment_window = predicted_labels[start_idx:end_idx]
        most_common = Counter(segment_window).most_common(1)[0][0]
        smoothed_labels.append(most_common)

    # --- Convert to Praat TextGrid Segments ---
    segments = []
    if len(smoothed_labels) > 0:
        current_label = smoothed_labels[0]
        current_start_time = times[0]
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] != current_label:
                segments.append(tgio.Interval(current_start_time, times[i], current_label))
                current_label = smoothed_labels[i]
                current_start_time = times[i]
        # Add the last segment
        segments.append(tgio.Interval(current_start_time, times[-1] + (hop_duration_ms / 1000), current_label))

    # --- Export to Praat TextGrid ---
    tg = tgio.Textgrid()
    tier_name = "Guessed_Phonemes"
    tg.addTier(tgio.IntervalTier(tier_name, [], 0, sound_parsel.duration))
    for seg in segments:
        tg.addInterval(tier_name, seg.start, seg.end, seg.label)

    output_textgrid_path = os.path.splitext(audio_path)[0] + ".TextGrid"
    tg.save(output_textgrid_path)
    print(f"Initial segmentation saved to {output_textgrid_path}")
    print("Please open this TextGrid in Praat for manual review and correction.")
    return output_textgrid_path

# Example Usage (replace 'your_audio.wav' with your actual file):
# Make sure you have a .wav file to test with.
# For example, record yourself saying "ba", "ab", "aba" and save as 'test_speech.wav'.
# audio_file_path = "test_speech.wav"
# if not os.path.exists(audio_file_path):
#     print(f"Error: {audio_file_path} not found. Please create or specify a valid audio file.")
# else:
#     generated_textgrid_path = guess_phonetic_segments(audio_file_path, n_clusters=5)
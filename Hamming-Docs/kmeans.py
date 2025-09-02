import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

def analyze_with_kmeans(audio_path, n_clusters=3):
    # Check if this is an ISO file (consonant + vowel sequence)
    filename = audio_path.split('\\')[-1].lower()
    is_iso_file = filename.startswith('iso')
    if is_iso_file:
        n_clusters = 2  # Just consonant and vowel
    """
    Use K-means clustering to segment audio into consonant ('b' sound) and vowel ('a' sound)
    """
    print(f"Analyzing: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Calculate frame parameters
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop
    
    # Extract features
    print("Extracting features...")
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Time axis
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Combine features into feature matrix
    features = np.vstack([
        rms,
        zcr,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_flatness,
        mfcc[:5]  # Use first 5 MFCCs
    ]).T
    
    # Handle any NaN values
    features = np.nan_to_num(features, nan=0.0)
    
    print(f"Feature matrix shape: {features.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply K-means clustering
    print(f"Applying K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    print(f"\n--- Cluster Analysis ---")
    cluster_info = {}
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            # Calculate average features for this cluster
            avg_rms = np.mean(rms[cluster_indices])
            avg_zcr = np.mean(zcr[cluster_indices])
            avg_centroid = np.mean(spectral_centroid[cluster_indices])
            avg_bandwidth = np.mean(spectral_bandwidth[cluster_indices])
            
            # Calculate time statistics
            cluster_times = times[cluster_indices]
            total_duration = len(cluster_indices) * (hop_length / sr)
            avg_time_position = np.mean(cluster_times)
            
            cluster_info[i] = {
                'avg_rms': avg_rms,
                'avg_zcr': avg_zcr,
                'avg_centroid': avg_centroid,
                'avg_bandwidth': avg_bandwidth,
                'total_duration': total_duration,
                'avg_time_position': avg_time_position,
                'frame_count': len(cluster_indices)
            }
            
            print(f"Cluster {i}:")
            print(f"  Frame count: {len(cluster_indices)}")
            print(f"  Total duration: {total_duration:.3f}s")
            print(f"  Avg time position: {avg_time_position:.3f}s")
            print(f"  Avg RMS: {avg_rms:.4f}")
            print(f"  Avg ZCR: {avg_zcr:.4f}")
            print(f"  Avg Centroid: {avg_centroid:.1f} Hz")
            print(f"  Avg Bandwidth: {avg_bandwidth:.1f} Hz")
    
    # Classify clusters based on acoustic characteristics
    cluster_classifications = classify_clusters(cluster_info, is_iso_file)
    
    print(f"\n--- Cluster Classifications ---")
    for cluster_id, classification in cluster_classifications.items():
        info = cluster_info[cluster_id]
        print(f"Cluster {cluster_id}: {classification['label']} (confidence: {classification['confidence']:.1f}%)")
        print(f"  Reasoning: {classification['reasoning']}")
    
    # Find time segments for each sound type
    segments = find_segments_by_type(cluster_labels, times, cluster_classifications, hop_length, sr)
    
    # Plot results
    plot_kmeans_results(y, times, rms, zcr, spectral_centroid, cluster_labels, 
                       cluster_classifications, segments, audio_path, duration)
    
    return segments, cluster_classifications

def classify_clusters(cluster_info, is_iso_file=False):
    """
    Identify clusters - for ISO files, classify as consonant and vowel based on timing and energy
    """
    classifications = {}
    
    if is_iso_file and len(cluster_info) == 2:
        # For ISO files with 2 clusters: consonant precedes vowel
        # Find cluster with highest RMS (vowel) and earliest timing (consonant)
        max_rms = 0
        vowel_cluster_id = None
        min_time = float('inf')
        consonant_cluster_id = None
        
        for cluster_id, info in cluster_info.items():
            if info['avg_rms'] > max_rms:
                max_rms = info['avg_rms']
                vowel_cluster_id = cluster_id
            if info['avg_time_position'] < min_time:
                min_time = info['avg_time_position']
                consonant_cluster_id = cluster_id
        
        for cluster_id, info in cluster_info.items():
            rms = info['avg_rms']
            zcr = info['avg_zcr']
            centroid = info['avg_centroid']
            avg_time = info['avg_time_position']
            
            if cluster_id == vowel_cluster_id:
                classifications[cluster_id] = {
                    'label': 'Vowel',
                    'confidence': 100,
                    'reasoning': f'Highest RMS energy ({rms:.4f}) indicates vowel',
                    'group': 'vowel'
                }
            else:  # Must be consonant
                classifications[cluster_id] = {
                    'label': 'Consonant',
                    'confidence': 100,
                    'reasoning': f'Earlier timing ({avg_time:.3f}s) indicates consonant',
                    'group': 'consonant'
                }
    else:
        # Original 3-cluster logic
        max_rms = 0
        vowel_cluster_id = None
        for cluster_id, info in cluster_info.items():
            if info['avg_rms'] > max_rms:
                max_rms = info['avg_rms']
                vowel_cluster_id = cluster_id
        
        for cluster_id, info in cluster_info.items():
            rms = info['avg_rms']
            zcr = info['avg_zcr']
            centroid = info['avg_centroid']
            bandwidth = info['avg_bandwidth']
            
            if cluster_id == vowel_cluster_id:
                classifications[cluster_id] = {
                    'label': 'Vowel',
                    'confidence': 100,
                    'reasoning': f'Highest RMS energy ({rms:.4f}) indicates vowel',
                    'group': 'vowel'
                }
            elif cluster_id == 0:
                classifications[cluster_id] = {
                    'label': 'Vowel Tail',
                    'confidence': 100,
                    'reasoning': f'Low energy tail region ({rms:.4f})',
                    'group': 'vowel_tail'
                }
            elif cluster_id == 1:
                classifications[cluster_id] = {
                    'label': 'Consonant',
                    'confidence': 100,
                    'reasoning': f'High ZCR ({zcr:.4f}) indicates consonant',
                    'group': 'consonant'
                }
            else:
                classifications[cluster_id] = {
                    'label': f'Cluster {cluster_id}',
                    'confidence': 100,
                    'reasoning': f'RMS: {rms:.4f}, ZCR: {zcr:.4f}, Centroid: {centroid:.0f} Hz',
                    'group': f'cluster_{cluster_id}'
                }
    
    return classifications

def find_segments_by_type(cluster_labels, times, classifications, hop_length, sr):
    """
    Find segments dynamically based on cluster transitions without timing assumptions
    """
    segments = {'consonant': [], 'vowel': [], 'vowel_tail': [], 'cluster_0': [], 'cluster_1': [], 'cluster_2': []}
    
    # Find cluster groups
    cluster_groups = {}
    for cluster_id, classification in classifications.items():
        cluster_groups[cluster_id] = classification.get('group', 'unknown')
    
    # Find continuous segments of each type
    current_group = None
    segment_start = None
    
    for i, cluster_id in enumerate(cluster_labels):
        frame_group = cluster_groups.get(cluster_id, 'unknown')
        
        if frame_group != current_group:
            # End previous segment if it exists
            if current_group is not None and segment_start is not None:
                if current_group in segments:
                    segments[current_group].append((times[segment_start], times[i-1]))
            
            # Start new segment
            current_group = frame_group
            segment_start = i
    
    # Handle final segment
    if current_group is not None and segment_start is not None:
        if current_group in segments:
            segments[current_group].append((times[segment_start], times[-1]))
    
    # Merge adjacent segments of the same type that are very close
    for sound_type in segments:
        if len(segments[sound_type]) > 1:
            merged = []
            current_seg = segments[sound_type][0]
            
            for next_seg in segments[sound_type][1:]:
                # If gap between segments is less than 50ms, merge them
                if next_seg[0] - current_seg[1] < 0.05:
                    current_seg = (current_seg[0], next_seg[1])
                else:
                    merged.append(current_seg)
                    current_seg = next_seg
            
            merged.append(current_seg)
            segments[sound_type] = merged
    
    return segments

def plot_kmeans_results(y, times, rms, zcr, spectral_centroid, cluster_labels, 
                       classifications, segments, audio_path, duration):
    """
    Plot the K-means clustering results
    """
    plt.figure(figsize=(15, 10))
    
    # Waveform with cluster coloring
    plt.subplot(4, 1, 1)
    plt.plot(np.linspace(0, duration, len(y)), y, alpha=0.7, color='gray')
    plt.title('Waveform with K-means Clustering')
    plt.ylabel('Amplitude')
    
    # Add colored regions for segments
    colors = {'consonant': 'red', 'vowel': 'green', 'vowel_tail': 'lightgreen', 'cluster_0': 'blue', 'cluster_1': 'orange', 'cluster_2': 'purple'}
    for cluster_group, segs in segments.items():
        for start, end in segs:
            plt.axvspan(start, end, alpha=0.3, color=colors.get(cluster_group, 'gray'), 
                       label=f'{cluster_group}' if cluster_group not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.legend()
    
    # RMS Energy with clusters
    plt.subplot(4, 1, 2)
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label_name = classifications[cluster_id]['label']
        plt.scatter(times[mask], rms[mask], c=[cluster_colors[i]], 
                   label=f'Cluster {cluster_id}: {label_name}', alpha=0.7, s=10)
    
    plt.plot(times, rms, color='black', alpha=0.3, linewidth=0.5)
    plt.title('RMS Energy by Cluster')
    plt.ylabel('Energy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Zero Crossing Rate
    plt.subplot(4, 1, 3)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(times[mask], zcr[mask], c=[cluster_colors[i]], alpha=0.7, s=10)
    
    plt.plot(times, zcr, color='black', alpha=0.3, linewidth=0.5)
    plt.title('Zero Crossing Rate by Cluster')
    plt.ylabel('ZCR')
    
    # Spectral Centroid
    plt.subplot(4, 1, 4)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(times[mask], spectral_centroid[mask], c=[cluster_colors[i]], alpha=0.7, s=10)
    
    plt.plot(times, spectral_centroid, color='black', alpha=0.3, linewidth=0.5)
    plt.title('Spectral Centroid by Cluster')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plot_path = audio_path.replace('.wav', '_kmeans_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"K-means analysis plot saved to: {plot_path}")

if __name__ == "__main__":
    audio_file = r"D:\Github\phone-cleaner\Hamming-Docs\iso_[f]_f_unvoiced labial non sibilant fricative.wav"
    
    try:
        segments, classifications = analyze_with_kmeans(audio_file, n_clusters=3)
        
        print(f"\n" + "="*60)
        print(f"K-MEANS CLUSTERING RESULTS")
        print(f"="*60)
        
        for cluster_group in ['consonant', 'vowel', 'vowel_tail', 'cluster_0', 'cluster_1', 'cluster_2']:
            if cluster_group in segments and segments[cluster_group]:
                print(f"\n{cluster_group.upper()} segments:")
                for i, (start, end) in enumerate(segments[cluster_group]):
                    duration = end - start
                    print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
            else:
                print(f"\n{cluster_group.upper()}: No segments detected")
        
        print(f"\nAnalysis complete. Check the generated plot for visual verification.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the audio file exists and required libraries are installed.")
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

def analyze_with_kmeans(audio_path, n_clusters=3):
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
    cluster_classifications = classify_clusters(cluster_info)
    
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

def classify_clusters(cluster_info):
    """
    Classify clusters as silence, consonant, or vowel based on acoustic features
    """
    classifications = {}
    
    for i, (cluster_id, info) in enumerate(cluster_info.items()):
        rms = info['avg_rms']
        zcr = info['avg_zcr']
        centroid = info['avg_centroid']
        bandwidth = info['avg_bandwidth']
        duration = info['total_duration']
        avg_time = info['avg_time_position']
        
        confidence = 50  # Base confidence
        
        # Silence: lowest energy
        if rms < 0.04:
            confidence += 30
            classifications[cluster_id] = {
                'label': 'Silence',
                'confidence': min(confidence, 95),
                'reasoning': 'Very low RMS energy indicates silence',
                'group': 'silence'
            }
        
        # Consonant: early timing - reduced range to not extend too far
        elif (avg_time < 0.25 and rms > 0.04):
            confidence += 25
            if avg_time < 0.15:
                confidence += 15  # Higher confidence for very early timing
            if rms > 0.1:
                confidence += 10  # Higher confidence for higher energy
            classifications[cluster_id] = {
                'label': 'Consonant (including transitions)',
                'confidence': min(confidence, 90),
                'reasoning': f'Early timing ({avg_time:.3f}s), consonant region',
                'group': 'consonant'
            }
        
        # Vowel: mid to later timing with vowel-like characteristics
        elif (avg_time >= 0.25 or (rms > 0.08 and zcr < 0.08)):
            confidence += 35
            if 600 < centroid < 1500 and zcr < 0.08:
                confidence += 15
            classifications[cluster_id] = {
                'label': 'Vowel',
                'confidence': min(confidence, 95),
                'reasoning': f'Mid-to-later timing or vowel-like features (centroid: {centroid:.0f} Hz)',
                'group': 'vowel'
            }
        
        # Fallback classifications
        elif rms < 0.03:
            classifications[cluster_id] = {
                'label': 'Silence',
                'confidence': 60,
                'reasoning': 'Low energy suggests silence',
                'group': 'silence'
            }
        else:
            # Default based on timing - more conservative consonant boundary
            if avg_time < 0.2:
                classifications[cluster_id] = {
                    'label': 'Consonant',
                    'confidence': 50,
                    'reasoning': 'Early timing suggests consonant',
                    'group': 'consonant'
                }
            else:
                classifications[cluster_id] = {
                    'label': 'Vowel',
                    'confidence': 50,
                    'reasoning': 'Later timing suggests vowel',
                    'group': 'vowel'
                }
    
    return classifications

def find_segments_by_type(cluster_labels, times, classifications, hop_length, sr):
    """
    Find segments for silence, consonant, and vowel with conservative consonant boundary
    """
    segments = {'silence': [], 'consonant': [], 'vowel': []}
    
    # Find cluster groups
    silence_clusters = []
    consonant_clusters = []
    vowel_clusters = []
    
    for cluster_id, classification in classifications.items():
        group = classification.get('group', 'unknown')
        if group == 'silence':
            silence_clusters.append(cluster_id)
        elif group == 'consonant':
            consonant_clusters.append(cluster_id)
        elif group == 'vowel':
            vowel_clusters.append(cluster_id)
    
    # Find conservative consonant end time (earlier boundary)
    consonant_end_time = 0
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id in consonant_clusters:
            consonant_end_time = max(consonant_end_time, times[i])
    
    # Find vowel start time
    vowel_start_time = times[-1]
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id in vowel_clusters:
            vowel_start_time = min(vowel_start_time, times[i])
    
    # Set more conservative boundary - favor shorter consonant region
    if consonant_end_time > 0 and vowel_start_time < times[-1]:
        # Use 1/3 weighting toward consonant end, 2/3 toward vowel start
        boundary_time = consonant_end_time * 0.3 + vowel_start_time * 0.7
    else:
        # Default conservative boundary at 1/4 of the audio
        boundary_time = times[len(times)//4]
    
    # Handle silence segments
    silence_start = None
    silence_end = None
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id in silence_clusters:
            if silence_start is None:
                silence_start = times[i]
            silence_end = times[i]
        else:
            if silence_start is not None:
                segments['silence'].append((silence_start, silence_end))
                silence_start = None
    
    # Add final silence segment if exists
    if silence_start is not None:
        segments['silence'].append((silence_start, silence_end))
    
    # Create consonant and vowel segments with conservative boundary
    segments['consonant'].append((times[0], boundary_time))
    segments['vowel'].append((boundary_time, times[-1]))
    
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
    colors = {'silence': 'lightblue', 'consonant': 'red', 'vowel': 'green'}
    for sound_group, segs in segments.items():
        for start, end in segs:
            plt.axvspan(start, end, alpha=0.3, color=colors.get(sound_group, 'gray'), 
                       label=f'{sound_group}' if sound_group not in plt.gca().get_legend_handles_labels()[1] else "")
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
    audio_file = r"D:\Github\phone-cleaner\Hamming-Docs\iso_[b]_b_voiced unaspirated bilabial stop.wav"
    
    try:
        segments, classifications = analyze_with_kmeans(audio_file, n_clusters=3)
        
        print(f"\n" + "="*60)
        print(f"K-MEANS CLUSTERING RESULTS")
        print(f"="*60)
        
        for sound_group in ['silence', 'consonant', 'vowel']:
            if sound_group in segments and segments[sound_group]:
                print(f"\n{sound_group.upper()} segments:")
                for i, (start, end) in enumerate(segments[sound_group]):
                    duration = end - start
                    print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
            else:
                print(f"\n{sound_group.upper()}: No clear segments detected")
        
        print(f"\nAnalysis complete. Check the generated plot for visual verification.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the audio file exists and required libraries are installed.")
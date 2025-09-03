from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
import re
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


AudioSegment.converter = "D:/Github/phone-cleaner/bin/ffmpeg.exe"
AudioSegment.ffprobe = "D:/Github/phone-cleaner/bin/ffprobe.exe"
input_folder = "./phoneme-Samples/Glossika/wav-no-music/"
output_folder = "./phoneme-Samples/Glossika/tight-snips/"

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

volume_threshold = -45  # dB
lead_time = 200  # milliseconds
follow_time = 200  # milliseconds

def find_name(input_string):
    # Use a regular expression to find the content within the first pair of square brackets
    match = re.search(r'\] ([^\]]+)\.', input_string)
    
    # If a match is found, trim leading and trailing spaces
    if match:
        return match.group(1).strip()
    else:
        return None
    
def find_bracket_contents(input_string):
    # Use a regular expression to find the content within the first pair of square brackets
    match = re.search(r'\[([^\]]+)\]', input_string)
    
    # If a match is found, trim leading and trailing spaces
    if match:
        return match.group(1).strip()
    else:
        return None

def find_segments_by_cluster(cluster_labels, times):
    """
    Find continuous segments for each cluster
    """
    segments_by_cluster = {}
    
    # Find continuous segments of each cluster
    current_cluster = None
    segment_start = None
    
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id != current_cluster:
            # End previous segment if it exists
            if current_cluster is not None and segment_start is not None:
                if current_cluster not in segments_by_cluster:
                    segments_by_cluster[current_cluster] = []
                segments_by_cluster[current_cluster].append((times[segment_start], times[i-1]))
            
            # Start new segment
            current_cluster = cluster_id
            segment_start = i
    
    # Handle final segment
    if current_cluster is not None and segment_start is not None:
        if current_cluster not in segments_by_cluster:
            segments_by_cluster[current_cluster] = []
        segments_by_cluster[current_cluster].append((times[segment_start], times[-1]))
    
    # Merge adjacent segments of the same cluster that are very close
    for cluster_id in segments_by_cluster:
        if len(segments_by_cluster[cluster_id]) > 1:
            merged = []
            current_seg = segments_by_cluster[cluster_id][0]
            
            for next_seg in segments_by_cluster[cluster_id][1:]:
                # If gap between segments is less than 50ms, merge them
                if next_seg[0] - current_seg[1] < 0.05:
                    current_seg = (current_seg[0], next_seg[1])
                else:
                    merged.append(current_seg)
                    current_seg = next_seg
            
            merged.append(current_seg)
            segments_by_cluster[cluster_id] = merged
    
    return segments_by_cluster

def plot_speech_segments(audio, nonsilent_ranges, filename, output_folder, plot_single_region=None):
    """
    Plot waveform with detected speech segments highlighted
    
    Args:
        audio: AudioSegment object
        nonsilent_ranges: List of (start, end) tuples in milliseconds
        filename: Original audio filename
        output_folder: Directory to save plots
        plot_single_region: If specified (int), plot only this region index. If None, plot all regions.
    """
    # Convert audio to numpy array for plotting
    audio_samples = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        audio_samples = audio_samples.reshape((-1, 2))
        audio_samples = audio_samples.mean(axis=1)  # Convert to mono
    
    if plot_single_region is not None and 0 <= plot_single_region < len(nonsilent_ranges):
        # Plot single region
        start_ms, end_ms = nonsilent_ranges[plot_single_region]
        
        # Convert to sample indices
        sample_rate = audio.frame_rate
        start_sample = int(start_ms * sample_rate / 1000)
        end_sample = int(end_ms * sample_rate / 1000)
        
        # Extract region samples with some padding
        padding_ms = 500  # 500ms padding on each side
        padding_samples = int(padding_ms * sample_rate / 1000)
        
        plot_start = max(0, start_sample - padding_samples)
        plot_end = min(len(audio_samples), end_sample + padding_samples)
        
        region_samples = audio_samples[plot_start:plot_end]
        region_time_axis = np.linspace(plot_start * 1000 / sample_rate, 
                                     plot_end * 1000 / sample_rate, 
                                     len(region_samples))
        
        plt.figure(figsize=(12, 6))
        plt.plot(region_time_axis, region_samples, alpha=0.7, color='gray', linewidth=0.5)
        plt.axvspan(start_ms, end_ms, alpha=0.3, color='red', 
                   label=f'Segment {plot_single_region}: {end_ms-start_ms}ms')
        plt.title(f'Speech Segment {plot_single_region}: {filename}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        base_filename = os.path.splitext(filename)[0]
        plot_path = os.path.join(output_folder, f"{base_filename}_segment_{plot_single_region}.png")
        
    else:
        # Plot all regions on full waveform
        plt.figure(figsize=(15, 6))
        
        # Create time axis in milliseconds
        time_axis = np.linspace(0, len(audio), len(audio_samples))
        
        # Plot the waveform
        plt.plot(time_axis, audio_samples, alpha=0.7, color='gray', linewidth=0.5)
        plt.title(f'Detected Speech Segments: {filename}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        
        # Highlight each nonsilent segment
        colors = plt.cm.tab10(np.linspace(0, 1, len(nonsilent_ranges)))
        for i, (start, end) in enumerate(nonsilent_ranges):
            plt.axvspan(start, end, alpha=0.3, color=colors[i], 
                       label=f'Segment {i}: {end-start}ms')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        base_filename = os.path.splitext(filename)[0]
        plot_path = os.path.join(output_folder, f"{base_filename}_segments.png")
    
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()  # Display the plot
    plt.close()
    print(f"Segment plot saved: {plot_path}")
    
    return plot_path

def plot_kmeans_results(y, times, rms, zcr, spectral_centroid, cluster_labels, 
                       cluster_classifications, segment_type, audio_filename, duration, segment_idx):
    """
    Plot the K-means clustering results for a segment with cluster highlighting
    """
    plt.figure(figsize=(15, 10))
    
    # Find segments by cluster for coloring
    segments_by_cluster = find_segments_by_cluster(cluster_labels, times)
    
    # Waveform with cluster coloring
    plt.subplot(4, 1, 1)
    plt.plot(np.linspace(0, duration, len(y)), y, alpha=0.7, color='gray')
    plt.title(f'Segment {segment_idx} - Detected as: {segment_type.upper()}')
    plt.ylabel('Amplitude')
    
    # Add colored regions for each cluster
    unique_clusters = np.unique(cluster_labels)
    cluster_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster_id: cluster_colors[i] for i, cluster_id in enumerate(unique_clusters)}
    
    for cluster_id, segments in segments_by_cluster.items():
        color = color_map[cluster_id]
        label_name = cluster_classifications[cluster_id]['label']
        
        for i, (start, end) in enumerate(segments):
            # Only add label for first segment of each cluster to avoid duplicate legends
            label = f'Cluster {cluster_id}: {label_name}' if i == 0 else ""
            plt.axvspan(start, end, alpha=0.3, color=color, label=label)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # RMS Energy with clusters
    plt.subplot(4, 1, 2)
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        label_name = cluster_classifications[cluster_id]['label']
        plt.scatter(times[mask], rms[mask], c=[cluster_colors[i]], 
                   label=f'Cluster {cluster_id}: {label_name}', alpha=0.7, s=15)
    
    plt.plot(times, rms, color='black', alpha=0.3, linewidth=0.5)
    plt.title('RMS Energy by Cluster')
    plt.ylabel('Energy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Zero Crossing Rate
    plt.subplot(4, 1, 3)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(times[mask], zcr[mask], c=[cluster_colors[i]], alpha=0.7, s=15)
    
    plt.plot(times, zcr, color='black', alpha=0.3, linewidth=0.5)
    plt.title('Zero Crossing Rate by Cluster')
    plt.ylabel('ZCR')
    
    # Spectral Centroid
    plt.subplot(4, 1, 4)
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(times[mask], spectral_centroid[mask], c=[cluster_colors[i]], alpha=0.7, s=15)
    
    plt.plot(times, spectral_centroid, color='black', alpha=0.3, linewidth=0.5)
    plt.title('Spectral Centroid by Cluster')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    
    plt.tight_layout()
    
    # Save plot
    base_filename = os.path.splitext(audio_filename)[0]
    plot_path = os.path.join(output_folder, f"{base_filename}_segment_{segment_idx}_{segment_type}_kmeans.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"K-means plot saved: {plot_path}")

def analyze_segment_with_kmeans(audio_data, sr, audio_filename="", segment_idx=0, save_plot=True):
    """
    Analyze audio segment using kmeans to determine if it's isolated C, CV, VCV, or VC
    """
    # Play the audio segment for debugging
    import sounddevice as sd
    import time
    print(f"\n--- Playing segment {segment_idx} from {audio_filename} ---")
    sd.play(audio_data, sr)
    # Wait for playback to complete
    time.sleep(len(audio_data) / sr)
    sd.stop()
    
    # Calculate frame parameters
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop
    
    # Extract features
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Time axis
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    duration = len(audio_data) / sr
    
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

    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Check segment length - skip if too short
    duration_ms = len(audio_data) / sr * 1000
    if duration_ms < 100:
        print(f"Segment too short ({duration_ms:.1f}ms), skipping analysis")
        return "iso"
        
    # Use 2 clusters to separate consonants and vowels (ignore silence)
    n_clusters = 2
    if len(features) < 2:
        return "iso"  # Too short to analyze
        
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters and classify them as SILENCE, CONSONANT, or VOWEL
    cluster_info = {}
    print(f"\n=== CLUSTER ANALYSIS FOR SEGMENT {segment_idx} ===")
    
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) > 0:
            avg_rms = np.mean(rms[cluster_indices])
            avg_zcr = np.mean(zcr[cluster_indices])
            avg_centroid = np.mean(spectral_centroid[cluster_indices])
            avg_time_position = np.mean(times[cluster_indices])
            
            cluster_info[i] = {
                'avg_rms': avg_rms,
                'avg_zcr': avg_zcr,
                'avg_centroid': avg_centroid,
                'avg_time_position': avg_time_position,
                'frame_count': len(cluster_indices)
            }
            
            print(f"Cluster {i}:")
            print(f"  RMS Energy: {avg_rms:.4f}")
            print(f"  Zero Crossing Rate: {avg_zcr:.4f}")
            print(f"  Spectral Centroid: {avg_centroid:.1f} Hz")
            print(f"  Average Time: {avg_time_position:.3f}s")
            print(f"  Frame Count: {len(cluster_indices)} ({len(cluster_indices)/len(times)*100:.1f}% of segment)")
    
    # Classify clusters based on acoustic properties (2 clusters: consonant and vowel)
    # Sort clusters by RMS to identify vowels (highest) and consonants (lowest)
    clusters_by_rms = sorted(cluster_info.items(), key=lambda x: x[1]['avg_rms'])
    
    consonant_cluster = None  
    vowel_cluster = None
    
    # Lowest RMS = consonant
    consonant_cluster = clusters_by_rms[0][0]
    consonant_info = clusters_by_rms[0][1]
    
    # Highest RMS = vowel
    vowel_cluster = clusters_by_rms[-1][0]
    vowel_info = clusters_by_rms[-1][1]
    
    # Assign cluster types
    cluster_types = {}
    cluster_types[consonant_cluster] = 'CONSONANT'
    cluster_types[vowel_cluster] = 'VOWEL'
    
    print(f"\n=== CLUSTER CLASSIFICATION ===")
    for cluster_id, cluster_type in cluster_types.items():
        info = cluster_info[cluster_id]
        print(f"Cluster {cluster_id}: {cluster_type}")
        print(f"  RMS: {info['avg_rms']:.4f}, ZCR: {info['avg_zcr']:.4f}")
        print(f"  Time: {info['avg_time_position']:.3f}s, Frames: {info['frame_count']}")
        print()
    
    # Create cluster classifications for plotting
    cluster_classifications = {}
    for cluster_id, info in cluster_info.items():
        rms_val = info['avg_rms']
        zcr_val = info['avg_zcr']
        centroid_val = info['avg_centroid']
        
        cluster_classifications[cluster_id] = {
            'label': f'RMS:{rms_val:.3f} ZCR:{zcr_val:.3f}',
            'group': f'cluster_{cluster_id}'
        }
    
    # Classify segment based on temporal order of silence, consonant, and vowel clusters
    segment_type = "unk"  # Default
    print(f"=== CLASSIFICATION LOGIC ===")
    
    # Get temporal positions of each cluster type
    cluster_positions = []
    for cluster_id, cluster_type in cluster_types.items():
        time_pos = cluster_info[cluster_id]['avg_time_position']
        cluster_positions.append((time_pos, cluster_type, cluster_id))
    
    # Sort by time
    cluster_positions.sort(key=lambda x: x[0])
    
    print("Temporal sequence:")
    temporal_sequence = []
    for time_pos, cluster_type, cluster_id in cluster_positions:
        print(f"  {time_pos:.3f}s: {cluster_type} (Cluster {cluster_id})")
        temporal_sequence.append(cluster_type)
    
    # Analyze temporal distribution to detect VCV patterns
    # Get segments for each cluster type to see if they appear in multiple regions
    consonant_segments = find_segments_by_cluster(cluster_labels, times)
    
    consonant_regions = consonant_segments.get(consonant_cluster, [])
    vowel_regions = consonant_segments.get(vowel_cluster, [])
    
    print(f"Consonant regions: {len(consonant_regions)} segments")
    for i, (start, end) in enumerate(consonant_regions):
        print(f"  C{i}: {start:.3f}s - {end:.3f}s ({(end-start)*1000:.0f}ms)")
    
    print(f"Vowel regions: {len(vowel_regions)} segments") 
    for i, (start, end) in enumerate(vowel_regions):
        print(f"  V{i}: {start:.3f}s - {end:.3f}s ({(end-start)*1000:.0f}ms)")
    
    # Determine pattern based on temporal arrangement
    consonant_time = cluster_info[consonant_cluster]['avg_time_position']
    vowel_time = cluster_info[vowel_cluster]['avg_time_position']
    
    print(f"Overall consonant cluster time: {consonant_time:.3f}s")
    print(f"Overall vowel cluster time: {vowel_time:.3f}s")
    
    # Sort all regions by start time to get the first segment
    all_regions = []
    for start, end in consonant_regions:
        all_regions.append((start, end, 'C'))
    for start, end in vowel_regions:
        all_regions.append((start, end, 'V'))
    all_regions.sort(key=lambda x: x[0])
    
    if len(all_regions) > 0:
        first_segment_type = all_regions[0][2]
        print(f"First segment type: {first_segment_type}")
        
        # If first segment is consonant, assume CV pattern
        if first_segment_type == 'C':
            segment_type = "pre"  # CV pattern
            print("Classification: PRE (CV) - starts with consonant")
        else:
            # First segment is vowel, check for VCV or VC
            if len(vowel_regions) >= 2 and len(consonant_regions) >= 1:
                # Extract sequence pattern
                pattern = [region[2] for region in all_regions]
                pattern_str = ''.join(pattern)
                print(f"Temporal pattern: {pattern_str}")
                
                # Check for VCV pattern
                if 'VCV' in pattern_str or pattern_str.startswith('V') and pattern_str.endswith('V') and 'C' in pattern_str:
                    segment_type = "med"  # VCV pattern
                    print("Classification: MED (VCV) - vowel-consonant-vowel pattern detected")
                else:
                    segment_type = "post"  # VC pattern  
                    print("Classification: POST (VC) - vowel before consonant")
            else:
                segment_type = "post"  # VC pattern  
                print("Classification: POST (VC) - starts with vowel")
    else:
        segment_type = "unk"
        print("Classification: UNK - no regions found")
    
    # Generate plot if requested
    if save_plot and len(audio_filename) > 0:
        plot_kmeans_results(audio_data, times, rms, zcr, spectral_centroid, cluster_labels,
                           cluster_classifications, segment_type, audio_filename, duration, segment_idx)
    
    return segment_type


#Clipping Glossika with K-means Analysis
for filename in os.listdir(input_folder):
    
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_folder, filename)
        # Use Unicode string
        audio_path = audio_path
        
        
        print(audio_path)
         # Print the audio path to debug
        print("Processing file:", audio_path)
        
        # Convert the file to a standard format
        temp_audio_path = os.path.join(output_folder, "temp_output.wav")
        
        # Use subprocess for better Unicode handling and error capture
        import subprocess
        
        conversion_command = [
            "ffmpeg.exe",
            "-y",
            "-i", audio_path,
            "-acodec", "pcm_s16le", 
            "-ar", "44100",
            temp_audio_path
        ]

        print(f"Running ffmpeg conversion for: {filename}")
        try:
            result = subprocess.run(conversion_command, 
                                  capture_output=True, 
                                  text=True, 
                                  encoding='utf-8',
                                  timeout=30)
            
            if result.returncode != 0:
                print(f"FFmpeg error (return code {result.returncode}):")
                print(f"stderr: {result.stderr}")
                print(f"stdout: {result.stdout}")
                continue
            else:
                print("Conversion successful")
                
        except subprocess.TimeoutExpired:
            print("FFmpeg conversion timed out")
            continue
        except Exception as e:
            print(f"Error running ffmpeg: {e}")
            continue
        
        # Check if temp file was created
        if not os.path.exists(temp_audio_path):
            print(f"Error: temp file not created at {temp_audio_path}")
            continue
        
        try:
            # Load the converted audio file
            audio = AudioSegment.from_file(temp_audio_path, format="wav")
        except Exception as e:
            print("Error loading audio file:", e)
            continue
        
        # Detect nonsilent segments with less sensitive parameters
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=-45)
            
        phoneme = find_bracket_contents(filename)
        name = find_name(filename)
        
        # Display waveform with all detected segments highlighted
        #plot_speech_segments(audio, nonsilent_ranges, filename, output_folder, plot_single_region=None)
        #print(f"Found {len(nonsilent_ranges)} speech segments")
        
        # Wait for user to examine the plot before continuing
        #input("Press Enter to continue with segment analysis...")
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            # Plot this specific segment
            plot_speech_segments(audio, nonsilent_ranges, filename, output_folder, plot_single_region=i)
            
            clip_length = end - start
            start = max(0, start - lead_time)
            end = min(len(audio), end + follow_time)
            clip = audio[start:end]
            
            # Trim silence from beginning and end of the clip
            trimmed_clip = clip.strip_silence(silence_len=50, silence_thresh=-45, padding=25)
            
            # Use trimmed clip if it's not too short, otherwise keep original
            if len(trimmed_clip) > 100:  # Keep at least 100ms
                clip = trimmed_clip
                print(f"Trimmed silence: {len(audio[start:end])}ms -> {len(clip)}ms")
            else:
                print(f"Clip too short after trimming, keeping original: {len(clip)}ms")
            
            # Convert AudioSegment to numpy array for analysis
            audio_samples = np.array(clip.get_array_of_samples())
            if clip.channels == 2:
                audio_samples = audio_samples.reshape((-1, 2))
                audio_samples = audio_samples.mean(axis=1)  # Convert to mono
            
            # Convert to float and normalize
            audio_samples = audio_samples.astype(np.float32)
            if audio_samples.max() > 1.0:
                audio_samples = audio_samples / (2**15)  # Normalize 16-bit audio
            
            # Analyze with k-means to determine segment type (with plotting)
            try:
                segment_type = analyze_segment_with_kmeans(
                    audio_samples, 
                    clip.frame_rate, 
                    audio_filename=filename, 
                    segment_idx=i, 
                    save_plot=True
                )
                print(f"Segment {i}: Detected as {segment_type}")
            except Exception as e:
                print(f"Error analyzing segment {i}: {e}")
                segment_type = "unk"  # Default fallback
            
            # Generate filename based on acoustic analysis
            output_filename = f"{os.path.splitext(filename)[0]}_clip_{i}.wav"
            
            if segment_type == "iso":
                output_filename = f"iso_[{phoneme}]_{phoneme}_{name}-{i}.wav"
            elif segment_type == "pre":
                output_filename = f"pre_[{phoneme}]_{phoneme}ə_{name}-{i}.wav"
            elif segment_type == "med":
                output_filename = f"med_[{phoneme}]_ə{phoneme}ə_{name}-{i}.wav"
            elif segment_type == "post":
                output_filename = f"post_[{phoneme}]_ə{phoneme}_{name}-{i}.wav"
                
            output_path = os.path.join(output_folder, output_filename)
            
            # Only export the first 4 segments (as before)
            if 0 <= i <= 3:
                # Print the output path to debug
                print("Exporting clip to:", output_path)

                try:
                    # Export clip with Unicode handling
                    if ("_clean" in output_path):
                        output_path = output_path.replace("_clean", "")
                    clip.export(output_path, format="wav")
                    print(f"Successfully exported: {output_filename} (Type: {segment_type})")
                except Exception as e:
                    print("Error exporting file:", e)
                    continue

                
# Segment detection legend:
# iso: Isolated consonant/phoneme
# pre: Pre-vowel (CV pattern)  
# med: Medial (VCV pattern) - vowel-consonant-vowel
# post: Post-vowel (VC pattern)

# Each processed segment will generate:
# 1. Audio file with classified name (iso_, pre_, med_, or post_)
# 2. PNG plot showing kmeans analysis with waveform, RMS, ZCR, and spectral centroid
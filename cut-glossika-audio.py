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
lead_time = 100  # milliseconds
follow_time = 100  # milliseconds

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
    
    # Use 2 or 3 clusters based on segment length
    n_clusters = 2 #min(3, len(features))
    if n_clusters < 2:
        return "iso"
        
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Analyze clusters
    cluster_info = {}
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
    
    # Classify segment type based on clusters
    segment_type = "unk"  # Default
    
    if n_clusters == 2:
        # Find highest RMS (vowel) and earliest timing (consonant)
        max_rms = 0
        vowel_cluster = None
        min_time = float('inf')
        consonant_cluster = None
        
        for cluster_id, info in cluster_info.items():
            if info['avg_rms'] > max_rms:
                max_rms = info['avg_rms']
                vowel_cluster = cluster_id
            if info['avg_time_position'] < min_time:
                min_time = info['avg_time_position']
                consonant_cluster = cluster_id
        
        # Determine if it's CV or VC based on cluster sequence
        if vowel_cluster != consonant_cluster:
            # Check temporal order
            vowel_time = cluster_info[vowel_cluster]['avg_time_position']
            consonant_time = cluster_info[consonant_cluster]['avg_time_position']
            
            if consonant_time < vowel_time:
                segment_type = "pre"  # CV pattern
            else:
                segment_type = "post"  # VC pattern
        else:
            segment_type = "unk"  # Single dominant cluster type
            
    elif n_clusters == 3:
        # Three clusters - identify consonant and vowel clusters
        # Find the cluster with highest RMS as vowel
        max_rms = 0
        vowel_cluster = None
        for cluster_id, info in cluster_info.items():
            if info['avg_rms'] > max_rms:
                max_rms = info['avg_rms']
                vowel_cluster = cluster_id
        
        # Get temporal positions of all clusters
        cluster_times = [(cluster_id, info['avg_time_position']) for cluster_id, info in cluster_info.items()]
        cluster_times.sort(key=lambda x: x[1])  # Sort by time
        
        # Check if vowel is in the middle position (VCV pattern)
        middle_cluster = cluster_times[1][0]  # Cluster in middle position temporally
        
        if middle_cluster == vowel_cluster:
            # Vowel is in middle - this is VCV (vowel-consonant-vowel) pattern
            segment_type = "med"  # VCV pattern
        else:
            # Vowel is not in middle, check if it's at start or end
            first_cluster = cluster_times[0][0]
            if first_cluster == vowel_cluster:
                segment_type = "post"  # VCC or VC pattern - vowel at start
            else:
                segment_type = "pre"  # CCV or CV pattern - vowel at end
    
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
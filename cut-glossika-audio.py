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

def classify_cv_vc_vcv(audio_data, sr, min_vowel_ms=40, min_consonant_ms=30, plot_debug=False):
    """
    Classifies an audio segment as CV, VC, or VCV using F1 and F2 formants.
    Returns: "cv", "vc", "vcv", or "unk"
    """
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    
    # Get RMS first to establish the correct number of frames
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    n_frames = len(rms)
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
    
    # Extract F1 and F2 formants using LPC analysis
    def extract_formants(audio, sr, frame_length, hop_length, n_frames):
        """Extract F1 and F2 formants using Linear Predictive Coding"""
        import scipy.signal
        
        # Pre-emphasize audio
        audio_preemph = scipy.signal.lfilter([1, -0.97], 1, audio)
        
        f1_values = []
        f2_values = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = min(start + frame_length, len(audio_preemph))
            frame = audio_preemph[start:end]
            
            if len(frame) < frame_length // 2:
                # Pad short frames
                frame = np.pad(frame, (0, frame_length - len(frame)))
            
            # Apply window
            windowed = frame * np.hanning(len(frame))
            
            try:
                # LPC analysis (order 12 works well for formants)
                lpc_order = min(12, len(windowed) // 4)
                a = librosa.lpc(windowed, order=lpc_order)
                
                # Find roots and convert to frequencies
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]  # Keep positive imaginary parts
                
                # Convert to frequencies
                freqs = np.angle(roots) * sr / (2 * np.pi)
                freqs = freqs[freqs > 0]  # Only positive frequencies
                freqs = np.sort(freqs)
                
                # Typically F1 is 200-800Hz, F2 is 800-2500Hz
                f1_candidates = freqs[(freqs >= 200) & (freqs <= 800)]
                f2_candidates = freqs[(freqs >= 800) & (freqs <= 2500)]
                
                f1 = f1_candidates[0] if len(f1_candidates) > 0 else 0
                f2 = f2_candidates[0] if len(f2_candidates) > 0 else 0
                
            except:
                f1, f2 = 0, 0
            
            f1_values.append(f1)
            f2_values.append(f2)
        
        return np.array(f1_values), np.array(f2_values)
    
    # Extract formants
    f1, f2 = extract_formants(audio_data, sr, frame_length, hop_length, n_frames)
    
    print(f"F1 range: {np.min(f1[f1>0]):.0f}-{np.max(f1):.0f} Hz")
    print(f"F2 range: {np.min(f2[f2>0]):.0f}-{np.max(f2):.0f} Hz")
    
    # Vowel detection based on formants
    # Vowels have clear F1 and F2 formants and reasonable RMS
    has_formants = (f1 > 200) & (f1 < 800) & (f2 > 800) & (f2 < 2500)
    has_energy = rms > np.percentile(rms, 30)  # Some minimum energy
    
    is_vowel = has_formants & has_energy
    
    print(f"Frames with clear formants: {np.sum(has_formants)}/{len(has_formants)} ({np.sum(has_formants)/len(has_formants)*100:.1f}%)")
    print(f"Frames with energy: {np.sum(has_energy)}/{len(has_energy)} ({np.sum(has_energy)/len(has_energy)*100:.1f}%)")
    print(f"Vowel frames: {np.sum(is_vowel)}/{len(is_vowel)} ({np.sum(is_vowel)/len(is_vowel)*100:.1f}%)")

    # Find contiguous vowel regions
    vowel_regions = []
    in_vowel = False
    start_idx = 0
    for i, v in enumerate(is_vowel):
        if v and not in_vowel:
            in_vowel = True
            start_idx = i
        elif not v and in_vowel:
            in_vowel = False
            end_idx = i - 1
            duration = (times[end_idx] - times[start_idx]) * 1000
            if duration >= min_vowel_ms:
                vowel_regions.append((start_idx, end_idx))
    # Handle trailing vowel
    if in_vowel:
        end_idx = len(is_vowel) - 1
        duration = (times[end_idx] - times[start_idx]) * 1000
        if duration >= min_vowel_ms:
            vowel_regions.append((start_idx, end_idx))

    # Merge close vowel regions (<30ms apart)
    merged_vowel_regions = []
    for region in vowel_regions:
        if not merged_vowel_regions:
            merged_vowel_regions.append(region)
        else:
            prev = merged_vowel_regions[-1]
            gap = (times[region[0]] - times[prev[1]]) * 1000
            if gap < 30:
                merged_vowel_regions[-1] = (prev[0], region[1])
            else:
                merged_vowel_regions.append(region)
    vowel_regions = merged_vowel_regions

    # Guarantee at least one vowel region
    if len(vowel_regions) == 0:
        # Pick the frame with highest RMS as a "vowel"
        max_rms_idx = np.argmax(rms)
        vowel_regions = [(max(0, max_rms_idx-1), min(len(rms)-1, max_rms_idx+1))]

    # Now classify
    if len(vowel_regions) == 1:
        v_start, v_end = vowel_regions[0]
        consonant_before = v_start > 0 and (times[v_start] - times[0]) * 1000 >= min_consonant_ms
        consonant_after = v_end < len(times)-1 and (times[-1] - times[v_end]) * 1000 >= min_consonant_ms
        if consonant_before and not consonant_after:
            return "cv"
        elif consonant_after and not consonant_before:
            return "vc"
        elif consonant_before and consonant_after:
            return "cv"  # ambiguous, could be CVC
        else:
            return "unk"
    elif len(vowel_regions) == 2:
        gap = (times[vowel_regions[1][0]] - times[vowel_regions[0][1]]) * 1000
        if gap >= min_consonant_ms:
            return "vcv"
        else:
            return "unk"
    else:
        return "unk"


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
            
            # Analyze with determine segment type (with plotting)
            try:
                segment_type = classify_cv_vc_vcv(audio_samples, clip.frame_rate)
                print(f"Segment {i}: Detected as {segment_type}")
            except Exception as e:
                print(f"Error analyzing segment {i}: {e}")
                segment_type = "unk"  # Default fallback
            
            # Generate filename based on acoustic analysis
            output_filename = f"{os.path.splitext(filename)[0]}_clip_{i}.wav"
            
            if segment_type == "pre":
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
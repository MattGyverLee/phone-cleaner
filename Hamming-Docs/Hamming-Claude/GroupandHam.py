import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
import matplotlib.pyplot as plt

class ConsonantDetector:
    def __init__(self, frame_length=1024, hop_length=512, n_clusters=3):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        
    def extract_features(self, audio, sr):
        """Extract multiple audio features for clustering"""
        # Short-time energy (RMS)
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                  hop_length=self.hop_length)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length,
                                                hop_length=self.hop_length)[0]
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr,
                                                             hop_length=self.hop_length)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr,
                                                           hop_length=self.hop_length)[0]
        
        # MFCC (first few coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=4, 
                                    hop_length=self.hop_length)
        
        # Combine features
        features = np.vstack([rms, zcr, spectral_centroid, spectral_rolloff, mfccs])
        return features.T  # Shape: (n_frames, n_features)
    
    def detect_consonants(self, audio_file, pattern_info):
        """
        Detect consonant timestamps in audio
        
        Args:
            audio_file: path to audio file
            pattern_info: dict with keys like {'pattern': 'ba-aba-ab', 'consonants': ['b', 'b', 'b']}
        
        Returns:
            list of (start_time, end_time) tuples for each consonant
        """
        # Load audio
        audio, sr = librosa.load(audio_file)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Store features for cluster identification
        self.last_features = features
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # Smooth labels to reduce noise
        labels_smooth = medfilt(labels, kernel_size=5)
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(np.arange(len(labels_smooth)), 
                                           sr=sr, hop_length=self.hop_length)
        
        # Find consonant segments
        consonant_segments = self._find_segments(labels_smooth, frame_times, pattern_info)
        
        return consonant_segments
    
    def _identify_clusters(self, labels, features):
        """Identify which cluster represents vowels, consonants, and silence"""
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_features = features[cluster_indices]
            
            cluster_stats[cluster_id] = {
                'count': len(cluster_indices),
                'mean_rms': np.mean(cluster_features[:, 0]),  # RMS energy
                'mean_zcr': np.mean(cluster_features[:, 1]),  # Zero crossing rate
                'mean_centroid': np.mean(cluster_features[:, 2]),  # Spectral centroid
                'std_rms': np.std(cluster_features[:, 0])
            }
        
        # Sort clusters by RMS energy (silence < consonant < vowel typically)
        sorted_clusters = sorted(cluster_stats.keys(), 
                               key=lambda x: cluster_stats[x]['mean_rms'])
        
        # Heuristic assignment:
        # - Silence: lowest RMS energy
        # - Consonants: medium RMS, higher ZCR (more noise-like)
        # - Vowels: highest RMS, lower ZCR (more tonal)
        silence_cluster = sorted_clusters[0]
        
        # Between remaining two, consonants typically have higher ZCR
        remaining = sorted_clusters[1:]
        if len(remaining) == 2:
            consonant_cluster = max(remaining, key=lambda x: cluster_stats[x]['mean_zcr'])
            vowel_cluster = min(remaining, key=lambda x: cluster_stats[x]['mean_zcr'])
        else:
            consonant_cluster = remaining[0]
            vowel_cluster = remaining[0]
        
        return {
            'silence': silence_cluster,
            'consonant': consonant_cluster,
            'vowel': vowel_cluster
        }
    
    def _find_segments(self, labels, frame_times, pattern_info):
        """Find consonant segments based on clustering results"""
        # Get original features for cluster identification
        cluster_mapping = self._identify_clusters(labels, self.last_features)
        
        consonant_cluster = cluster_mapping['consonant']
        
        # Find transitions and group into segments
        transitions = np.where(np.diff(labels) != 0)[0]
        
        segments = []
        start_idx = 0
        
        for trans_idx in transitions:
            end_idx = trans_idx + 1
            segment_label = labels[start_idx]
            
            if segment_label == consonant_cluster:
                start_time = frame_times[start_idx]
                end_time = frame_times[min(end_idx, len(frame_times)-1)]
                segments.append((start_time, end_time))
            
            start_idx = end_idx
        
        # Handle last segment
        if start_idx < len(labels) and labels[start_idx] == consonant_cluster:
            start_time = frame_times[start_idx]
            end_time = frame_times[-1]
            segments.append((start_time, end_time))
        
        return segments
    
    def get_all_segments(self, audio_file, pattern_info):
        """
        Get all segments (vowels, consonants, silence) with labels
        
        Returns:
            dict with 'vowel', 'consonant', 'silence' keys containing segment lists
        """
        # Load audio
        audio, sr = librosa.load(audio_file)
        
        # Extract features
        features = self.extract_features(audio, sr)
        self.last_features = features
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(features_scaled)
        
        # Smooth labels to reduce noise
        labels_smooth = medfilt(labels, kernel_size=5)
        
        # Convert frame indices to time
        frame_times = librosa.frames_to_time(np.arange(len(labels_smooth)), 
                                           sr=sr, hop_length=self.hop_length)
        
        # Identify cluster types
        cluster_mapping = self._identify_clusters(labels_smooth, features)
        
        # Find all segments
        all_segments = {'vowel': [], 'consonant': [], 'silence': []}
        
        transitions = np.where(np.diff(labels_smooth) != 0)[0]
        start_idx = 0
        
        for trans_idx in transitions:
            end_idx = trans_idx + 1
            segment_label = labels_smooth[start_idx]
            
            start_time = frame_times[start_idx]
            end_time = frame_times[min(end_idx, len(frame_times)-1)]
            
            # Determine segment type
            if segment_label == cluster_mapping['vowel']:
                all_segments['vowel'].append((start_time, end_time))
            elif segment_label == cluster_mapping['consonant']:
                all_segments['consonant'].append((start_time, end_time))
            elif segment_label == cluster_mapping['silence']:
                all_segments['silence'].append((start_time, end_time))
            
            start_idx = end_idx
        
        # Handle last segment
        if start_idx < len(labels_smooth):
            segment_label = labels_smooth[start_idx]
            start_time = frame_times[start_idx]
            end_time = frame_times[-1]
            
            if segment_label == cluster_mapping['vowel']:
                all_segments['vowel'].append((start_time, end_time))
            elif segment_label == cluster_mapping['consonant']:
                all_segments['consonant'].append((start_time, end_time))
            elif segment_label == cluster_mapping['silence']:
                all_segments['silence'].append((start_time, end_time))
        
        return all_segments, cluster_mapping
        """
        Create Hamming windows around consonant segments
        
        Args:
            audio_file: path to audio file
            consonant_segments: list of (start_time, end_time) tuples
            window_padding: padding around consonant in seconds
        
        Returns:
            list of windowed audio segments
        """
        audio, sr = librosa.load(audio_file)
        windowed_segments = []
        
        for start_time, end_time in consonant_segments:
            # Add padding
            padded_start = max(0, start_time - window_padding)
            padded_end = min(len(audio)/sr, end_time + window_padding)
            
            # Convert to sample indices
            start_sample = int(padded_start * sr)
            end_sample = int(padded_end * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Apply Hamming window
            hamming_window = np.hamming(len(segment))
            windowed_segment = segment * hamming_window
            
            windowed_segments.append({
                'audio': windowed_segment,
                'start_time': padded_start,
                'end_time': padded_end,
                'original_bounds': (start_time, end_time)
            })
        
    def create_hamming_windows(self, audio_file, consonant_segments, window_padding=0.05):
        """
        Create Hamming windows around consonant segments
        
        Args:
            audio_file: path to audio file
            consonant_segments: list of (start_time, end_time) tuples
            window_padding: padding around consonant in seconds
        
        Returns:
            list of windowed audio segments
        """
        audio, sr = librosa.load(audio_file)
        windowed_segments = []
        
        for start_time, end_time in consonant_segments:
            # Add padding
            padded_start = max(0, start_time - window_padding)
            padded_end = min(len(audio)/sr, end_time + window_padding)
            
            # Convert to sample indices
            start_sample = int(padded_start * sr)
            end_sample = int(padded_end * sr)
            
            # Extract segment
            segment = audio[start_sample:end_sample]
            
            # Apply Hamming window
            hamming_window = np.hamming(len(segment))
            windowed_segment = segment * hamming_window
            
            windowed_segments.append({
                'audio': windowed_segment,
                'start_time': padded_start,
                'end_time': padded_end,
                'original_bounds': (start_time, end_time)
            })
        
        return windowed_segments
    
    def visualize_detection(self, audio_file, all_segments=None, consonant_segments=None):
        """Visualize the detection results with all segment types"""
        audio, sr = librosa.load(audio_file)
        
        plt.figure(figsize=(12, 10))
        
        # Plot waveform
        plt.subplot(4, 1, 1)
        time_axis = np.linspace(0, len(audio)/sr, len(audio))
        plt.plot(time_axis, audio, color='black', alpha=0.7)
        plt.title('Waveform with Segmentation')
        plt.ylabel('Amplitude')
        
        # Color mapping for segments
        colors = {'vowel': 'blue', 'consonant': 'red', 'silence': 'gray'}
        
        if all_segments:
            for segment_type, segments in all_segments.items():
                for start, end in segments:
                    plt.axvspan(start, end, alpha=0.4, color=colors[segment_type], 
                               label=f'{segment_type.capitalize()}' if segments.index((start, end)) == 0 else "")
        elif consonant_segments:
            # Fallback to old behavior
            for start, end in consonant_segments:
                plt.axvspan(start, end, alpha=0.3, color='red', label='Consonant')
        
        plt.legend()
        
        # Plot features
        features = self.extract_features(audio, sr)
        frame_times = librosa.frames_to_time(np.arange(len(features)), 
                                           sr=sr, hop_length=self.hop_length)
        
        plt.subplot(4, 1, 2)
        plt.plot(frame_times, features[:, 0], label='RMS Energy', color='blue')
        plt.title('RMS Energy')
        plt.ylabel('Energy')
        plt.legend()
        
        plt.subplot(4, 1, 3)
        plt.plot(frame_times, features[:, 1], label='Zero Crossing Rate', color='green')
        plt.title('Zero Crossing Rate')
        plt.ylabel('ZCR')
        plt.legend()
        
        # Plot spectrogram
        plt.subplot(4, 1, 4)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, hop_length=self.hop_length, 
                               x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        # Highlight segments on spectrogram
        if all_segments:
            for segment_type, segments in all_segments.items():
                for start, end in segments:
                    plt.axvspan(start, end, alpha=0.3, color=colors[segment_type])
        elif consonant_segments:
            for start, end in consonant_segments:
                plt.axvspan(start, end, alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.show()

# Example usage
def process_audio_file(audio_file, pattern_info):
    """
    Process a single audio file to detect all segments
    
    Args:
        audio_file: path to audio file
        pattern_info: dict with pattern information
    """
    detector = ConsonantDetector(n_clusters=3)  # vowel, consonant, silence
    
    # Get all segments
    all_segments, cluster_mapping = detector.get_all_segments(audio_file, pattern_info)
    
    print(f"Cluster mapping: {cluster_mapping}")
    print(f"\nSegmentation results:")
    
    for segment_type, segments in all_segments.items():
        print(f"\n{segment_type.capitalize()} segments ({len(segments)}):")
        for i, (start, end) in enumerate(segments):
            print(f"  {i+1}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)")
    
    # Create Hamming windows for consonants only
    consonant_segments = all_segments['consonant']
    windowed_segments = detector.create_hamming_windows(audio_file, consonant_segments)
    
    # Visualize results
    detector.visualize_detection(audio_file, all_segments=all_segments)
    
    return all_segments, windowed_segments, cluster_mapping

# Example for batch processing
def process_multiple_files(file_list):
    """Process multiple audio files"""
    detector = ConsonantDetector(n_clusters=3)
    results = {}
    
    for audio_file, pattern_info in file_list:
        print(f"\nProcessing: {audio_file}")
        print(f"Pattern: {pattern_info['pattern']}")
        
        all_segments, cluster_mapping = detector.get_all_segments(audio_file, pattern_info)
        windowed_segments = detector.create_hamming_windows(audio_file, all_segments['consonant'])
        
        results[audio_file] = {
            'all_segments': all_segments,
            'windowed_segments': windowed_segments,
            'cluster_mapping': cluster_mapping
        }
    
    return results

# Example usage:
if __name__ == "__main__":
    # Example for a single file
    audio_file = "ba_aba_ab.wav"  # Replace with your audio file
    pattern_info = {
        'pattern': 'ba-aba-ab',
        'consonants': ['b', 'b', 'b'],
        'vowel': 'a'
    }
    
    all_segments, windowed_segments, cluster_mapping = process_audio_file(audio_file, pattern_info)
    
    # Access specific segment types
    vowel_segments = all_segments['vowel']
    consonant_segments = all_segments['consonant']
    silence_segments = all_segments['silence']
    
    # Example for multiple files
    file_list = [
        ("ba_aba_ab.wav", {'pattern': 'ba-aba-ab', 'consonants': ['b', 'b', 'b']}),
        ("pa_apa_ap.wav", {'pattern': 'pa-apa-ap', 'consonants': ['p', 'p', 'p']}),
        # Add more files as needed
    ]
    
    # results = process_multiple_files(file_list)
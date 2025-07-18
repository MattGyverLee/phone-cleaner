import os
import numpy as np
import librosa
import scipy.signal
from scipy.optimize import minimize_scalar
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioMusicSeparator:
    def __init__(self, source_dir="./source", music_dir="./music", output_dir="./output"):
        self.source_dir = Path(source_dir)
        self.music_dir = Path(music_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.music_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Audio processing parameters
        self.sr = 22050  # Sample rate
        self.hop_length = 512
        self.n_fft = 2048
        self.n_chroma = 12
        
    def load_audio_files(self):
        """Load all audio files from source directory"""
        audio_files = []
        audio_data = []
        
        # Supported audio formats
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        for ext in extensions:
            audio_files.extend(self.source_dir.glob(f'*{ext}'))
        
        print(f"Found {len(audio_files)} audio files")
        
        for file_path in audio_files:
            try:
                y, sr = librosa.load(file_path, sr=self.sr)
                audio_data.append({
                    'path': file_path,
                    'audio': y,
                    'name': file_path.stem
                })
                print(f"Loaded: {file_path.name} ({len(y)/sr:.1f}s)")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        return audio_data
    
    def extract_chromagram(self, audio):
        """Extract chromagram features for alignment"""
        chroma = librosa.feature.chroma_stft(
            y=audio, 
            sr=self.sr, 
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_chroma=self.n_chroma
        )
        # Normalize each frame
        chroma_norm = librosa.util.normalize(chroma, axis=0)
        return chroma_norm
    
    def find_alignment_offset(self, chroma1, chroma2, max_offset_sec=30):
        """Find optimal time offset between two chromagrams"""
        max_offset_frames = int(max_offset_sec * self.sr / self.hop_length)
        
        # Compute cross-correlation
        correlation = scipy.signal.correlate(
            chroma1.T.flatten(), 
            chroma2.T.flatten(), 
            mode='full'
        )
        
        # Find peak within reasonable range
        center = len(correlation) // 2
        start_idx = max(0, center - max_offset_frames * self.n_chroma)
        end_idx = min(len(correlation), center + max_offset_frames * self.n_chroma)
        
        local_corr = correlation[start_idx:end_idx]
        peak_idx = np.argmax(local_corr)
        
        # Convert back to frame offset
        offset_frames = (peak_idx + start_idx - center) // self.n_chroma
        offset_seconds = offset_frames * self.hop_length / self.sr
        
        return offset_frames, offset_seconds, correlation[start_idx + peak_idx]
    
    def find_earliest_audio_start(self, audio_data):
        """Find the clip where meaningful audio starts earliest"""
        earliest_starts = []
        
        for i, data in enumerate(audio_data):
            audio = data['audio']
            
            # Calculate RMS energy in sliding windows
            frame_length = int(0.1 * self.sr)  # 100ms windows
            hop_length = int(0.05 * self.sr)   # 50ms hop
            
            rms_energy = []
            for start in range(0, len(audio) - frame_length, hop_length):
                window = audio[start:start + frame_length]
                rms = np.sqrt(np.mean(window ** 2))
                rms_energy.append(rms)
            
            rms_energy = np.array(rms_energy)
            
            # Find threshold for meaningful audio (above background noise)
            noise_threshold = np.percentile(rms_energy, 20)  # Bottom 20% as noise floor
            audio_threshold = noise_threshold * 3  # 3x above noise floor
            
            # Find first sustained audio activity
            sustained_frames = 10  # Require 10 consecutive frames (0.5s) above threshold
            
            for j in range(len(rms_energy) - sustained_frames):
                if np.all(rms_energy[j:j + sustained_frames] > audio_threshold):
                    earliest_start = j * hop_length / self.sr
                    earliest_starts.append(earliest_start)
                    print(f"  {data['name']}: audio starts at {earliest_start:.2f}s")
                    break
            else:
                # If no sustained activity found, use first frame above threshold
                above_threshold = np.where(rms_energy > audio_threshold)[0]
                if len(above_threshold) > 0:
                    earliest_start = above_threshold[0] * hop_length / self.sr
                    earliest_starts.append(earliest_start)
                    print(f"  {data['name']}: audio starts at {earliest_start:.2f}s (weak detection)")
                else:
                    earliest_starts.append(0.0)
                    print(f"  {data['name']}: audio starts at 0.00s (no clear start detected)")
        
        # Return index of clip with earliest start
        return np.argmin(earliest_starts)
    
    def align_clips(self, audio_data):
        """Align all clips to find common music timing"""
        print("\nAligning clips...")
        
        # Extract chromagrams
        chromas = []
        for data in audio_data:
            chroma = self.extract_chromagram(data['audio'])
            chromas.append(chroma)
            data['chroma'] = chroma
        
        # Find best reference clip (earliest audio start)
        ref_idx = self.find_earliest_audio_start(audio_data)
        ref_chroma = chromas[ref_idx]
        
        print(f"Using '{audio_data[ref_idx]['name']}' as reference (earliest audio start)")
        
        # Align all clips to reference
        alignments = []
        for i, chroma in enumerate(chromas):
            if i == ref_idx:
                alignments.append({'offset_frames': 0, 'offset_seconds': 0.0, 'score': 1.0})
            else:
                offset_frames, offset_seconds, score = self.find_alignment_offset(ref_chroma, chroma)
                alignments.append({
                    'offset_frames': offset_frames,
                    'offset_seconds': offset_seconds,
                    'score': score
                })
                print(f"  {audio_data[i]['name']}: offset = {offset_seconds:.2f}s, score = {score:.3f}")
        
        return alignments, ref_idx
    
    def extract_music(self, audio_data, alignments):
        """Extract clean music by averaging aligned spectrograms"""
        print("\nExtracting music...")
        
        # Convert to spectrograms
        spectrograms = []
        for i, data in enumerate(audio_data):
            stft = librosa.stft(data['audio'], hop_length=self.hop_length, n_fft=self.n_fft)
            spectrograms.append(stft)
            data['stft'] = stft
        
        # Find the maximum length needed (longest clip + maximum alignment offset)
        max_frames = max(spec.shape[1] for spec in spectrograms)
        max_offset = max(abs(align['offset_frames']) for align in alignments)
        
        # Add buffer for alignment
        max_frames += max_offset

        print(f"Max frames: {max_frames}, approx. {max_frames * self.hop_length / self.sr:.1f}s")

        # Align and pad spectrograms to same length
        aligned_spectrograms = []
        for i, spec in enumerate(spectrograms):
            offset = alignments[i]['offset_frames']
            
            # Create a padded spectrogram of max length
            aligned_spec = np.zeros((spec.shape[0], max_frames), dtype=spec.dtype)
            
            if offset >= 0:
                # Positive offset: music starts later in this clip
                # Copy the available frames starting from offset
                end_idx = min(max_frames, offset + spec.shape[1])
                copy_frames = end_idx - offset
                aligned_spec[:, offset:end_idx] = spec[:, :copy_frames]
            else:
                # Negative offset: music starts earlier in this clip
                # Copy frames starting from the beginning, skipping the offset
                start_src = -offset
                copy_frames = min(spec.shape[1] - start_src, max_frames)
                if copy_frames > 0:
                    aligned_spec[:, :copy_frames] = spec[:, start_src:start_src + copy_frames]
            
            aligned_spectrograms.append(aligned_spec)
        
        # Stack and create a mask for valid (non-zero) regions across all spectrograms
        stacked_specs = np.stack(aligned_spectrograms, axis=0)
        
        # Create a mask where at least 2 clips have content (or all if only 2 clips)
        min_clips = 2 if len(audio_data) > 2 else len(audio_data)
        valid_mask = (np.abs(stacked_specs).sum(axis=1) > 0).sum(axis=0) >= min_clips

        # Find the longest contiguous segment with valid data
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(valid_mask)

        if num_features == 0:
            print("Warning: No valid overlapping music found. Using entire range.")
            start_idx, end_idx = 0, max_frames
        else:
            # Find largest continuous segment
            segment_sizes = ndimage.sum(valid_mask, labeled_mask, range(1, num_features + 1))
            largest_segment = np.argmax(segment_sizes) + 1
            segment_indices = np.where(labeled_mask == largest_segment)[0]

            start_idx, end_idx = segment_indices[0], segment_indices[-1] + 1
            segment_duration = (end_idx - start_idx) * self.hop_length / self.sr
            print(f"Found continuous music segment: {segment_duration:.1f}s")

        # Use median to suppress speech variations, but only in the valid region
        music_magnitude = np.zeros((stacked_specs.shape[1], end_idx - start_idx))

        # Only process the valid segment to save memory
        valid_segment = stacked_specs[:, :, start_idx:end_idx]
        music_magnitude = np.median(np.abs(valid_segment), axis=0)

        # Use phase from the reference clip
        ref_spec = aligned_spectrograms[0][:, start_idx:end_idx]
        music_phase = np.angle(ref_spec)

        # Combine magnitude and phase
        music_stft = music_magnitude * np.exp(1j * music_phase)
        # Convert back to time domain
        music_audio = librosa.istft(music_stft, hop_length=self.hop_length)
        print(f"Extracted music length: {len(music_audio)/self.sr:.1f}s")
        
        return music_audio, music_stft
    
    def align_music_to_clip(self, music_audio, clip_audio):
        """Fine-tune alignment between extracted music and individual clip"""
        # Extract features for fine alignment
        music_chroma = self.extract_chromagram(music_audio)
        clip_chroma = self.extract_chromagram(clip_audio)

        # Find best alignment
        offset_frames, offset_seconds, score = self.find_alignment_offset(clip_chroma, music_chroma)

        return offset_frames, offset_seconds, score

    def subtract_music(self, clip_audio, music_audio, offset_frames, alpha=None):
        """Subtract music from clip with adaptive gain"""
        # Convert to spectrograms
        clip_stft = librosa.stft(clip_audio, hop_length=self.hop_length, n_fft=self.n_fft)
        music_stft = librosa.stft(music_audio, hop_length=self.hop_length, n_fft=self.n_fft)

        # Ensure music_stft is at least as long as clip_stft
        if music_stft.shape[1] < clip_stft.shape[1]:
            # Pad music with zeros if it's shorter
            pad_width = clip_stft.shape[1] - music_stft.shape[1]
            music_stft = np.pad(music_stft, ((0, 0), (0, pad_width)), mode='constant')

        # Align music spectrogram based on offset
        if offset_frames >= 0:
            # Positive offset: music starts later, pad music at the beginning
            music_stft_aligned = np.pad(music_stft, ((0, 0), (offset_frames, 0)), mode='constant')
        else:
            # Negative offset: music starts earlier, crop music from the beginning
            music_stft_aligned = music_stft[:, -offset_frames:]

        # Match lengths - trim to shorter of the two
        min_frames = min(clip_stft.shape[1], music_stft_aligned.shape[1])

        if min_frames <= 0:
            print("Warning: No overlapping frames found, returning original audio")
            return clip_audio, 0.0

        clip_stft = clip_stft[:, :min_frames]
        music_stft_aligned = music_stft_aligned[:, :min_frames]

        # Estimate optimal gain if not provided
        if alpha is None:
            def objective(a):
                residual = clip_stft - a * music_stft_aligned
                return np.mean(np.abs(residual) ** 2)

            result = minimize_scalar(objective, bounds=(0, 2), method='bounded')
            alpha = result.x

        # Perform spectral subtraction with Wiener filtering
        music_power = np.abs(music_stft_aligned) ** 2
        clip_power = np.abs(clip_stft) ** 2

        # Wiener filter
        wiener_gain = np.maximum(
            (clip_power - alpha * music_power) / (clip_power + 1e-10),
            0.1  # Minimum gain to avoid artifacts
        )

        # Apply filter
        clean_stft = clip_stft * wiener_gain

        # Convert back to time domain
        clean_audio = librosa.istft(clean_stft, hop_length=self.hop_length)

        return clean_audio, alpha

    def process_files(self):
        """Main processing pipeline"""
        print("Starting audio music separation pipeline...")
        
        # Load audio files
        audio_data = self.load_audio_files()
        
        if len(audio_data) < 2:
            print("Need at least 2 audio files for music extraction")
            return
        
        # Align clips
        alignments, ref_idx = self.align_clips(audio_data)
        
        # Extract music
        music_audio, music_stft = self.extract_music(audio_data, alignments)
        
        # Save extracted music
        music_path = self.music_dir / "extracted_music.wav"
        sf.write(music_path, music_audio, self.sr)
        print(f"\nSaved extracted music: {music_path} ({len(music_audio)/self.sr:.1f}s)")
        
        # Process each clip
        print("\nProcessing individual clips...")
        for i, data in enumerate(audio_data):
            print(f"\nProcessing: {data['name']}")
            
            # Fine-tune alignment
            offset_frames, offset_seconds, score = self.align_music_to_clip(music_audio, data['audio'])
            print(f"  Fine alignment offset: {offset_seconds:.2f}s, score: {score:.3f}")
            
            # Subtract music
            clean_audio, alpha = self.subtract_music(data['audio'], music_audio, offset_frames)
            print(f"  Estimated music gain: {alpha:.3f}")
            
            # Save result
            output_path = self.output_dir / f"{data['name']}_clean.wav"
            sf.write(output_path, clean_audio, self.sr)
            print(f"  Saved: {output_path}")
        
        print(f"\nProcessing complete!")
        print(f"Music saved to: {self.music_dir}")
        print(f"Clean audio saved to: {self.output_dir}")

def main():
    # Create and run the separator
    separator = AudioMusicSeparator()
    separator.process_files()

if __name__ == "__main__":
    main()

"""
Audio Music Separation Pipeline - Implementation Notes

## Key Features:

**1. Multi-Clip Alignment**
- Uses chromagram features (harmonic content) for robust alignment
- Finds optimal time offsets between clips using cross-correlation
- Handles up to 30 seconds of timing variation

**2. Music Extraction**
- Converts aligned clips to spectrograms
- Uses median averaging to suppress speech (varies between clips) while preserving music (constant)
- Reconstructs clean music using magnitude median + reference phase

**3. Adaptive Music Subtraction**
- Fine-tunes alignment between extracted music and each clip
- Estimates optimal music gain using least squares
- Applies Wiener filtering for clean separation
- Prevents artifacts with minimum gain threshold

## Usage:

1. **Install dependencies:**
```bash
pip install librosa soundfile scipy numpy pathlib
```

2. **Setup folders:**
   - Put your audio files in `./source/`
   - Creates `./music/` and `./output/` automatically

3. **Run the script:**
```bash
python audio_separator.py
```

## Output:
- `./music/extracted_music.wav` - The longest possible clean music track
- `./output/[filename]_clean.wav` - Each original file with music removed

The pipeline supports common audio formats (WAV, MP3, FLAC, M4A, OGG) and provides detailed progress feedback including alignment scores and estimated music gains.

The algorithm is particularly robust because it uses multiple clips as references, making it much more effective than traditional single-reference approaches.
"""
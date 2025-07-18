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
        
        # Normalize input features
        norm_chroma1 = librosa.util.normalize(chroma1, axis=0)
        norm_chroma2 = librosa.util.normalize(chroma2, axis=0)

        # Use 2D correlation for better alignment of harmonic features
        corr = np.zeros(2 * max_offset_frames + 1)

        for i in range(-max_offset_frames, max_offset_frames + 1):
            if i < 0:
                # chroma2 is shifted to the left (starts earlier)
                idx = -i
                if idx >= norm_chroma2.shape[1]:
                    continue
                len_compare = min(norm_chroma1.shape[1], norm_chroma2.shape[1] - idx)
                if len_compare <= 0:
                    continue
                corr_val = np.sum(norm_chroma1[:, :len_compare] * norm_chroma2[:, idx:idx+len_compare])
            else:
                # chroma2 is shifted to the right (starts later)
                if i >= norm_chroma1.shape[1]:
                    continue
                len_compare = min(norm_chroma1.shape[1] - i, norm_chroma2.shape[1])
                if len_compare <= 0:
                    continue
                corr_val = np.sum(norm_chroma1[:, i:i+len_compare] * norm_chroma2[:, :len_compare])

            corr[i + max_offset_frames] = corr_val / len_compare  # Normalize by length

        # Find peak
        peak_idx = np.argmax(corr)
        offset_frames = peak_idx - max_offset_frames
        offset_seconds = offset_frames * self.hop_length / self.sr

        return offset_frames, offset_seconds, corr[peak_idx]

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

        """Align clips focusing on the beginning instrumental section with high precision"""
        print("\nAligning clips...")


        # Focus on the beginning of each clip where music is likely to be present without voice
        initial_sections = []
        for i, data in enumerate(audio_data):
            audio = data['audio']


            # Use first 3-5 seconds (likely to contain intro music)
            duration = min(5.0, len(audio) / self.sr)
            samples = int(duration * self.sr)

            initial_sections.append({
                'audio': audio[:samples],
                'name': data['name'],
                'full_audio': audio
            })

            print(f"  {data['name']}: Using first {duration:.1f}s for alignment")


        # Extract chromagrams for higher precision alignment
        chromas = []
        for section in initial_sections:













































































































































































            # Use higher time-frequency resolution for precise alignment
            hop_length_align = self.hop_length // 2  # Higher time resolution

            chroma = librosa.feature.chroma_stft(
                y=section['audio'],
                sr=self.sr,
                hop_length=hop_length_align,
                n_fft=self.n_fft,
                n_chroma=24  # Higher pitch resolution
            )
            chromas.append(chroma)
            section['chroma'] = chroma
































        # Choose the clip with highest energy as reference
        energies = [np.sum(np.abs(section['audio'])) for section in initial_sections]
        ref_idx = np.argmax(energies)
        ref_chroma = chromas[ref_idx]

        print(f"Using '{initial_sections[ref_idx]['name']}' as reference (highest energy)")

        # Align all clips to reference with sub-frame precision
        alignments = []
        for i, chroma in enumerate(chromas):
            if i == ref_idx:
                alignments.append({
                    'offset_frames': 0,
                    'offset_seconds': 0.0,
                    'score': 1.0,
                    'offset_samples': 0
                })
            else:
                # Use normalized cross-correlation for robust alignment
                corr = scipy.signal.correlate2d(
                    chroma, ref_chroma, mode='valid', boundary='fill', fillvalue=0
                )
                peak_idx = np.argmax(corr)

                # Convert to time offset
                offset_frames = peak_idx / (self.hop_length // 2)  # Account for finer hop_length
                offset_seconds = offset_frames * self.hop_length / self.sr
                offset_samples = int(offset_seconds * self.sr)

                alignments.append({
                    'offset_frames': offset_frames,
                    'offset_seconds': offset_seconds,
                    'score': corr[peak_idx],
                    'offset_samples': offset_samples
                })

                print(f"  {initial_sections[i]['name']}: offset = {offset_seconds:.3f}s, score = {corr[peak_idx]:.3f}")

        return alignments, ref_idx

    def extract_music(self, audio_data, alignments):
        """Extract music using principal component analysis of aligned spectrograms"""
        print("\nExtracting music using spectral analysis...")

        # Convert to time-domain aligned audio first
        max_length = 0
        for i, data in enumerate(audio_data):

            aligned_length = len(data['audio']) + abs(alignments[i]['offset_samples'])
            max_length = max(max_length, aligned_length)

        # Create aligned audio arrays
        aligned_audio = []
        for i, data in enumerate(audio_data):
            offset = alignments[i]['offset_samples']
            audio = data['audio']

            # Create padded array
            padded_audio = np.zeros(max_length)

            if offset >= 0:
                # This audio starts later than reference
                padded_audio[offset:offset + len(audio)] = audio
            else:
                # This audio starts earlier than reference
                padded_audio[:len(audio) + offset] = audio[-offset:]

            aligned_audio.append(padded_audio)

        # Convert to spectrograms
        aligned_specs = []
        for audio in aligned_audio:
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            aligned_specs.append(stft)

        # Calculate magnitudes and phases
        magnitudes = [np.abs(spec) for spec in aligned_specs]
        phases = [np.angle(spec) for spec in aligned_specs]

        # Stack magnitudes
        stacked_magnitudes = np.stack(magnitudes, axis=0)

        # Use Principal Component Analysis approach:
        # The first principal component tends to capture the common elements (music)
        # while later components capture the varying elements (speech)

        # For each frequency bin, calculate correlation matrix across time frames
        n_clips, n_freq, n_frames = stacked_magnitudes.shape

        # Reshape to combine frequency and time
        reshaped_magnitudes = stacked_magnitudes.reshape(n_clips, -1)

        # Normalize each clip's magnitudes
        magnitudes_mean = np.mean(reshaped_magnitudes, axis=1, keepdims=True)
        magnitudes_std = np.std(reshaped_magnitudes, axis=1, keepdims=True)
        normalized_magnitudes = (reshaped_magnitudes - magnitudes_mean) / (magnitudes_std + 1e-8)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(normalized_magnitudes)

        # Find the clip that best correlates with all others (most representative)
        avg_correlation = np.mean(corr_matrix, axis=1)
        best_clip_idx = np.argmax(avg_correlation)

        print(f"Using {audio_data[best_clip_idx]['name']} as primary source (most representative)")

        # Use the magnitude from the most representative clip
        music_magnitude = magnitudes[best_clip_idx]

        # Scale other frequencies based on correlation
        for i in range(n_freq):
            freq_corr = np.corrcoef([mag[i, :] for mag in magnitudes])[best_clip_idx]

            # Only keep highly correlated frequency bins
            threshold = 0.5  # Correlation threshold
            scaling = np.mean(freq_corr > threshold)

            if scaling < 0.5:  # If less than half of clips have this frequency
                music_magnitude[i, :] *= scaling  # Scale down this frequency bin

        # Reconstruct with phase from best clip
        music_stft = music_magnitude * np.exp(1j * phases[best_clip_idx])

        # Convert back to time domain
        music_audio = librosa.istft(music_stft, hop_length=self.hop_length)

        print(f"Extracted music length: {len(music_audio)/self.sr:.1f}s")

        return music_audio, music_stft

    def align_music_to_clip(self, music_audio, clip_audio):
        """Fine-tune alignment between extracted music and individual clip with high precision"""
        # For more precise alignment, use onset strength
        # This focuses on rhythmic elements which are more distinct in music

        # Extract onset envelopes
        hop_length_onset = 512  # Smaller hop for precise alignment

        music_onset = librosa.onset.onset_strength(
            y=music_audio, sr=self.sr, hop_length=hop_length_onset
        )

        clip_onset = librosa.onset.onset_strength(
            y=clip_audio, sr=self.sr, hop_length=hop_length_onset
        )

        # Normalize
        music_onset = music_onset / np.max(music_onset) if np.max(music_onset) > 0 else music_onset
        clip_onset = clip_onset / np.max(clip_onset) if np.max(clip_onset) > 0 else clip_onset

        # Find alignment using cross-correlation
        correlation = scipy.signal.correlate(clip_onset, music_onset, mode='full')

        # Find peak within reasonable range (±5 seconds)
        max_offset_frames = int(5.0 * self.sr / hop_length_onset)

        center = len(correlation) // 2
        start_idx = max(0, center - max_offset_frames)
        end_idx = min(len(correlation), center + max_offset_frames)

        local_corr = correlation[start_idx:end_idx]
        peak_idx = np.argmax(local_corr)

        # Convert to frame offset
        offset_frames = (peak_idx + start_idx - center)

        # Convert to STFT frames (different hop_length)
        offset_frames_stft = offset_frames * hop_length_onset / self.hop_length

        # Convert to time
        offset_seconds = offset_frames * hop_length_onset / self.sr

        return offset_frames_stft, offset_seconds, local_corr[peak_idx]

    def subtract_music(self, clip_audio, music_audio, offset_frames, alpha=None):
        """Subtract music from clip using adaptive filtering"""
        # Convert offset_frames to integer
        offset_frames = int(np.round(offset_frames))

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
            # Trim to match clip length if it became longer
            music_stft_aligned = music_stft_aligned[:, :clip_stft.shape[1]] if music_stft_aligned.shape[1] > clip_stft.shape[1] else music_stft_aligned
        else:
            # Negative offset: music starts earlier, crop music from the beginning
            # Ensure we don't try to start from a negative index
            start_idx = min(-offset_frames, music_stft.shape[1])
            if start_idx >= music_stft.shape[1]:
                print("Warning: Alignment offset is too large, returning original audio")
                return clip_audio, 0.0

            music_stft_aligned = music_stft[:, start_idx:]

        # Match lengths - trim to shorter of the two
        min_frames = min(clip_stft.shape[1], music_stft_aligned.shape[1])

        if min_frames <= 0:
            print("Warning: No overlapping frames found, returning original audio")
            return clip_audio, 0.0

        clip_stft = clip_stft[:, :min_frames]
        music_stft_aligned = music_stft_aligned[:, :min_frames]

        # Estimate optimal gain if not provided
        if alpha is None:
            # Try multiple alpha values and choose the best one
            alphas = np.linspace(0.2, 1.5, 14)
            best_alpha = 0
            best_score = -float('inf')

            clip_mag = np.abs(clip_stft)
            music_mag = np.abs(music_stft_aligned)

            for a in alphas:
                # Calculate residual after music removal
                residual = np.maximum(clip_mag - a * music_mag, 0)

                # Calculate energy preservation ratio (higher is better)
                # We want to preserve as much energy as possible while removing music
                energy_preserved = np.sum(residual) / np.sum(clip_mag)

                # Calculate spectral correlation between music and residual
                # Lower correlation means better music removal
                flat_music = music_mag.flatten()
                flat_residual = residual.flatten()

                # Normalize for correlation
                flat_music = (flat_music - np.mean(flat_music)) / (np.std(flat_music) + 1e-8)
                flat_residual = (flat_residual - np.mean(flat_residual)) / (np.std(flat_residual) + 1e-8)

                corr = np.corrcoef(flat_music, flat_residual)[0, 1]
                decorrelation = 1 - abs(corr)  # Higher is better

                # Combined score
                score = 0.7 * energy_preserved + 0.3 * decorrelation

                if score > best_score:
                    best_score = score
                    best_alpha = a

            alpha = best_alpha
            print(f"    Optimal music gain found: {alpha:.3f}")

        # Adaptive spectral subtraction
        clip_mag = np.abs(clip_stft)
        music_mag = np.abs(music_stft_aligned)
        clip_phase = np.angle(clip_stft)

        # Compute time-frequency mask
        mask = np.maximum(1.0 - alpha * music_mag / (clip_mag + 1e-8), 0.1)

        # Apply mask to preserve original phase
        result_stft = clip_stft * mask

        # Convert back to time domain
        clean_audio = librosa.istft(result_stft, hop_length=self.hop_length, length=len(clip_audio))

        return clean_audio, alpha

    def process_files(self):
        """Main processing pipeline"""
        print("Starting audio music separation pipeline...")

        # Load audio files
        audio_data = self.load_audio_files()

        if len(audio_data) < 2:
            print("Need at least 2 audio files for music extraction")
            return

        # Align clips focusing on initial instrumental sections
        alignments, ref_idx = self.align_clips(audio_data)

        # Extract music by finding common frequencies
        music_audio, music_stft = self.extract_music(audio_data, alignments)
        # Save extracted music
        music_path = self.music_dir / "extracted_music.wav"
        sf.write(music_path, music_audio, self.sr)
        print(f"\nSaved extracted music: {music_path} ({len(music_audio)/self.sr:.1f}s)")

        # Process each clip
        print("\nProcessing individual clips...")
        for i, data in enumerate(audio_data):
            print(f"\nProcessing: {data['name']}")

            # Fine-tune alignment for this specific clip
            offset_frames, offset_seconds, score = self.align_music_to_clip(music_audio, data['audio'])
            print(f"  Fine alignment offset: {offset_seconds:.2f}s, score: {score:.3f}")

            # Remove music using spectral masking
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
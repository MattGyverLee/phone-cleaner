import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_audio_segments(audio_path):
    """
    Simple audio segmentation to identify likely 'b' and 'a' sounds
    based on energy and spectral characteristics.
    """
    print(f"Analyzing: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Calculate frame parameters
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop
    
    # Extract features
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Time axis
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Multiple thresholds for better detection
    high_threshold = np.percentile(rms, 70)  # For clear speech
    low_threshold = np.percentile(rms, 10)   # For weak sounds like 'b'
    
    # Find all potential segments with low threshold
    speech_mask = rms > low_threshold
    segments = find_speech_segments(speech_mask, times, min_duration=0.020)  # Lower minimum
    
    # More detailed analysis
    detailed_segments = analyze_detailed_segments(y, sr, rms, zcr, spectral_centroid, times, segments)
    
    # Additional fine-grained analysis for 'b' detection
    print(f"\n--- Fine-grained Analysis for 'b' Detection ---")
    b_candidates = detect_b_sound_candidates(y, sr, rms, zcr, times)
    for candidate in b_candidates:
        print(f"Potential 'b' sound: {candidate['start']:.3f}s - {candidate['end']:.3f}s (confidence: {candidate['confidence']:.1f}%)")
    
    print(f"\nAudio Duration: {duration:.3f} seconds")
    print(f"Low threshold: {low_threshold:.4f}, High threshold: {high_threshold:.4f}")
    print(f"Found {len(segments)} potential segments:")
    
    return detailed_segments

def analyze_detailed_segments(y, sr, rms, zcr, spectral_centroid, times, segments):
    """More detailed segment analysis to identify 'b' and 'a' sounds"""
    hop_length = int(sr * 0.010)
    detailed_results = []
    
    for i, (start, end) in enumerate(segments):
        segment_duration = end - start
        
        # Get features for this segment
        start_frame = int(start * sr / hop_length)
        end_frame = int(end * sr / hop_length)
        
        if start_frame < len(rms) and end_frame <= len(rms) and end_frame > start_frame:
            seg_rms = rms[start_frame:end_frame]
            seg_zcr = zcr[start_frame:end_frame]
            seg_centroid = spectral_centroid[start_frame:end_frame]
            
            # Calculate statistics
            avg_rms = np.mean(seg_rms)
            max_rms = np.max(seg_rms)
            avg_zcr = np.mean(seg_zcr)
            avg_centroid = np.mean(seg_centroid)
            
            # Look for burst patterns (typical of 'b')
            rms_increase = detect_burst_pattern(seg_rms)
            
            # Classify more precisely
            segment_type, confidence = classify_segment_detailed(
                avg_rms, max_rms, avg_zcr, avg_centroid, segment_duration, rms_increase
            )
            
            result = {
                'segment': i + 1,
                'start': start,
                'end': end,
                'duration': segment_duration,
                'type': segment_type,
                'confidence': confidence,
                'avg_rms': avg_rms,
                'avg_zcr': avg_zcr,
                'avg_centroid': avg_centroid
            }
            
            detailed_results.append(result)
            
            print(f"  Segment {i+1}: {start:.3f}s - {end:.3f}s ({segment_duration:.3f}s)")
            print(f"    RMS: {avg_rms:.4f}, ZCR: {avg_zcr:.4f}, Centroid: {avg_centroid:.1f} Hz")
            print(f"    Classification: {segment_type} (confidence: {confidence:.1f}%)")
            
            # Detailed 'b' sound analysis
            if 'b' in segment_type.lower():
                print(f"    > Possible 'b' sound at {start:.3f}s - {end:.3f}s")
            elif 'a' in segment_type.lower():
                print(f"    > Possible 'a' sound at {start:.3f}s - {end:.3f}s")
    
    return detailed_results

def detect_b_sound_candidates(y, sr, rms, zcr, times):
    """Specific detection for 'b' sounds using fine-grained analysis"""
    hop_length = int(sr * 0.010)
    candidates = []
    
    # Look for burst patterns in the first 200ms where 'b' is likely
    search_frames = min(20, len(rms))  # First 200ms
    
    for i in range(1, search_frames):
        # Look for energy bursts
        if i < len(rms) - 1:
            current_rms = rms[i]
            prev_rms = rms[i-1] if i > 0 else 0
            next_rms = rms[i+1] if i < len(rms) - 1 else current_rms
            
            # Burst pattern: low -> high -> sustained
            burst_ratio = current_rms / (prev_rms + 0.0001)  # Avoid division by zero
            
            if (burst_ratio > 3 and current_rms > 0.05 and 
                times[i] < 0.15):  # Within first 150ms
                
                # Find end of burst
                end_frame = i
                for j in range(i + 1, min(i + 10, len(rms))):
                    if rms[j] > current_rms * 0.8:  # Still high energy
                        end_frame = j
                    else:
                        break
                
                confidence = min(95, 40 + burst_ratio * 10)
                
                candidates.append({
                    'start': times[max(0, i-1)],
                    'end': times[min(end_frame, len(times)-1)],
                    'confidence': confidence,
                    'burst_ratio': burst_ratio
                })
    
    return candidates

def detect_burst_pattern(rms_values):
    """Detect burst pattern typical of stop consonants like 'b'"""
    if len(rms_values) < 3:
        return False
    
    # Look for sharp increase in energy (burst)
    for i in range(1, len(rms_values)):
        if rms_values[i] > 2 * rms_values[i-1]:
            return True
    
    return False

def classify_segment_detailed(avg_rms, max_rms, avg_zcr, avg_centroid, duration, has_burst):
    """Detailed classification with confidence scores"""
    
    confidence = 50  # Base confidence
    
    # Very short segments with burst pattern = likely 'b'
    if duration < 0.08 and has_burst:
        confidence += 30
        return "'b' consonant (stop burst)", min(confidence, 95)
    
    # Very short with high energy = likely consonant
    elif duration < 0.06:
        if avg_zcr > 0.1:
            confidence += 20
            return "Brief consonant (possibly 'b')", min(confidence, 85)
        else:
            return "Brief sound", confidence
    
    # Longer segments with steady energy = likely vowel
    elif duration > 0.15:
        if avg_zcr < 0.08 and 500 < avg_centroid < 1500:
            confidence += 35
            return "'a' vowel (sustained)", min(confidence, 95)
        elif avg_zcr < 0.1:
            confidence += 20
            return "Vowel-like sound", min(confidence, 80)
        else:
            return "Mixed/complex sound", confidence
    
    # Medium duration
    else:
        if avg_zcr > 0.1:
            confidence += 15
            return "Consonant-like", min(confidence, 75)
        elif avg_zcr < 0.05:
            confidence += 25
            return "Vowel-like (possibly 'a')", min(confidence, 85)
        else:
            return "Transitional sound", confidence
    
    # Plot analysis
    plt.figure(figsize=(12, 8))
    
    # Waveform
    plt.subplot(4, 1, 1)
    plt.plot(np.linspace(0, duration, len(y)), y)
    plt.title('Waveform')
    plt.ylabel('Amplitude')
    
    # RMS Energy
    plt.subplot(4, 1, 2)
    plt.plot(times, rms, label='RMS Energy')
    plt.axhline(y=low_threshold, color='r', linestyle='--', label='Low Threshold')
    plt.axhline(y=high_threshold, color='orange', linestyle='--', label='High Threshold')
    for start, end in segments:
        plt.axvspan(start, end, alpha=0.3, color='yellow')
    plt.title('RMS Energy')
    plt.ylabel('Energy')
    plt.legend()
    
    # Zero Crossing Rate
    plt.subplot(4, 1, 3)
    plt.plot(times, zcr, label='ZCR')
    plt.title('Zero Crossing Rate')
    plt.ylabel('ZCR')
    plt.legend()
    
    # Spectral Centroid
    plt.subplot(4, 1, 4)
    plt.plot(times, spectral_centroid, label='Spectral Centroid')
    plt.title('Spectral Centroid')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plot_path = audio_path.replace('.wav', '_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close instead of showing
    print(f"Analysis plot saved to: {plot_path}")
    
    return detailed_segments

def find_speech_segments(speech_mask, times, min_duration=0.05):
    """Find continuous speech segments"""
    segments = []
    in_speech = False
    start_time = 0
    
    for i, is_speech in enumerate(speech_mask):
        if is_speech and not in_speech:
            # Start of speech
            start_time = times[i]
            in_speech = True
        elif not is_speech and in_speech:
            # End of speech
            end_time = times[i]
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
            in_speech = False
    
    # Handle case where speech continues to end
    if in_speech:
        segments.append((start_time, times[-1]))
    
    return segments


if __name__ == "__main__":
    audio_file = r"D:\Github\phone-cleaner\Hamming-Docs\iso_[b]_b_voiced unaspirated bilabial stop.wav"
    
    try:
        segments = analyze_audio_segments(audio_file)
        
        print(f"\n" + "="*50)
        print(f"FINAL ANALYSIS SUMMARY")
        print(f"="*50)
        
        # Based on the analysis, provide best estimates
        if len(segments) > 0:
            main_segment = segments[0]
            start_time = main_segment['start']
            end_time = main_segment['end']
            
            # Estimate 'b' sound at the very beginning of speech
            b_estimate_start = max(0, start_time - 0.020)  # 20ms before main segment
            b_estimate_end = min(start_time + 0.080, end_time)  # Up to 80ms into segment
            
            # Estimate 'a' sound as the sustained portion
            a_estimate_start = b_estimate_end
            a_estimate_end = end_time
            
            print(f"Best Estimates:")
            print(f"  'b' sound: ~{b_estimate_start:.3f}s - {b_estimate_end:.3f}s")
            print(f"  'a' sound: ~{a_estimate_start:.3f}s - {a_estimate_end:.3f}s")
            print(f"")
            print(f"Note: The 'b' sound is very brief and may overlap with the")
            print(f"beginning of the 'a' sound. Check the generated plot for visual confirmation.")
        else:
            print("No clear speech segments detected.")
        
        print(f"\nAnalysis complete. Check the generated plot for visual analysis.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the audio file exists and required libraries are installed.")
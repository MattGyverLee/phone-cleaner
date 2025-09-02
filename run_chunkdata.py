import sys
import os
import importlib.util

# Load the module directly from file path
spec = importlib.util.spec_from_file_location("chunkdata", r"D:\Github\phone-cleaner\Hamming-Docs\1. Chunkdata.py")
chunkdata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunkdata)

# Run the function on the specified audio file
audio_file = r"D:\Github\phone-cleaner\Hamming-Docs\iso_[b]_b_voiced unaspirated bilabial stop.wav"

if os.path.exists(audio_file):
    print(f"Processing audio file: {audio_file}")
    generated_textgrid_path = chunkdata.guess_phonetic_segments(audio_file, n_clusters=5)
    print(f"TextGrid saved to: {generated_textgrid_path}")
else:
    print(f"Error: Audio file not found at {audio_file}")
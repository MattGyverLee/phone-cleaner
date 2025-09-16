#!/usr/bin/env python3
"""
Audio Clip Classification Tool
A web interface for classifying phoneme audio clips by position (initial, medial, final, other)
"""

import os
import re
import shutil
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_file, url_for
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Configuration
SOURCE_DIR = r"D:\Github\phone-cleaner\phoneme-Samples\Glossika\tight-snips"
OUTPUT_DIR = r"D:\Github\phone-cleaner\phoneme-Samples\Glossika\classified"
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# Create static directory for temporary files
os.makedirs(STATIC_DIR, exist_ok=True)

class ClipClassifier:
    def __init__(self):
        self.groups = {}
        self.group_names = []
        self.current_group_index = 0
        self.classifications = {}
        self._load_groups()

    def _load_groups(self):
        """Group audio files by phoneme (everything before '_clip_')"""
        if not os.path.exists(SOURCE_DIR):
            print(f"Source directory not found: {SOURCE_DIR}")
            return

        grouped_files = defaultdict(list)

        for filename in os.listdir(SOURCE_DIR):
            if filename.endswith('.wav'):
                # Extract phoneme part (everything before '_clip_')
                match = re.match(r'(.+)_clip_\d+\.wav$', filename)
                if match:
                    phoneme = match.group(1)
                    grouped_files[phoneme].append(filename)

        # Sort files within each group
        for phoneme in grouped_files:
            grouped_files[phoneme].sort()

        self.groups = dict(grouped_files)
        self.group_names = sorted(self.groups.keys())
        print(f"Loaded {len(self.group_names)} phoneme groups")

    def get_current_group(self):
        """Get current group of files"""
        if not self.group_names:
            return None, []

        group_name = self.group_names[self.current_group_index]
        files = self.groups[group_name]
        return group_name, files

    def next_group(self):
        """Move to next group"""
        if self.current_group_index < len(self.group_names) - 1:
            self.current_group_index += 1
            return True
        return False

    def previous_group(self):
        """Move to previous group"""
        if self.current_group_index > 0:
            self.current_group_index -= 1
            return True
        return False

    def generate_waveform(self, filename):
        """Generate waveform plot for audio file"""
        filepath = os.path.join(SOURCE_DIR, filename)

        try:
            # Load audio file
            y, sr = librosa.load(filepath)

            # Create waveform plot
            plt.figure(figsize=(12, 4))
            plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
            plt.title(f'Waveform: {filename}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)

            # Save to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            return f"data:image/png;base64,{image_base64}"

        except Exception as e:
            print(f"Error generating waveform for {filename}: {e}")
            return None

    def classify_file(self, filename, classification):
        """Store classification for a file"""
        self.classifications[filename] = classification

    def save_classified_files(self):
        """Copy classified files to output directory with prefixes"""
        if not self.classifications:
            return 0

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        copied_count = 0

        for filename, classification in self.classifications.items():
            source_path = os.path.join(SOURCE_DIR, filename)
            new_filename = f"{classification}_{filename}"
            dest_path = os.path.join(OUTPUT_DIR, new_filename)

            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

        # Clear classifications after saving
        self.classifications.clear()
        return copied_count

# Global classifier instance
classifier = ClipClassifier()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/current-group')
def get_current_group():
    """Get current group information"""
    group_name, files = classifier.get_current_group()

    if not group_name:
        return jsonify({'error': 'No groups available'})

    # Generate waveforms for all files in the group
    waveforms = {}
    for filename in files:
        waveform = classifier.generate_waveform(filename)
        if waveform:
            waveforms[filename] = waveform

    return jsonify({
        'group_name': group_name,
        'files': files,
        'waveforms': waveforms,
        'group_index': classifier.current_group_index,
        'total_groups': len(classifier.group_names),
        'classifications': classifier.classifications
    })

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a file"""
    data = request.json
    filename = data.get('filename')
    classification = data.get('classification')

    if not filename or not classification:
        return jsonify({'error': 'Missing filename or classification'})

    if classification not in ['initial', 'medial', 'final', 'other']:
        return jsonify({'error': 'Invalid classification'})

    classifier.classify_file(filename, classification)
    return jsonify({'success': True})

@app.route('/api/next-group', methods=['POST'])
def next_group():
    """Move to next group"""
    # Save current classifications first
    saved_count = classifier.save_classified_files()

    if classifier.next_group():
        return jsonify({
            'success': True,
            'saved_files': saved_count
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No more groups',
            'saved_files': saved_count
        })

@app.route('/api/previous-group', methods=['POST'])
def previous_group():
    """Move to previous group"""
    if classifier.previous_group():
        return jsonify({'success': True})
    else:
        return jsonify({
            'success': False,
            'message': 'Already at first group'
        })

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve audio files"""
    filepath = os.path.join(SOURCE_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    else:
        return "File not found", 404

if __name__ == '__main__':
    print(f"Starting Clip Classifier...")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Found {len(classifier.group_names)} phoneme groups")

    app.run(debug=True, host='127.0.0.1', port=5000)
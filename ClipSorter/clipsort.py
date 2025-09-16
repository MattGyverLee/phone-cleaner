#!/usr/bin/env python3
"""
Audio Clip Classification Tool
A web interface for classifying phoneme audio clips by position (initial, medial, final, other)
"""

import os
import re
import shutil
import json
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
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), 'classification_progress.json')

# Create static directory for temporary files
os.makedirs(STATIC_DIR, exist_ok=True)

class ClipClassifier:
    def __init__(self):
        self.groups = {}
        self.group_names = []
        self.current_group_index = 0
        self.classifications = {}
        self.completed_groups = set()
        self._load_progress()
        self._load_groups()
        self._find_next_incomplete_group()

    def _load_progress(self):
        """Load completed groups from progress file"""
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
                    self.completed_groups = set(data.get('completed_groups', []))
                    print(f"Loaded progress: {len(self.completed_groups)} completed groups")
        except Exception as e:
            print(f"Error loading progress: {e}")
            self.completed_groups = set()

    def _save_progress(self):
        """Save completed groups to progress file"""
        try:
            data = {
                'completed_groups': list(self.completed_groups)
            }
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving progress: {e}")

    def _find_next_incomplete_group(self):
        """Find the first incomplete group and set current index"""
        for i, group_name in enumerate(self.group_names):
            if group_name not in self.completed_groups:
                self.current_group_index = i
                return
        # If all groups are completed, stay at current index
        print("All groups completed!")

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
        print(f"Completed groups: {len(self.completed_groups)}")

    def get_current_group(self):
        """Get current group of files"""
        if not self.group_names:
            return None, []

        group_name = self.group_names[self.current_group_index]
        files = self.groups[group_name]
        return group_name, files

    def next_group(self):
        """Move to next incomplete group"""
        start_index = self.current_group_index + 1
        for i in range(start_index, len(self.group_names)):
            if self.group_names[i] not in self.completed_groups:
                self.current_group_index = i
                return True
        return False

    def previous_group(self):
        """Move to previous incomplete group"""
        start_index = self.current_group_index - 1
        for i in range(start_index, -1, -1):
            if self.group_names[i] not in self.completed_groups:
                self.current_group_index = i
                return True
        return False

    def generate_waveform(self, filename):
        """Generate waveform plot for audio file"""
        filepath = os.path.join(SOURCE_DIR, filename)

        try:
            # Load audio file
            y, sr = librosa.load(filepath)

            # Create waveform plot (smaller for compact layout)
            plt.figure(figsize=(4, 2))
            plt.plot(np.linspace(0, len(y)/sr, len(y)), y)
            plt.title(filename, fontsize=8)
            plt.xlabel('Time (s)', fontsize=8)
            plt.ylabel('Amplitude', fontsize=8)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.ylim(-0.8, 0.8)  # Set y-axis limits to ±0.8
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
        current_group_name, _ = self.get_current_group()

        for filename, classification in self.classifications.items():
            # Skip files classified as "ignore"
            if classification == 'ignore':
                continue

            source_path = os.path.join(SOURCE_DIR, filename)
            new_filename = f"{classification}_{filename}"
            dest_path = os.path.join(OUTPUT_DIR, new_filename)

            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {filename}: {e}")

        # Mark current group as completed if any files were processed (including ignored ones)
        if current_group_name and len(self.classifications) > 0:
            self.completed_groups.add(current_group_name)
            self._save_progress()
            print(f"Marked group '{current_group_name}' as completed")

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

    remaining_groups = len(classifier.group_names) - len(classifier.completed_groups)
    completed_in_total = len(classifier.completed_groups)

    return jsonify({
        'group_name': group_name,
        'files': files,
        'waveforms': waveforms,
        'group_index': classifier.current_group_index,
        'total_groups': len(classifier.group_names),
        'remaining_groups': remaining_groups,
        'completed_groups': completed_in_total,
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

    if classification not in ['initial', 'medial', 'final', 'other', 'ignore']:
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
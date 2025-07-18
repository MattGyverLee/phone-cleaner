import os
import subprocess

def convert_webm_to_wav(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .webm files in the input folder
    webm_files = [f for f in os.listdir(input_folder) if f.endswith('.webm')]

    # Iterate over each .webm file and convert to WAV
    for webm_file in webm_files:
        input_path = os.path.join(input_folder, webm_file)
        output_file = os.path.splitext(webm_file)[0] + '.wav'
        output_path = os.path.join(output_folder, output_file)

        # Ensure paths are correctly formatted for subprocess
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)

        # FFMPEG command for conversion
        command = [
            'ffmpeg',
            '-v', 'verbose',  # Add verbose output for debugging
            '-i', input_path,
            '-acodec', 'pcm_s16le',  # 16-bit little-endian PCM audio
            '-ar', '44100',
            '-f', 'wav',
            output_path
        ]

        # Print command for debugging
        print(f"Running command: {' '.join(command)}")

        # Execute the FFMPEG command
        result = subprocess.run(command, cwd=input_folder, capture_output=True, text=True)


        # Print stdout and stderr for debugging
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)

        # Check if the output file was created successfully
        if os.path.exists(output_path):
            print(f'Conversion complete: {webm_file} -> {output_file}')
        else:
            print(f'Error in conversion: {webm_file}')

if __name__ == "__main__":
    input_folder = "C:\\Github\\phone-cleaner\\phoneme-Samples\\Glossika\\webm-audio\\"
    output_folder = "C:\\Github\\phone-cleaner\\phoneme-Samples\\Glossika\\wav-audio\\"

    convert_webm_to_wav(input_folder, output_folder)

import os
import subprocess

def convert_webm_to_wav(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .webm files in the input folder
    webm_files = [f for f in os.listdir(input_folder) if f.endswith('.webm')]

    # Iterate over each .webm file and convert to WAV
    for webm_file in webm_files:
        input_path = os.path.join(input_folder, webm_file)
        output_file = os.path.splitext(webm_file)[0] + '.wav'
        output_path = os.path.join(output_folder, output_file)

        # FFMPEG command for conversion
        command = [
            'ffmpeg',
            '-i', input_path,
            '-acodec', 'pcm_s16le',  # 16-bit little-endian PCM audio
            '-ar' , '44100',
            output_path
        ]

        # Execute the FFMPEG command
        subprocess.run(command)

        print(f'Conversion complete: {webm_file} -> {output_file}')

if __name__ == "__main__":
    input_folder = "C:\\Github\\phone-cleaner\\phoneme-Samples\\Glossika\\webm-audio\\"
    
    output_folder = "C:\\Github\\phone-cleaner\\phoneme-Samples\\Glossika\\wav-audio\\"

    convert_webm_to_wav(input_folder, output_folder)
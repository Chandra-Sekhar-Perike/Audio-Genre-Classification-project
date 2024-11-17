import os
import subprocess

# Define the directories containing audio files
genre_dirs = [
    '/home/dhruvesh/Desktop/dsp-final/genres/blues',
    '/home/dhruvesh/Desktop/dsp-final/genres/classical',
    '/home/dhruvesh/Desktop/dsp-final/genres/country',
    '/home/dhruvesh/Desktop/dsp-final/genres/disco',
    '/home/dhruvesh/Desktop/dsp-final/genres/hiphop',
    '/home/dhruvesh/Desktop/dsp-final/genres/jazz',
    '/home/dhruvesh/Desktop/dsp-final/genres/metal',
    '/home/dhruvesh/Desktop/dsp-final/genres/pop',
    '/home/dhruvesh/Desktop/dsp-final/genres/reggae',
    '/home/dhruvesh/Desktop/dsp-final/genres/rock'
]

for genre_dir in genre_dirs:
    # List files in the directory before conversion
    print(f"Contents of {genre_dir} before conversion:")
    before_conversion = os.listdir(genre_dir)
    print(before_conversion)

    # Convert .au files to .wav
    for root, _, files in os.walk(genre_dir):
        for file in files:
            if file.endswith('.au'):
                source_file = os.path.join(root, file)
                output_file = os.path.join(root, file[:-3] + "wav")
                subprocess.run(['sox', source_file, output_file], check=True)

    # Remove .au files after conversion
    for root, _, files in os.walk(genre_dir):
        for file in files:
            if file.endswith('.au'):
                os.remove(os.path.join(root, file))

    # List files in the directory after conversion
    print(f"Contents of {genre_dir} after conversion:")
    after_conversion = os.listdir(genre_dir)
    print(after_conversion)
    print("\n")

print("Conversion complete. Check respective directories.")

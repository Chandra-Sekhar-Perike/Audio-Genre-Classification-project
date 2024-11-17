import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
from utils1 import GENRE_DIR, GENRE_LIST

def extract_fft_features(wavfile):
    """Extract FFT features from a WAV file and save them as a numpy array."""
    try:
        sample_rate, song_array = wavfile.read(wavfile)
        print(f"Processing {wavfile} at {sample_rate} Hz")

        # Extract FFT for the first 30,000 samples (handles mono/stereo input)
        fft_features = abs(fft(song_array[:30000]))

        # Save FFT features to a .fft file
        base_filename = os.path.splitext(wavfile)[0]
        output_filename = base_filename + ".fft.npy"
        np.save(output_filename, fft_features)
        print(f"Saved FFT features to {output_filename}")

    except Exception as e:
        print(f"Error processing {wavfile}: {e}")

def main():
    for label, genre in enumerate(GENRE_LIST):
        genre_path = os.path.join(GENRE_DIR, genre)
        if not os.path.exists(genre_path):
            print(f"Genre directory not found: {genre_path}")
            continue

        # Use glob to list .wav files in the directory
        wav_files = glob.glob(os.path.join(genre_path, "*.wav"))

        if not wav_files:
            print(f"No .wav files found in {genre_path}")
            continue

        for wavfile in wav_files:
            extract_fft_features(wavfile)

if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
import scipy.io.wavfile
from python_speech_features import mfcc
from utils1 import GENRE_DIR, GENRE_LIST


def create_ceps(wavfile):
    """
    Computes and saves the MFCC features for a given WAV file.

    Args:
        wavfile (str): Path to the input WAV file.
    """
    try:
        sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
        print(f"Processing {wavfile} with sampling rate: {sampling_rate}")

        # Compute MFCC features
        ceps = mfcc(song_array, samplerate=sampling_rate)
        print(f"MFCC shape: {ceps.shape}")

        # Replace NaN or infinite values in the MFCC array
        ceps = np.nan_to_num(ceps, nan=0.0, posinf=0.0, neginf=0.0)

        # Save the computed MFCC features
        write_ceps(ceps, wavfile)
    except Exception as e:
        print(f"Error processing {wavfile}: {e}")


def write_ceps(ceps, wavfile):
    """
    Saves the MFCC data to a file.

    Args:
        ceps (np.ndarray): Computed MFCC features.
        wavfile (str): Path to the original WAV file.
    """
    base_wav, _ = os.path.splitext(wavfile)
    data_wav = f"{base_wav}.ceps"
    np.save(data_wav, ceps)
    print(f"Saved MFCC to {data_wav}")


def main():
    """
    Main function to compute and save MFCCs for WAV files in genre directories.
    """
    for genre in GENRE_LIST:
        genre_path = os.path.join(GENRE_DIR, genre)
        for wavfile in glob.glob(os.path.join(genre_path, "*.wav")):
            create_ceps(wavfile)


if __name__ == "__main__":
    main()

import os
import sys
import scipy.io.wavfile
import matplotlib.pyplot as plt


def plot_spectrograms(directory, output_file="Spectrogram.png"):
    """
    Reads all WAV files in a directory and plots their spectrograms.

    Args:
        directory (str): Path to the directory containing WAV files.
        output_file (str): File name for saving the spectrogram plot.
    """
    # Change to the target directory
    os.chdir(directory)

    # Collect WAV files from the directory
    wavfiles = sorted([f for f in os.listdir(directory) if f.endswith(".wav")])

    if not wavfiles:
        print("No WAV files found in the directory.")
        return

    print(f"Found {len(wavfiles)} WAV files.")

    # Prepare to store sampling rates and song arrays
    sampling_rates = []
    song_arrays = []

    # Read and store data from each WAV file
    for wavfile in wavfiles:
        try:
            sampling_rate, song_array = scipy.io.wavfile.read(wavfile)
            sampling_rates.append(sampling_rate)
            song_arrays.append(song_array)
            print(f"Loaded {wavfile} (Sampling rate: {sampling_rate} Hz)")
        except Exception as e:
            print(f"Error reading {wavfile}: {e}")
            continue

    # Plot spectrograms
    num_files = len(song_arrays)
    plt.figure(figsize=(15, 15))  # Adjust figure size to fit all subplots
    rows = cols = int(num_files**0.5) + (1 if num_files**0.5 % 1 else 0)

    for i, (song_id, song_array, sampling_rate) in enumerate(
        zip(wavfiles, song_arrays, sampling_rates), start=1
    ):
        plt.subplot(rows, cols, i)
        plt.specgram(song_array[:30000], Fs=sampling_rate, cmap="viridis")
        plt.title(song_id[:10])  # Truncate title if too long
        plt.axis("off")  # Hide axes for cleaner visuals

    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(output_file)
    plt.show()
    print(f"Spectrograms saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    if not os.path.isdir(input_dir):
        print(f"Invalid directory: {input_dir}")
        sys.exit(1)

    plot_spectrograms(input_dir)

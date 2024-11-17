import os
import numpy as np
from python_speech_features import mfcc
from sklearn.externals import joblib
import scipy.io.wavfile
from utils1 import GENRE_DIR, GENRE_LIST


def create_fft(wavfile):
    """Creates FFT features from a wav file and saves them."""
    sample_rate, song_array = scipy.io.wavfile.read(wavfile)
    fft_features = abs(np.fft.fft(song_array[:30000]))  # Use np.fft for efficiency
    base_fn, _ = os.path.splitext(wavfile)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)
    print(f"FFT features saved to: {data_fn}")
    return data_fn


def create_ceps_test(wavfile):
    """
    Creates MFCC features from a wav file, saves them, and returns the saved file name.
    """
    sample_rate, song_array = scipy.io.wavfile.read(wavfile)
    ceps = mfcc(song_array, samplerate=sample_rate)
    ceps = np.nan_to_num(ceps)  # Replace NaN or Inf values with 0
    base_fn, _ = os.path.splitext(wavfile)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print(f"MFCC features saved to: {data_fn}")
    return data_fn


def read_ceps_test(test_file):
    """
    Reads MFCC features from a file and returns them as numpy arrays.
    """
    ceps = np.load(test_file)
    num_ceps = len(ceps)
    X = [np.mean(ceps[int(num_ceps / 10):int(num_ceps * 9 / 10)], axis=0)]
    y = list(range(len(GENRE_LIST)))
    return np.array(X), np.array(y)


def test_model_on_single_file(file_path):
    """
    Tests the model on a single file and predicts its genre.
    """
    model_path = "saved_models/model_mfcc_knn.pkl"  # Path to the pre-trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    clf = joblib.load(model_path)
    test_file = create_ceps_test(file_path) + ".npy"
    X, _ = read_ceps_test(test_file)

    probs = clf.predict_proba(X)[0]
    max_prob_index = np.argmax(probs)
    predicted_genre = GENRE_LIST[max_prob_index]

    print("\nGenre Probabilities:")
    for genre, prob in zip(GENRE_LIST, probs):
        print(f"{genre}: {prob:.3f}")

    print(f"\nPredicted genre: {predicted_genre}")
    return predicted_genre


if __name__ == "__main__":
    test_file = "/home/dhruvesh/Desktop/dsp-final/country.wav"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
    else:
        predicted_genre = test_model_on_single_file(test_file)

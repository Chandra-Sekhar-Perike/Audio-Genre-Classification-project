import os
import glob
import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from utils1 import GENRE_DIR, GENRE_LIST

def load_features(file_pattern, label, feature_type):
    """Load features (FFT or MFCC) from files."""
    features, labels = [], []
    for file_path in glob.glob(file_pattern):
        data = np.load(file_path)
        if feature_type == "MFCC":
            num_samples = len(data)
            data = np.mean(data[int(num_samples * 0.1):int(num_samples * 0.9)], axis=0)
        features.append(data)
        labels.append(label)
    return features, labels

def prepare_data(genre_list, base_dir, feature_type):
    """Prepare data for training and testing."""
    X, y = [], []
    for label, genre in enumerate(genre_list):
        file_pattern = os.path.join(base_dir, genre, f"*.{feature_type.lower()}.npy")
        features, labels = load_features(file_pattern, label, feature_type)
        X.extend(features)
        y.extend(labels)
    return np.array(X), np.array(y)

def train_and_evaluate(X_train, y_train, X_test, y_test, genre_list, model_type):
    """Train and evaluate classifiers."""
    if model_type == "logistic":
        model = linear_model.LogisticRegression()
        model_name = "model_mfcc_log.pkl"
    elif model_type == "knn":
        model = KNeighborsClassifier()
        model_name = "model_mfcc_knn.pkl"
    else:
        raise ValueError("Unsupported model type!")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"{model_type.capitalize()} accuracy: {accuracy:.2f}")
    print(f"{model_type.capitalize()} confusion matrix:\n{cm}")

    joblib.dump(model, f'saved_models/{model_name}')
    plot_confusion_matrix(cm, f"{model_type.capitalize()} Confusion Matrix", genre_list)

def plot_confusion_matrix(cm, title, genre_list, cmap=plt.cm.Blues):
    """Plot a confusion matrix."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(genre_list))
    plt.xticks(tick_marks, genre_list, rotation=45)
    plt.yticks(tick_marks, genre_list)
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def main():
    base_dir = GENRE_DIR
    genre_list = ["blues", "classical", "country", "disco", "metal"]
    feature_type = "MFCC"  # Use "FFT" for FFT features

    print(f"Preparing data using {feature_type} features...")
    X, y = prepare_data(genre_list, base_dir, feature_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training and evaluating models using {feature_type} features...")
    train_and_evaluate(X_train, y_train, X_test, y_test, genre_list, "logistic")
    train_and_evaluate(X_train, y_train, X_test, y_test, genre_list, "knn")

if __name__ == "__main__":
    main()

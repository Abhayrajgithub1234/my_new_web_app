import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from skimage import feature
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Dataset path and categories
dataset_path = "dataset"
categories = ["Dasana", "Kepula", "Parijata", "Surali"]
image_size = (64, 64)
unknown_label = "Unknown"  # Label for unknown images

# Preparing data and labels
data = []
labels = []

# Load images and labels
for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append(img)
            labels.append(label)

# Convert data and labels to numpy arrays
data = np.array(data) / 255.0  # Normalize image data
labels = np.array(labels)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# HOG Feature Extraction Function
def extract_hog_features(images):
    hog_features = []
    for img in images:
        img = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

# LBP Feature Extraction Function
def extract_lbp_features(images):
    lbp_features = []
    for img in images:
        img = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(img_gray, P=8, R=1, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        hist = hist.astype("float")
        hist /= hist.sum()  # Normalize the histogram
        lbp_features.append(hist)
    return np.array(lbp_features)

# Extract HOG and LBP features for training and testing sets
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

X_train_lbp = extract_lbp_features(X_train)
X_test_lbp = extract_lbp_features(X_test)

# Combine HOG and LBP features
X_train_combined = np.hstack((X_train_hog, X_train_lbp))
X_test_combined = np.hstack((X_test_hog, X_test_lbp))

# Train the SVM model using Grid Search for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train_combined, y_train)

# Predict on the test set
y_pred = grid.predict(X_test_combined)

# Introduce unknown handling: Classify any incorrect prediction as "Unknown"
y_test_with_unknowns = []
for true_label, pred_label in zip(y_test, y_pred):
    if true_label == pred_label:
        y_test_with_unknowns.append(categories[true_label])
    else:
        y_test_with_unknowns.append(unknown_label)

# Convert y_test to string labels for consistency
y_test_strings = [categories[label] for label in y_test]

# Print Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test_strings, y_test_with_unknowns))
print("\nClassification Report:\n", classification_report(y_test_strings, y_test_with_unknowns, target_names=categories + [unknown_label]))

# Save the trained model
joblib.dump(grid.best_estimator_, 'leaf_classifier_model.pkl')
print("Model saved as 'leaf_classifier_model.pkl'")

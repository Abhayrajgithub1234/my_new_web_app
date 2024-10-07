import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.feature import hog
from skimage import feature
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Image Augmentation Functions using OpenCV
def augment_image(image):
    # Flip the image horizontally
    flipped = cv2.flip(image, 1)
    
    # Rotate the image by a random angle between -20 and 20 degrees
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

    # Adjust brightness by a factor of 0.8 to 1.2
    brightness_factor = np.random.uniform(0.8, 1.2)
    bright = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)

    # Translate image: shift horizontally and vertically by a random factor
    max_shift = 5  # max shift in pixels
    tx = np.random.randint(-max_shift, max_shift)
    ty = np.random.randint(-max_shift, max_shift)
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, translation_matrix, (w, h))

    return [flipped, rotated, bright, translated]

# Augment the training data using the OpenCV functions
data_augmented = []
labels_augmented = []

for img, label in zip(data, labels):
    augmented_images = augment_image(img)
    data_augmented.extend(augmented_images)
    labels_augmented.extend([label] * len(augmented_images))

# Combine original and augmented data
data_combined = np.concatenate((data, np.array(data_augmented)))
labels_combined = np.concatenate((labels, np.array(labels_augmented)))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_combined, labels_combined, test_size=0.2, random_state=42)

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
grid = GridSearchCV(svm.SVC(probability=True), param_grid, refit=True, verbose=2)  # Enable probability outputs
grid.fit(X_train_combined, y_train)

# Predict on the test set
y_pred = grid.predict(X_test_combined)
probabilities = grid.predict_proba(X_test_combined)

# Introduce unknown handling: Classify any sample with low confidence as "Unknown"
confidence_threshold = 0.6  # You can adjust this threshold based on experimentation
y_pred_with_unknowns = []

for prob, pred_label in zip(probabilities, y_pred):
    if np.max(prob) < confidence_threshold:  # Check if the maximum probability is below the threshold
        y_pred_with_unknowns.append(unknown_label)
    else:
        y_pred_with_unknowns.append(categories[pred_label])

# Convert true labels to category names for readability
y_test_strings = [categories[label] for label in y_test]

# Print Confusion Matrix and Classification Report
print("Confusion Matrix:\n", confusion_matrix(y_test_strings, y_pred_with_unknowns))
print("\nClassification Report:\n", classification_report(y_test_strings, y_pred_with_unknowns, target_names=categories + [unknown_label]))

# Plot the Confusion Matrix
conf_mat = confusion_matrix(y_test_strings, y_pred_with_unknowns)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories + [unknown_label], yticklabels=categories + [unknown_label])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate F1 Score
f1 = f1_score(y_test_strings, y_pred_with_unknowns, average='weighted')
print(f"Weighted F1 Score: {f1}")

# Save the trained model
joblib.dump(grid.best_estimator_, 'leaf_classifier_model_with_unknowns.pkl')
print("Model saved as 'leaf_classifier_model_with_unknowns.pkl'")

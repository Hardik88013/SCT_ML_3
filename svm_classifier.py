import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Constants
IMAGE_SIZE = (64, 64)
categories = ['cats', 'dogs']
data_dir = 'training_set'

# Load dataset
X = []
y = []

print("Loading images...")

for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)
        if img is not None:
            img_resized = cv2.resize(img, IMAGE_SIZE)
            X.append(img_resized.flatten())
            y.append(label)

print("Images loaded successfully.")

# Prepare data
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("Training model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
print("Model training complete.")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'svm_cats_dogs_model.pkl')
print("Model saved as 'svm_cats_dogs_model.pkl'")

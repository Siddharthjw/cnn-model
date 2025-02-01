import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

# Define directories
MALWARE_DIR = "malware_images/"
BENIGN_DIR = "benign_images/"

# Load images
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        if img is not None:
            images.append(img)
            labels.append(label)
    return images, labels

# Load malware & benign images
malware_images, malware_labels = load_images_from_folder(MALWARE_DIR, 1)  # Label: 1 (Malware)
benign_images, benign_labels = load_images_from_folder(BENIGN_DIR, 0)  # Label: 0 (Benign)

# Convert lists to numpy arrays
X = np.array(malware_images + benign_images) / 255.0  # Normalize (0-1)
y = np.array(malware_labels + benign_labels)

# Reshape data for CNN
X = X.reshape(X.shape[0], 128, 128, 1)  # Add channel dimension

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "iot_malware_detector.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")




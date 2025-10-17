import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -----------------------------
# Parameters
# -----------------------------
csv_file = "ckextended_processed.csv"  # <-- your CSV file name
model_save_path = "emotion_model.h5"
img_size = 48
batch_size = 32
epochs = 10

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_file)

# FER2013 format: 'emotion' + 'pixels'
# pixels: space-separated string of grayscale pixel values
X = []
for pixel_seq in df['pixels']:
    img = np.array(pixel_seq.split(), dtype='float32').reshape(img_size, img_size, 1)
    X.append(img)
X = np.array(X) / 255.0  # Normalize

y = to_categorical(df['emotion'].values)
num_classes = y.shape[1]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size
)

# -----------------------------
# Save Model
# -----------------------------
model.save(model_save_path)
print(f"Model saved at {model_save_path}")

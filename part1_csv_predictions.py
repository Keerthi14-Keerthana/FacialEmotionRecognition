import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# Parameters
# -----------------------------
csv_file = "ckextended_processed.csv"       # Your CSV file
model_file = "emotion_model.h5"    # Trained model from train_from_csv.py
img_size = 48
num_samples_to_show = 9

# Emotion labels (must match training CSV)
class_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_file)

# Convert first batch to array
X = []
y_true = []
for index, row in df.iterrows():
    img = np.array(row['pixels'].split(), dtype='float32').reshape(img_size, img_size, 1) / 255.0
    X.append(img)
    y_true.append(row['emotion'])
X = np.array(X)
y_true = np.array(y_true)

# -----------------------------
# Load Trained Model
# -----------------------------
model = load_model(model_file)

# -----------------------------
# Predict
# -----------------------------
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# Show Sample Images with Predictions
# -----------------------------
plt.figure(figsize=(12,6))
for i in range(num_samples_to_show):
    plt.subplot(3,3,i+1)
    plt.imshow(X[i].reshape(img_size,img_size), cmap='gray')
    plt.title(f"True: {class_labels[y_true[i]]}\nPred: {class_labels[y_pred[i]]}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# Summary
# -----------------------------
from collections import Counter
count = Counter([class_labels[p] for p in y_pred])
print("Predicted emotion distribution:")
for e, freq in count.items():
    print(f"{e}: {freq} face(s)")

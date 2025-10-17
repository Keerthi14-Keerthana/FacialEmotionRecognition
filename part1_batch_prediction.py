import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from collections import Counter
import json

# -----------------------------
# Emotion labels (adjust if your model uses different order)
# -----------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -----------------------------
# Load the trained model
# -----------------------------
model_path = "emotion_model.h5"  # make sure this file exists in the same folder
model = load_model(model_path)

# -----------------------------
# Function to preprocess and predict emotion
# -----------------------------
def predict_emotion(face_img):
    face_resized = cv2.resize(face_img, (48, 48))
    face_array = face_resized.astype("float") / 255.0
    face_array = np.expand_dims(face_array, axis=-1)  # add channel dimension
    face_array = np.expand_dims(face_array, axis=0)   # add batch dimension
    preds = model.predict(face_array, verbose=0)
    emotion = emotion_labels[np.argmax(preds)]
    confidence = float(np.max(preds))
    return emotion, confidence

# -----------------------------
# Process uploaded images
# -----------------------------
image_dir = "uploaded_images"
os.makedirs(image_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"⚠️ No images found in '{image_dir}'. Please add images to test.")
    exit()

all_emotions = []
predictions = {}

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces using OpenCV Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    img_emotions = []
    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_img)
        img_emotions.append(emotion)

        # Draw bounding box & label
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    all_emotions.extend(img_emotions)
    predictions[img_file] = img_emotions

    # Show image
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{img_file} - {', '.join(img_emotions) if img_emotions else 'No Face Detected'}")
    plt.axis('off')
    plt.show()

# -----------------------------
# Summary
# -----------------------------
if all_emotions:
    count = Counter(all_emotions)
    print("\n📊 Summary of emotions across all uploaded images:")
    for e, freq in count.items():
        print(f"{e}: {freq} face(s)")

    # Plot distribution
    plt.figure(figsize=(7, 4))
    plt.bar(count.keys(), count.values(), color='skyblue')
    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

    # Save predictions as JSON
    with open("predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)
    print("\n✅ Predictions saved to predictions.json")
else:
    print("⚠️ No emotions detected.")

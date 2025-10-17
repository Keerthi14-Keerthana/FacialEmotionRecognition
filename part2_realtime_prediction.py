import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from collections import Counter
import time

# -----------------------------
# Define emotion labels
# -----------------------------
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# -----------------------------
# Load trained model
# -----------------------------
model_path = 'emotion_model.h5'  # Replace with your trained model
model = load_model(model_path)

# -----------------------------
# Initialize face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))

# -----------------------------
# Function to predict emotion
# -----------------------------
def predict_emotion(face_img):
    face_resized = cv2.resize(face_img, (48,48))
    face_array = face_resized.astype("float") / 255.0
    face_array = np.expand_dims(face_array, axis=-1)  # grayscale channel
    face_array = np.expand_dims(face_array, axis=0)   # batch dimension
    preds = model.predict(face_array, verbose=0)
    emotion = emotion_labels[np.argmax(preds)]
    confidence = np.max(preds)
    return emotion, confidence

# -----------------------------
# Choose mode: 'upload' or 'webcam'
# -----------------------------
mode = input("Enter mode ('upload' or 'webcam'): ").strip().lower()

if mode == 'upload':
    # -----------------------------
    # Batch Upload
    # -----------------------------
    image_dir = "uploaded_images"
    os.makedirs(image_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    if not image_files:
        raise Exception(f"No images found in {image_dir} folder!")

    all_emotions = []
    fig, axes = plt.subplots(nrows=(len(image_files)+2)//3, ncols=3, figsize=(15,5*((len(image_files)+2)//3)))
    axes = axes.flatten()

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        img_emotions = []
        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(face_img)
            img_emotions.append(emotion)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        all_emotions.extend(img_emotions)
        axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"{img_file} - Detected: {', '.join(img_emotions)}", fontsize=10)
        axes[idx].axis('off')

    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

    # Summary
    if all_emotions:
        count = Counter(all_emotions)
        print("Summary of emotions across uploaded images:")
        for e, freq in count.items():
            print(f"{e}: {freq} face(s)")

        plt.figure(figsize=(8,4))
        plt.bar(count.keys(), count.values(), color='skyblue')
        plt.title("Emotion Distribution Across Images")
        plt.xlabel("Emotion")
        plt.ylabel("Number of Faces")
        plt.show()

elif mode == 'webcam':
    # -----------------------------
    # Real-time Webcam
    # -----------------------------
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open camera. Check index or drivers.")

    print("Press 'q' to quit, 'c' to capture frame.")

    log_emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x,y,w,h) in faces:
            face_img = gray[y:y+h, x:x+w]
            emotion, confidence = predict_emotion(face_img)
            log_emotions.append(emotion)

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("Live Emotion Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(f"captured_frame_{int(time.time())}.jpg", frame)
            print("Captured current frame!")

    cap.release()
    cv2.destroyAllWindows()

    if log_emotions:
        count = Counter(log_emotions)
        print("Summary of emotions detected in live capture:")
        for e, freq in count.items():
            print(f"{e}: {freq} face(s)")

        plt.figure(figsize=(8,4))
        plt.bar(count.keys(), count.values(), color='lightgreen')
        plt.title("Emotion Distribution (Live Capture)")
        plt.xlabel("Emotion")
        plt.ylabel("Number of Faces")
        plt.show()

else:
    print("Invalid mode! Choose 'upload' or 'webcam'.")

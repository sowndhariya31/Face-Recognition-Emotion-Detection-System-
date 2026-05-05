import streamlit as st
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from fer import FER
from PIL import Image
from io import BytesIO
import os

st.set_page_config(page_title="Live Face Recognition + Age/Emotion", layout="wide")

# -------- TRAINING DATA PATH --------
train_path = r"C:\dipproject\DIP Project\train"

known_names = []
known_encodings = []

# -------- LOAD TRAIN IMAGES --------
if os.path.exists(train_path):
    for file in os.listdir(train_path):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(train_path, file)
            img = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(img)
            if len(enc) > 0:
                known_encodings.append(enc[0])
                known_names.append(os.path.splitext(file)[0].capitalize())
else:
    st.error(f"Training folder not found: {train_path}")

if known_names:
    st.success("Loaded known faces: " + ", ".join(known_names))
else:
    st.info("No known faces found in train folder. All faces will show as 'Unknown'.")

st.title("📹 Live Face Recognition + Age & Emotion")

# -------- HELPER: Robust age + emotion --------
fer_detector = FER(mtcnn=True)

def analyze_face_robust(face_rgb: np.ndarray):
    """Predict age using DeepFace and emotion using FER."""
    face_rgb = face_rgb.astype("uint8")
    
    # --- Age prediction ---
    try:
        res = DeepFace.analyze(img_path=face_rgb, actions=["age"], detector_backend="skip", enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
        age_out = int(res.get("age", 0))
        if 15 <= age_out <= 40:
            age_out -= 4  # calibration
    except:
        age_out = "N/A"
    
    # --- Emotion prediction ---
    try:
        results = fer_detector.detect_emotions(face_rgb)
        if results:
            results = sorted(results, key=lambda x: x["box"][2]*x["box"][3], reverse=True)
            emotions = results[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            emotion_out = dominant.capitalize()
        else:
            emotion_out = "Neutral"
    except:
        emotion_out = "Neutral"
    
    return age_out, emotion_out

# -------- STREAMLIT WEBCAM --------
run_live = st.checkbox("Activate Live Camera")

FRAME_WINDOW = st.image([])

if run_live:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot access camera")
    else:
        while run_live:
            ret, frame = cap.read()
            if not ret:
                st.warning("Unable to read frame")
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Face recognition
                name = "Unknown"
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_idx = np.argmin(distances)
                    if matches[best_idx]:
                        name = known_names[best_idx]

                # Crop face for age/emotion
                face_crop = rgb_frame[top:bottom, left:right]
                if face_crop.size > 0:
                    age, emotion = analyze_face_robust(face_crop)
                else:
                    age, emotion = "N/A", "Neutral"

                # Draw results
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"{emotion}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

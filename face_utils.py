import os
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from fer import FER
from PIL import Image

# -------- LOAD TRAINING FACES --------
def load_known_faces(train_path):
    known_encodings = []
    known_names = []

    if os.path.exists(train_path):
        for file in os.listdir(train_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(train_path, file)
                img = face_recognition.load_image_file(img_path)
                enc = face_recognition.face_encodings(img)
                if len(enc) > 0:
                    known_encodings.append(enc[0])
                    known_names.append(os.path.splitext(file)[0].capitalize())
    return known_encodings, known_names


# -------- AGE + EMOTION ANALYSIS --------
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
            age_out -= 4
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


# -------- FACE RECOGNITION --------
def recognize_face(face_rgb, known_encodings, known_names, box):
    name = "Unknown"
    try:
        fr_encodings = face_recognition.face_encodings(face_rgb, [box])
        if len(fr_encodings) > 0 and len(known_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, fr_encodings[0], tolerance=0.55)
            distances = face_recognition.face_distance(known_encodings, fr_encodings[0])
            best_idx = int(np.argmin(distances))
            if matches[best_idx]:
                name = known_names[best_idx]
    except:
        pass
    return name

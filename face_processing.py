# face_processing.py

import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
from fer import FER

fer_detector = FER(mtcnn=True)

known_names = []
known_encodings = []

def load_known_faces(train_path):
    import os
    global known_names, known_encodings
    known_names = []
    known_encodings = []

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
        raise FileNotFoundError(f"Training folder not found: {train_path}")
    return known_names

def analyze_face_robust(face_rgb: np.ndarray):
    face_rgb = face_rgb.astype("uint8")

    # --- Age prediction ---
    patches = [face_rgb]
    for size in [180, 224, 300]:
        try:
            resized = cv2.resize(face_rgb, (size, size))
            patches.append(resized)
        except:
            pass

    weights = [0.6, 0.15, 0.15, 0.1]
    ages = []

    for idx, patch in enumerate(patches):
        try:
            res = DeepFace.analyze(
                img_path=patch,
                actions=["age"],
                detector_backend="skip",
                enforce_detection=False
            )
            if isinstance(res, list):
                res = res[0]
            age_val = res.get("age", None)
            if isinstance(age_val, (int, float)):
                ages.append(float(age_val) * weights[idx])
        except:
            continue

    if ages:
        avg_age = sum(ages) / sum(weights[:len(ages)])
        if 15 <= avg_age <= 40:
            avg_age -= 4
        if avg_age < 1:
            avg_age = 1
        age_out = int(round(avg_age))
    else:
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


def detect_and_recognize_faces(rgb_img):
    import numpy as np
    import face_recognition
    from deepface import DeepFace

    results_list = []

    try:
        detected_faces = DeepFace.extract_faces(
            img_path=rgb_img,
            detector_backend="mtcnn",
            enforce_detection=False
        )
    except:
        detected_faces = []

    for det in detected_faces:
        region = det.get("facial_area", {})
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        y1, y2 = max(0, y), min(rgb_img.shape[0], y + h)
        x1, x2 = max(0, x), min(rgb_img.shape[1], x + w)
        face_crop = rgb_img[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        age, emotion = analyze_face_robust(face_crop)

        # Face recognition
        name = "Unknown"
        try:
            fr_encodings = face_recognition.face_encodings(rgb_img, [(y1, x2, y2, x1)])
            if len(fr_encodings) > 0 and len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, fr_encodings[0], tolerance=0.55)
                distances = face_recognition.face_distance(known_encodings, fr_encodings[0])
                best_idx = int(np.argmin(distances))
                if matches[best_idx]:
                    name = known_names[best_idx]
        except:
            pass

        results_list.append({
            "box": (x1, y1, x2, y2),
            "name": name,
            "age": age,
            "emotion": emotion
        })

    return results_list

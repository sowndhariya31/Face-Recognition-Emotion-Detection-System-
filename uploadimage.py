import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
from deepface import DeepFace
from io import BytesIO
from fer import FER  # New library for accurate emotion detection

st.set_page_config(page_title="Face Recognition + Age/Emotion", layout="wide")

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

st.title("📷 Face Recognition + Optimized Age & Emotion")

# -------- HELPER: ROBUST AGE + FER EMOTION --------
fer_detector = FER(mtcnn=True)  # Initialize FER once

def analyze_face_robust(face_rgb: np.ndarray):
    """
    Predict age using DeepFace, emotion using FER (more accurate)
    """
    face_rgb = face_rgb.astype("uint8")
    h, w = face_rgb.shape[:2]

    # --- Age prediction (DeepFace, robust) ---
    patches = [face_rgb]
    for size in [180, 224, 300]:  # Include larger size
        try:
            resized = cv2.resize(face_rgb, (size, size))
            patches.append(resized)
        except:
            pass

    weights = [0.6, 0.15, 0.15, 0.1]  # Weighted average
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
            avg_age -= 4  # Calibration
        if avg_age < 1:
            avg_age = 1
        age_out = int(round(avg_age))
    else:
        age_out = "N/A"

    # --- Emotion prediction (FER) ---
    try:
        results = fer_detector.detect_emotions(face_rgb)
        if results:
            # Pick largest detected face
            results = sorted(results, key=lambda x: x["box"][2]*x["box"][3], reverse=True)
            emotions = results[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            emotion_out = dominant.capitalize()
        else:
            emotion_out = "Neutral"
    except:
        emotion_out = "Neutral"

    return age_out, emotion_out


# -------- FILE UPLOAD --------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    rgb_img = np.array(pil_img)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    st.write("Detecting faces and analyzing attributes...")

    # -------- FACE DETECTION (MTCNN) --------
    try:
        detected_faces = DeepFace.extract_faces(
            img_path=rgb_img,
            detector_backend="mtcnn",
            enforce_detection=False
        )
    except Exception as e:
        st.error(f"Face detection failed: {e}")
        detected_faces = []

    if len(detected_faces) == 0:
        st.error("No faces detected in this image. Try a clearer photo.")
    else:
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

            # ---- Age & Emotion ----
            age, emotion = analyze_face_robust(face_crop)

            # ---- Face Recognition ----
            name = "Unknown"
            try:
                fr_encodings = face_recognition.face_encodings(
                    rgb_img,
                    [(y1, x2, y2, x1)]
                )
                if len(fr_encodings) > 0 and len(known_encodings) > 0:
                    matches = face_recognition.compare_faces(
                        known_encodings, fr_encodings[0], tolerance=0.55
                    )
                    distances = face_recognition.face_distance(
                        known_encodings, fr_encodings[0]
                    )
                    best_idx = int(np.argmin(distances))
                    if matches[best_idx]:
                        name = known_names[best_idx]
            except:
                pass

            # ---- Draw results ----
            cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                bgr_img, name,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7,
                (0, 255, 255), 2
            )

            cv2.putText(
                bgr_img, f"Age: {age}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (0, 255, 255), 2
            )

            cv2.putText(
                bgr_img, f"{emotion}",
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (0, 255, 255), 2
            )

        # -------- SHOW & DOWNLOAD RESULT --------
        final_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        st.image(final_rgb, width=900, caption="Result")

        buf = BytesIO()
        Image.fromarray(final_rgb).save(buf, format="PNG")
        st.download_button(
            "Download Result Image",
            buf.getvalue(),
            "output.png",
            "image/png"
        )
else:
    st.info("Upload a clear photo with visible faces to begin.")

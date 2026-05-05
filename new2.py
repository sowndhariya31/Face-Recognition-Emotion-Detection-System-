import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
from PIL import Image
from deepface import DeepFace
from io import BytesIO

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

# -------- HELPER: ROBUST AGE + EMOTION --------
def analyze_face_robust(face_rgb: np.ndarray):
    """
    Run DeepFace age + emotion multiple times on slightly
    different versions of the same face and average the results.
    This greatly stabilizes predictions.
    """

    # Ensure uint8 RGB
    face_rgb = face_rgb.astype("uint8")

    # Create two versions: original + resized
    h, w = face_rgb.shape[:2]
    patches = [face_rgb]

    # Resize to 224x224 (common CNN input size) for more consistency
    try:
        resized = cv2.resize(face_rgb, (224, 224))
        patches.append(resized)
    except Exception:
        pass

    ages = []
    emotion_accumulator = {}

    for patch in patches:
        try:
            res = DeepFace.analyze(
                img_path=patch,
                actions=["age", "emotion"],
                detector_backend="skip",   # we already cropped the face
                enforce_detection=False
            )
            if isinstance(res, list):
                res = res[0]
        except Exception:
            continue

        # ---- Age ----
        age_val = res.get("age", None)
        if isinstance(age_val, (int, float)):
            ages.append(float(age_val))

        # ---- Emotion (probabilities dict) ----
        emo_dict = res.get("emotion", {})
        if isinstance(emo_dict, dict):
            for k, v in emo_dict.items():
                try:
                    emotion_accumulator[k] = emotion_accumulator.get(k, 0.0) + float(v)
                except Exception:
                    continue

    # Final Age
    if ages:
        avg_age = int(round(sum(ages) / len(ages)))
        # Light calibration: DeepFace overestimates young adult ages a bit
        if 15 <= avg_age <= 45:
            avg_age -= 2
        if avg_age < 1:
            avg_age = 1
        age_out = avg_age
    else:
        age_out = "N/A"

    # Final Emotion (highest accumulated probability)
    if emotion_accumulator:
        dominant_emotion = max(emotion_accumulator, key=emotion_accumulator.get)
    else:
        dominant_emotion = "N/A"

    return age_out, dominant_emotion

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
            # MTCNN output: dict with 'facial_area' and 'face'
            region = det.get("facial_area", {})
            x = int(region.get("x", 0))
            y = int(region.get("y", 0))
            w = int(region.get("w", 0))
            h = int(region.get("h", 0))

            if w <= 0 or h <= 0:
                continue

            # Crop face safely
            y1, y2 = max(0, y), min(rgb_img.shape[0], y + h)
            x1, x2 = max(0, x), min(rgb_img.shape[1], x + w)
            face_crop = rgb_img[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            # ---- Age & Emotion (robust) ----
            age, emotion = analyze_face_robust(face_crop)

            # ---- Face Recognition (name) ----
            name = "Unknown"
            try:
                fr_encodings = face_recognition.face_encodings(
                    rgb_img,
                    [(y1, x2, y2, x1)]  # (top, right, bottom, left)
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
            except Exception:
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
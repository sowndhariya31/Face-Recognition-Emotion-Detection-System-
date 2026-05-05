import streamlit as st
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from fer import FER
from PIL import Image
from io import BytesIO
import os

st.set_page_config(page_title="Live + Upload Face Recognition", layout="wide")

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

st.title("📹 Face Recognition + Age & Emotion")

# -------- HELPER FUNCTIONS --------
fer_detector = FER(mtcnn=True)

def analyze_face_robust(face_rgb: np.ndarray):
    """Predict age using DeepFace and emotion using FER."""
    face_rgb = face_rgb.astype("uint8")

    # Age prediction
    try:
        res = DeepFace.analyze(img_path=face_rgb, actions=["age"], detector_backend="skip", enforce_detection=False)
        if isinstance(res, list):
            res = res[0]
        age_out = int(res.get("age", 0))
        if 15 <= age_out <= 40:
            age_out -= 4
    except Exception:
        age_out = "N/A"

    # Emotion prediction
    try:
        results = fer_detector.detect_emotions(face_rgb)
        if results:
            # pick the largest detected face
            results = sorted(results, key=lambda x: x["box"][2]*x["box"][3], reverse=True)
            emotions = results[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            emotion_out = dominant.capitalize()
        else:
            emotion_out = "Neutral"
    except Exception:
        emotion_out = "Neutral"

    return age_out, emotion_out

# -------- SELECT MODE --------
mode = st.sidebar.selectbox("Mode", ["Live Camera", "Upload Image"])

# -------- LIVE CAMERA MODE --------
if mode == "Live Camera":
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

                    # Draw rectangle and text
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, emotion, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()

# -------- UPLOAD IMAGE MODE --------
elif mode == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        rgb_img = np.array(pil_img)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        st.write("Detecting faces and analyzing attributes...")

        try:
            detected_faces = DeepFace.extract_faces(
                img_path=rgb_img,
                detector_backend="mtcnn",
                enforce_detection=False
            )
        except Exception as e:
            st.error(f"Face detection failed: {e}")
            detected_faces = []

        if not detected_faces:
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

                age, emotion = analyze_face_robust(face_crop)

                # Face recognition for upload
                name = "Unknown"
                try:
                    fr_encodings = face_recognition.face_encodings(
                        rgb_img, [(y1, x2, y2, x1)]
                    )
                    if fr_encodings and known_encodings:
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

                # draw on image
                cv2.rectangle(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bgr_img, name, (x1, y1 - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(bgr_img, f"Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(bgr_img, emotion, (x1, y2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

            final_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(final_rgb, width=900, caption="Result")

            buf = BytesIO()
            Image.fromarray(final_rgb).save(buf, format="PNG")
            st.download_button("Download Result Image", buf.getvalue(), "output.png", "image/png")
    else:
        st.info("Upload a clear photo with visible faces to begin.")

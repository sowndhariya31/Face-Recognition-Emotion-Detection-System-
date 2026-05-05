import streamlit as st
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import os

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

if known_names:
    st.success("Loaded known faces: " + ", ".join(known_names))
else:
    st.warning("No known faces found. All detected faces will show Unknown.")

st.title("📸 Face Recognition + Age & Emotion")

# -------- PREDICT AGE & EMOTION (DeepFace) --------
def analyze_face(face_crop):
    try:
        res = DeepFace.analyze(
            img_path=face_crop,
            actions=["age", "emotion"],
            detector_backend="skip",
            enforce_detection=False
        )
        if isinstance(res, list):
            res = res[0]

        age = int(res.get("age", 0))
        emotion_dict = res.get("emotion", {})
        if emotion_dict:
            emotion = max(emotion_dict, key=emotion_dict.get).capitalize()
        else:
            emotion = "Neutral"

        if 15 <= age <= 40:
            age -= 3  # calibration

        return age, emotion
    except:
        return "N/A", "Neutral"

# -------- MODE SELECTION --------
mode = st.radio("Choose Mode", ["Upload Image", "Live Camera"])

# ========== MODE 1 - IMAGE UPLOAD ==========
if mode == "Upload Image":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        rgb_img = np.array(pil_img)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        st.info("Detecting faces...please wait...")

        detected_faces = DeepFace.extract_faces(
            img_path=rgb_img,
            detector_backend="mtcnn",
            enforce_detection=False
        )

        if not detected_faces:
            st.error("❌ No faces detected.")
        else:
            for face in detected_faces:
                region = face.get("facial_area", {})
                x, y, w, h = [int(region.get(v, 0)) for v in ("x", "y", "w", "h")]

                y1, y2 = max(0, y), min(rgb_img.shape[0], y + h)
                x1, x2 = max(0, x), min(rgb_img.shape[1], x + w)
                face_crop = rgb_img[y1:y2, x1:x2]

                age, emotion = analyze_face(face_crop)

                name = "Unknown"
                fr_enc = face_recognition.face_encodings(rgb_img, [(y1, x2, y2, x1)])
                if fr_enc and known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, fr_enc[0], tolerance=0.55)
                    dists = face_recognition.face_distance(known_encodings, fr_enc[0])
                    best = np.argmin(dists)
                    if matches[best]:
                        name = known_names[best]

                cv2.rectangle(bgr_img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(bgr_img, name, (x1,y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(bgr_img, f"Age: {age}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                cv2.putText(bgr_img, emotion, (x1,y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            output = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(output, width=900)
else:
    st.write("")

# ========== MODE 2 - LIVE CAMERA ==========
if mode == "Live Camera":

    run_live = st.checkbox("Activate Camera")
    FRAME_WINDOW = st.image([])

    if run_live:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera not accessible.")
        else:
            st.info("Press ❌ Stop button to close camera live feed.")

        while run_live:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                face_crop = rgb_frame[top:bottom, left:right]
                age, emotion = analyze_face(face_crop)

                name = "Unknown"
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.55)
                    dists = face_recognition.face_distance(known_encodings, enc)
                    best = np.argmin(dists)
                    if matches[best]:
                        name = known_names[best]

                cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left,top-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
                cv2.putText(frame, f"Age: {age}", (left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
                cv2.putText(frame, emotion, (left,bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
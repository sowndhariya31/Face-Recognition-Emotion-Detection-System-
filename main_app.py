import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from face_utils import load_known_faces, analyze_face_robust, recognize_face
import face_recognition
import os

st.set_page_config(page_title="Face Recognition + Age/Emotion", layout="wide")

TRAIN_PATH = r"C:\dipproject\DIP Project\train"
known_encodings, known_names = load_known_faces(TRAIN_PATH)

if known_names:
    st.sidebar.success("Loaded known faces: " + ", ".join(known_names))
else:
    st.sidebar.info("No known faces found in train folder. All faces will show as 'Unknown'.")

st.title("📷 Face Recognition + Age/Emotion")

# -------- TABS: Upload or Live --------
tab1, tab2 = st.tabs(["Upload Image", "Live Camera"])

# -------- UPLOAD IMAGE --------
with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        rgb_img = np.array(pil_img)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        st.write("Detecting faces and analyzing attributes...")
        face_locations = face_recognition.face_locations(rgb_img)

        if not face_locations:
            st.error("No faces detected in this image.")
        else:
            for (top, right, bottom, left) in face_locations:
                face_crop = rgb_img[top:bottom, left:right]
                age, emotion = analyze_face_robust(face_crop)
                name = recognize_face(rgb_img, known_encodings, known_names, (top, right, bottom, left))

                # Draw results
                cv2.rectangle(bgr_img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(bgr_img, name, (left, top - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(bgr_img, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(bgr_img, f"{emotion}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

            final_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            st.image(final_rgb, width=900, caption="Result")
            buf = BytesIO()
            Image.fromarray(final_rgb).save(buf, format="PNG")
            st.download_button("Download Result Image", buf.getvalue(), "output.png", "image/png")

# -------- LIVE CAMERA --------
with tab2:
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
                    name = "Unknown"
                    if known_encodings:
                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_idx = np.argmin(distances)
                        if matches[best_idx]:
                            name = known_names[best_idx]

                    face_crop = rgb_frame[top:bottom, left:right]
                    age, emotion = analyze_face_robust(face_crop)

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Age: {age}", (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"{emotion}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

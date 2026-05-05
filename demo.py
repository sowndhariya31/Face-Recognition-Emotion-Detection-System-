# app.py

import streamlit as st
import face_recognition as fr
import cv2
import numpy as np
import os
from PIL import Image
from deepface import DeepFace

# ---------------- CONFIG & PATHS ----------------
st.set_page_config(page_title="Face Recognition + Age/Emotion App", layout="wide")

train_path = r"C:\dipproject\DIP Project\train"

# ---------------- LOAD TRAIN IMAGES FOR FACE RECOGNITION ----------------
known_names = []
known_encodings = []

for filename in os.listdir(train_path):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    image_path = os.path.join(train_path, filename)
    train_image = fr.load_image_file(image_path)
    encodings = fr.face_encodings(train_image)
    if len(encodings) == 0:
        st.warning(f"No face found in train image `{filename}`, skipping…")
        continue
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(os.path.splitext(filename)[0].capitalize())

if known_names:
    st.success(f"Loaded known faces: {', '.join(known_names)}")
else:
    st.error("No known faces loaded. Check your train folder.")

st.title("📷 Face Recognition + Age & Emotion Analysis")

# ---------------- UPLOAD TEST IMAGE ----------------
uploaded_file = st.file_uploader("Upload a test image", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image_pil)
    test_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # For recognition
    rgb_test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    face_locations = fr.face_locations(rgb_test_image)
    face_encodings = fr.face_encodings(rgb_test_image, face_locations)

    st.write(f"Found **{len(face_locations)}** face(s) in the image.")

    # For attribute analysis (age, emotion) using DeepFace
    # DeepFace.analyze returns a list of dicts, one per face
    # actions = which attributes to predict
    analysis = DeepFace.analyze(
        img_path = np.array(image_pil),  # you can pass numpy array or path
        actions = ['age', 'emotion'],
        enforce_detection = False  # if no face, won't crash
    )

    # analysis might be a dict (if one face) or list (many faces)
    # Normalize to list
    if not isinstance(analysis, list):
        analysis = [analysis]

    # Now draw on OpenCV image
    for (top, right, bottom, left), face_encoding, face_attr in zip(face_locations, face_encodings, analysis):
        # Face recognition part
        matches = fr.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        face_distances = fr.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        # Get age and emotion from DeepFace’s output
        age = face_attr.get('age', "N/A")
        emotion = face_attr.get('dominant_emotion', "N/A")

        # Draw rectangle and text
        cv2.rectangle(test_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(test_image, f"{name}", (left, top - 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(test_image, f"Age: {age}", (left, top - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(test_image, f"Emotion: {emotion}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)

    # Convert back to RGB for streamlit display
    result_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Result", width=800)

    # Download button
    output_pil = Image.fromarray(result_image)
    buf = output_pil.tobytes()
    st.download_button(
        label="Download result image",
        data=buf,
        file_name="result_with_age_emotion.png",
        mime="image/png"
    )

else:
    st.info("Please upload a test image to detect faces, age, and emotion.")

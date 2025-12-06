import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("../model/emotion_model.h5")

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocess face
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

# UI Title
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("😊 Emotion Recognition System")
st.write("Choose input method:")

option = st.radio("Select Option", ("Upload Image", "Use Webcam"))

# ---------------- IMAGE UPLOAD ----------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a facial image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')
        img_array = np.array(image)

        face = preprocess_face(img_array)
        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted Emotion: **{emotion.upper()}**")

# ---------------- WEBCAM ----------------
if option == "Use Webcam":
    st.warning("Click Start Camera and press Stop to exit")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    while run:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_input = preprocess_face(face)
            prediction = model.predict(face_input)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    camera.release()

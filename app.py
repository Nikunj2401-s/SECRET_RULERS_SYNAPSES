import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms

# Load the model
model = torch.load("emotion_model.pth", map_location=torch.device('cpu'))
model.eval()

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Emotion → music links
music_links = {
    'Happy': [
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "https://www.youtube.com/watch?v=cmSbXsFE3l8"
    ],
    'Sad': [
        "https://www.youtube.com/watch?v=ho9rZjlsyYY",
        "https://www.youtube.com/watch?v=4N3N1MlvVc4"
    ],
    'Angry': [
        "https://www.youtube.com/watch?v=04F4xlWSFh0",
        "https://www.youtube.com/watch?v=sO5APfKnR50"
    ],
    'Surprise': [
        "https://www.youtube.com/watch?v=5qap5aO4i9A",
        "https://www.youtube.com/watch?v=lTRiuFIWV54"
    ],
    'Fear': [
        "https://www.youtube.com/watch?v=FYgM3j1ZxPU",
        "https://www.youtube.com/watch?v=Lr31Nn8YDUQ"
    ],
    'Disgust': [
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
        "https://www.youtube.com/watch?v=kXYiU_JCYtU"
    ],
    'Neutral': [
        "https://www.youtube.com/watch?v=ktvTqknDobU",
        "https://www.youtube.com/watch?v=JGwWNGJdvx8"
    ]
}

# UI
st.title("🎶 EMOFLOW - Emotion-Based Music Recommender")
st.write("Capture your mood, and EMOFLOW suggests music that fits it!")

# Webcam input
img = st.camera_input("Take a picture")

if img is not None:
    try:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("No face detected. Please try again with a clearer image.")
        else:
            for (x, y, w, h) in faces:
                roi = frame[y:y + h, x:x + w]
                roi_tensor = transform(roi).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(roi_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion = EMOTIONS[predicted.item()]

                st.success(f"Detected Emotion: **{emotion}**")
                st.subheader("Music suggestions:")
                for link in music_links[emotion]:
                    st.markdown(f"- [🎵 Listen on YouTube]({link})")

                break
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

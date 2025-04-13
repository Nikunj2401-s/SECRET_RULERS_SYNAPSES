import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# Define model structure
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 24 * 24, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 24 * 24)
        x = self.fc1(x)
        return x

# Load model
model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device('cpu')))
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

# YouTube links
music_links = {
    'Happy': ["https://www.youtube.com/watch?v=ZbZSe6N_BXs"],
    'Sad': ["https://www.youtube.com/watch?v=4N3N1MlvVc4"],
    'Angry': ["https://www.youtube.com/watch?v=sO5APfKnR50"],
    'Surprise': ["https://www.youtube.com/watch?v=lTRiuFIWV54"],
    'Fear': ["https://www.youtube.com/watch?v=FYgM3j1ZxPU"],
    'Disgust': ["https://www.youtube.com/watch?v=2Vv-BfVoq4g"],
    'Neutral': ["https://www.youtube.com/watch?v=ktvTqknDobU"]
}

# UI
st.title("🎶 EMOFLOW - Emotion-Based Music Recommender")
st.write("Take a picture, and get a music suggestion based on your detected emotion.")

img = st.camera_input("📷 Capture your face")

if img is not None:
    try:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

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
                st.subheader("🎵 Suggested Music:")
                for link in music_links[emotion]:
                    st.markdown(f"- [🎧 Play]({link})")
                break
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

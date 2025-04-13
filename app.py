import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the model structure
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
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

# Transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

# YouTube music links by mood
mood_music = {
    "Happy": "https://www.youtube.com/results?search_query=happy+songs+playlist",
    "Sad": "https://www.youtube.com/results?search_query=sad+songs+playlist",
    "Angry": "https://www.youtube.com/results?search_query=calm+down+music",
    "Neutral": "https://www.youtube.com/results?search_query=lofi+chill+beats",
    "Fear": "https://www.youtube.com/results?search_query=relaxing+songs",
    "Surprise": "https://www.youtube.com/results?search_query=feel+good+music",
    "Disgust": "https://www.youtube.com/results?search_query=uplifting+songs",
}

# Streamlit UI
st.set_page_config(page_title="EmoFlow", page_icon="🎧", layout="centered")
st.title("🎧 EmoFlow: Emotion-Based Music Recommender")
st.markdown("""
Welcome to **EmoFlow**! We detect your facial emotion using your camera and recommend a personalized music playlist to match your mood.
""")

st.markdown("---")

use_camera = st.button("📸 Capture from Camera")

if use_camera:
    picture = st.camera_input("Take a selfie")

    if picture is not None:
        file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("😕 No face detected in the photo. Try again with a clearer selfie.")
        else:
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                face_tensor = transform(face).unsqueeze(0)
                output = model(face_tensor)
                _, predicted = torch.max(output, 1)
                emotion = EMOTIONS[predicted.item()]

                st.image(face, caption=f"Detected Emotion: {emotion}", channels="BGR")
                st.success(f"Emotion: **{emotion}**")

                confirm = st.radio("Do you want to continue with this mood?", ["Yes", "No"])
                if confirm == "Yes":
                    st.markdown(f"[🎵 Open Music for {emotion} Mood]({mood_music[emotion]})", unsafe_allow_html=True)
                break

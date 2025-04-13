import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import webbrowser
from torchvision import transforms

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the model structure
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 24 * 24, 7)  # 48x48 input → down to 24x24 → flat = 4608

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

# Streamlit app
st.title("EmoFlow: Mood-based YouTube Music Recommender")

if st.button("Detect Emotion from Camera"):
    cap = cv2.VideoCapture(0)
    st.write("Camera opened. Please look into the camera...")
    
    detected = False
    while not detected:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_tensor = transform(face).unsqueeze(0)
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
            emotion = EMOTIONS[predicted.item()]
            st.success(f"Detected Emotion: **{emotion}**")
            detected = True
            break

    cap.release()
    #cv2.destroyAllWindows()

    # Ask user to confirm
    if detected:
        confirm = st.radio("Do you want to continue with this mood?", ["Yes", "No"])
        if confirm == "Yes":
            st.write(f"Opening YouTube music for mood: {emotion}")
            webbrowser.open(mood_music[emotion])
        else:
            st.warning("Please re-run detection to try again.")

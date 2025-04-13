import torch
import torch.nn as nn

# Define a simple CNN
class SimpleEmotionCNN(nn.Module):
    def __init__(self):
        super(SimpleEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 24 * 24, 7)  # 7 emotion classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 24 * 24)
        x = self.fc1(x)
        return x

# Create model
model = SimpleEmotionCNN()

# Save the dummy model
torch.save(model.state_dict(), "emotion_model.pth")

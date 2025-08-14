import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTM(input_size=48*8*8, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(128, 50), nn.ReLU(), nn.Linear(50, 1))

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.cnn(x)
        x = x.view(b, t, -1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1])
    
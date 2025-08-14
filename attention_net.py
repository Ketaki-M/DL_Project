import torch
import torch.nn as nn

class AttentionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU()
        )
        self.attn = nn.Linear(64 * 66 * 200, 128)
        self.fc = nn.Sequential(nn.Linear(128, 50), nn.ReLU(), nn.Linear(50, 1))

    def forward(self, x):
        features = self.cnn(x)
        flat = features.view(x.size(0), -1)
        weights = torch.sigmoid(self.attn(flat))
        return self.fc(weights)
    
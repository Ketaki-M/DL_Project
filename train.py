from models.pilotnet import PilotNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np

class DrivingDataset(Dataset):
    def __init__(self, csv_file, image_folder):
        self.data = open(csv_file).readlines()
        self.image_folder = image_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip().split(',')
        img_path = os.path.join(self.image_folder, line[0])
        angle = float(line[1])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (200, 66)) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), torch.tensor([angle], dtype=torch.float32)

def train_and_evaluate(lr, wd):
    model = PilotNet()
    dataset = DrivingDataset("data/driving_dataset/labels.csv", "data/driving_dataset/")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()

    for epoch in range(2):  # You can increase to 10
        total_loss = 0
        for img, angle in loader:
            pred = model(img)
            loss = criterion(pred, angle)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)

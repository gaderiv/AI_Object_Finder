import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from models.i3d import I3D
from models.efficientdet import EfficientDet
from data.datasets import VideoDataset
from utils.preprocessing import preprocess_frame

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

def main():
    if len(sys.argv) < 3:
        print("Usage: python train.py <data_dir> <model_save_path>")
        sys.exit(1)

    data_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    batch_size = 8
    num_epochs = 10

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))


    dataset = VideoDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = I3D().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs)
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()

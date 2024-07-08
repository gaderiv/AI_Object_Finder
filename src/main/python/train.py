import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.i3d import I3D
from models.efficientdet import EfficientDet
from data.datasets import Videodataset
from utils.preprocessing import preprocess_video

def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs} Loss: {epoch_loss:.4f}')

if __name__ == "__main__":
    data_dir = "REMEBER TO PLACE PATH LATER"
    dataset = Videodataset(data_dir, transform=preprocess_video)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = I3D(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=25)

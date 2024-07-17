import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.i3d import I3D
from data.datasets import VideoDataset
from models.efficientdet_3d import get_efficientdet_3d

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device='cuda', log_callback=None):
    model.to(device)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted.squeeze() == labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs is None or labels is None:
                    continue
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted.squeeze() == labels).sum().item()

        train_loss = train_loss / train_total if train_total > 0 else 0
        val_loss = val_loss / val_total if val_total > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        log_message = f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
        if log_callback:
            log_callback(log_message)
        else:
            print(log_message)

    return train_losses, val_losses, train_accs, val_accs

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(data_dir, model_save_path, model_type='i3d', log_callback=None, model=None):
    batch_size = 32
    num_epochs = 15
    learning_rate = 0.0001

    if not os.path.exists(os.path.dirname(model_save_path)):
        os.makedirs(os.path.dirname(model_save_path))

    train_csv = os.path.join(data_dir, 'train.csv')
    val_csv = os.path.join(data_dir, 'val.csv')

    train_dataset = VideoDataset(train_csv)
    val_dataset = VideoDataset(val_csv)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        error_msg = "No valid videos found in the dataset. Please check your video paths."
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if model is None:
        if model_type == 'i3d':
            model = I3D().cuda()
        elif model_type == 'efficientdet':
            model = get_efficientdet_3d().cuda()
        else:
            raise ValueError("Invalid model type. Choose 'i3d' or 'efficientdet'.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs, log_callback=log_callback
    )
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    torch.save(model.state_dict(), model_save_path)

    if log_callback:
        log_callback(f"Model saved to {model_save_path}")
    else:
        print(f"Model saved to {model_save_path}")
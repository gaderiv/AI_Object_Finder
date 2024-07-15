import torch
import sys
import numpy as np
import cv2
from models.i3d import I3D
from models.efficientdet import EfficientDet
from utils.preprocessing import preprocess_frame
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_model(model_path, model_type="i3d"):
    if model_type == "i3d":
        model = I3D()
    elif model_type == "efficientdet":
        model = EfficientDet()
    else:
        raise ValueError("Unknown model type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def infer_on_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        frames.append(frame)
    cap.release()

    if len(frames) < 16:
        frames += [frames[-1]] * (16 - len(frames))

    frames = np.array(frames)
    frames = torch.tensor(frames).float().unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = frames.to(device)

    with torch.no_grad():
        predictions = model(frames)

    return torch.sigmoid(predictions).cpu().numpy()

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py <model_path> <video_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_file = sys.argv[2]

    model = load_model(model_path)
    predictions = infer_on_video(model, video_file)
    
    # Assuming you have true labels for the video
    true_labels = [1]  # Replace with actual labels
    
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()
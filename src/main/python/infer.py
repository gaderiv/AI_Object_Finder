import torch
import sys
import numpy as np
import cv2
from models.i3d import I3D
from models.efficientdet import EfficientDet
from models.efficientdet_3d import get_efficientdet_3d
from utils.preprocessing import preprocess_frame
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import random

def load_model(model_path, model_type="i3d"):
    if model_type == "i3d":
        model = I3D()
    elif model_type == 'efficientdet':
            model = get_efficientdet_3d().cuda()
    else:
        raise ValueError("Unknown model type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def infer_on_video_folder(model, video_path):
    video_folder = os.path.dirname(video_path)
    video_file = os.path.basename(video_path)
    
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
    
    if not video_files:
        print(f"No video files found in folder: {video_folder}")
        return None

    # Use the specific video file if it exists, otherwise randomly select one
    if video_file in video_files:
        selected_video = video_file
    else:
        selected_video = random.choice(video_files)
    
    video_path = os.path.join(video_folder, selected_video)
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened() and len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame, frame_size=(224, 224))  # Adjust size if needed
        frames.append(frame)
    cap.release()

    if len(frames) < 16:
        frames += [frames[-1]] * (16 - len(frames))

    frames = np.array(frames)
    # Reshape to [3, 16, 224, 224]
    frames = np.transpose(frames, (1, 0, 2, 3))
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
        print("Usage: python infer.py <model_path> <test_csv>")
        sys.exit(1)

    model_path = sys.argv[1]
    test_csv = sys.argv[2]

    model = load_model(model_path)
    
    # Load labels
    df = pd.read_csv(test_csv)
    
    true_labels = []
    predicted_labels = []
    
    for _, row in df.iterrows():
        video_path = row['video_path']
        
        if not os.path.exists(os.path.dirname(video_path)):
            print(f"Video folder not found: {os.path.dirname(video_path)}")
            continue
        
        predictions = infer_on_video_folder(model, video_path)
        
        if predictions is None:
            continue
        
        predicted_label = (predictions > 0.5).astype(int)
        
        true_labels.append(row['label'])
        predicted_labels.append(predicted_label[0][0])
        
        print(f"Video Path: {video_path}, True Label: {row['label']}, Predicted: {predicted_label[0][0]}")
    
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    print(f"\nOverall Results:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    plot_confusion_matrix(conf_matrix)

if __name__ == "__main__":
    main()
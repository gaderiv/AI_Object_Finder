import torch
import sys
import numpy as np
import cv2
from models.i3d import I3D
from models.efficientdet import EfficientDet
from tqdm import tqdm

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float().to(device) / 255.0
    frames = frames.unsqueeze(0)

    with torch.no_grad():
        predictions = model(frames)

    return predictions.cpu().numpy()

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py <model_path> <video_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    video_file = sys.argv[2]

    model_type = "i3d"
    model = load_model(model_path, model_type)

    predictions = infer_on_video(model, video_file)
    print(predictions)

if __name__ == "__main__":
    main()

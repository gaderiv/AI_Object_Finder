import torch
import numpy as np
from models.i3d import I3D
from utils.preprocessing import preprocess_video
import cv2

def load_model(model_path, device):
    model = I3D(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, video_path, device):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224,224))
        frames.append(frame)

    frames = preprocess_video(np.array(frames))
    inputs = torch.from_numpy(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    cap.release()
    return preds.item()

if __name__ == "__main__":
    model_path = "path_to_trained_model.pth"
    video_path = "path_to_video.mp4"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    prediction = predict(model, video_path, device)
    print(f'Prediction: {"Theft" if prediction == 1 else "No Theft"}')

import torch
import numpy as np
import cv2
from models.i3d import I3D
from models.efficientdet import EfficientDet
from tkinter import filedialog, Tk
from tqdm import tqdm

def load_model(model_path, model_type="i3d"):
    if model_type == "i3d":
        model = I3D()
    elif model_type == "efficientdet":
        model = EfficientDet()
    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cuda()
    return model

def select_video_file():
    root = Tk()
    root.withdraw()
    video_file = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return video_file

def infer_on_video(model, video_file, frame_size=(224, 224), frame_skip=10):
    cap = cv2.VideoCapture(video_file)
    count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_skip == 0:
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        count += 1

    cap.release()
    frames = np.array(frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2).float().cuda() / 255.0
    frames = frames.unsqueeze(0)

    with torch.no_grad():
        predictions = model(frames)
    
    return predictions.cpu().numpy()

def main():
    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pth")])
    video_file = select_video_file()
    model_type = "i3d"
    model = load_model(model_path, model_type)

    predictions = infer_on_video(model, video_file)
    print(predictions)

if __name__ == "__main__":
    main()

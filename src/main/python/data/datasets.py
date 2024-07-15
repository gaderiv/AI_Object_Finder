import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.preprocessing import preprocess_frame
import pandas as pd

class VideoDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.valid_data = self.data[self.data['video_path'].apply(os.path.exists)]
        print(f"Loaded {len(self.valid_data)} valid entries out of {len(self.data)} from {csv_file}")
        
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        video_path = self.valid_data.iloc[idx]['video_path']
        label = self.valid_data.iloc[idx]['label']
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while frame_count < 16:  # Limit to 16 frames
            ret, frame = cap.read()
            if not ret:
                break
            frame = preprocess_frame(frame)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            frame_count += 1
        cap.release()
        
        if len(frames) == 0:
            print(f"No frames read from video: {video_path}")
            # Return a blank frame instead of None
            frames = [np.zeros((3, 224, 224), dtype=np.float32)] * 16
        elif len(frames) < 16:
            print(f"Padding video with {16 - len(frames)} frames: {video_path}")
            frames += [frames[-1]] * (16 - len(frames))
        
        frames = np.array(frames)
        frames = torch.from_numpy(frames).float()
        label = torch.tensor(label, dtype=torch.float32)
        
        return frames, label
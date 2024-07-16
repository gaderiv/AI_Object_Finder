import os
import random
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
        self.valid_data = self.data[self.data['video_path'].apply(lambda x: os.path.exists(os.path.dirname(x)))]
        print(f"Loaded {len(self.valid_data)} valid entries out of {len(self.data)} from {csv_file}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        video_path = self.valid_data.iloc[idx]['video_path']
        label = self.valid_data.iloc[idx]['label']
        
        video_folder = os.path.dirname(video_path)
        video_file = os.path.basename(video_path)
        
        # Get all video files in the folder
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
        
        if not video_files:
            print(f"No video files found in folder: {video_folder}")
            # Return a blank frame instead of None
            return torch.zeros((3, 16, 224, 224), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        
        # Use the specific video file if it exists, otherwise randomly select one
        if video_file in video_files:
            selected_video = video_file
        else:
            selected_video = random.choice(video_files)
        
        video_path = os.path.join(video_folder, selected_video)
        
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
            frames = np.zeros((16, 3, 224, 224), dtype=np.float32)
        elif len(frames) < 16:
            print(f"Padding video with {16 - len(frames)} frames: {video_path}")
            frames += [frames[-1]] * (16 - len(frames))
        
        frames = np.array(frames)
        # Reshape to [3, 16, 224, 224]
        frames = np.transpose(frames, (1, 0, 2, 3))
        frames = torch.from_numpy(frames).float()
        label = torch.tensor(label, dtype=torch.float32)
        
        return frames, label
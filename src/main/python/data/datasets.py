import os
from torch.utils.data import Dataset
import cv2

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.videos = []
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(".mp4"):
                    self.videos.append(os.path.join(root, file))
        self.num_samples = len(self.videos)
        print(f"Found {self.num_samples} video files.")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return frames
    
    def _get_label(self, filename):
        return 0

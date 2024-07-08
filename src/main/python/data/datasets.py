import os
import numpy as np
import torch
from torch.utils.data import Dataset

class Videodataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_data = np.load(os.path.join(self.data_dir, video_file))

        if self.transform:
            video_data = self.transform(video_data)

        label = self._get_label(video_file)
        return video_data, label
    
    def _get_label(self, filename):
        return 0

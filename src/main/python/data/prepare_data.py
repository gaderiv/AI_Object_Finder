import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_size=(224, 224), frame_skip=10):
    cap = cv2.VideoCapture(video_path)
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
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{video_name}.npy")
    np.save(output_path, frames)

def select_video_files():
    root = tk.Tk()
    root.withdraw()
    video_files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return list(video_files)

def main():
    video_files = select_video_files()
    output_dir = filedialog.askdirectory(title="Select Output Directory")
    frame_size = (224, 224)
    frame_skip = 10

    for video_file in tqdm(video_files, desc="Processing videos"):
        extract_frames(video_file, output_dir, frame_size, frame_skip)

if __name__ == "__main__":
    main()

import cv2
import numpy as np

def preprocess_frame(frame, frame_size=(128, 128)):
    frame = cv2.resize(frame, frame_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame
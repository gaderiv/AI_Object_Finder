import numpy as np
import cv2

def preprocess_frame(frame, frame_size=(224, 224)):
    frame = cv2.resize(frame, frame_size)
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    return frame

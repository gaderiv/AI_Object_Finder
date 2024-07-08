import numpy as np

def preprocess_video(frames):
    frames = frames.astype(np.float32) / 255.0
    frames = np.transpose(frames, (3,0,1,2)) #converty to c,d,h,w
    return frames
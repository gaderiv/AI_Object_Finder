import os
import cv2
import numpy as np

def prepare_data(input_dir, output_dir, img_size=(224,224)):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(filepath)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, img_size)
            frames.append(frame)

        frames = np.array(frames)
        np.save(os.path.join(output_dir, filename.split('.')[0]), frames)
        cap.release()
    print("Data preparation completed.")

if __name__ == "__main__":
    prepare_data("path_to_raw_videos", "path_to_prepared_data")

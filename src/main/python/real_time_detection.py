import torch
import cv2
import numpy as np
from models.i3d import I3D
from models.efficientdet_3d import get_efficientdet_3d
from utils.preprocessing import preprocess_frame

class RealTimeDetector:
    def __init__(self, model_path, model_type="i3d"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path, model_type)
        self.frame_buffer = []
        self.detection_threshold = 0.5
        self.frame_size = (224, 224)
        
        # Person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Tracker
        self.trackers = []
        self.track_id = 0

    def load_model(self, model_path, model_type):
        if model_type == "i3d":
            model = I3D()
        elif model_type == "efficientdet":
            model = get_efficientdet_3d()
        else:
            raise ValueError("Unknown model type")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def detect_people(self, frame):
        boxes, _ = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
        return boxes

    def update_trackers(self, frame):
        new_trackers = []
        for tracker in self.trackers:
            success, box = tracker['tracker'].update(frame)
            if success:
                new_trackers.append(tracker)
        self.trackers = new_trackers

    def add_new_trackers(self, frame, boxes):
        for box in boxes:
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, tuple(box))
            self.trackers.append({
                'id': self.track_id,
                'tracker': tracker,
                'frame_buffer': []
            })
            self.track_id += 1

    def detect(self, frame):
        original_frame = frame.copy()
        
        # Detect people
        boxes = self.detect_people(frame)
        
        # Update existing trackers
        self.update_trackers(frame)
        
        # Add new trackers for newly detected people
        self.add_new_trackers(frame, boxes)
        
        results = []
        
        for tracker in self.trackers:
            success, box = tracker['tracker'].update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                person_frame = frame[y:y+h, x:x+w]
                
                if person_frame.size == 0:
                    continue
                
                processed_frame = preprocess_frame(person_frame, self.frame_size)
                tracker['frame_buffer'].append(processed_frame)
                
                if len(tracker['frame_buffer']) == 16:
                    input_frames = np.array(tracker['frame_buffer'])
                    input_frames = np.transpose(input_frames, (1, 0, 2, 3))
                    input_tensor = torch.tensor(input_frames).float().unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        output = self.model(input_tensor)
                        prediction = torch.sigmoid(output).cpu().numpy()

                    tracker['frame_buffer'].pop(0)
                    
                    is_theft = prediction[0][0] > self.detection_threshold
                    color = (0, 0, 255) if is_theft else (0, 255, 0)
                    
                    cv2.rectangle(original_frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(original_frame, f"ID: {tracker['id']}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    results.append({
                        'id': tracker['id'],
                        'is_theft': is_theft,
                        'box': (x, y, w, h)
                    })

        return results, original_frame
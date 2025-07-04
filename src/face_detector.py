import cv2 as cv
import dlib

from .config import SHAPE_PREDICTOR_PATH, FACE_DETECTION_SCALE

class FaceDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    def detect_faces(self, frame):
        # Optimization: Resize frame for detection
        detection_frame = cv.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        rgb_detection_frame = cv.cvtColor(detection_frame, cv.COLOR_BGR2RGB)
        
        dets = self.face_detector(rgb_detection_frame)
        return dets
    
    def get_face_landmarks(self, rgb_frame, detection):
        return self.shape_predictor(rgb_frame, detection)
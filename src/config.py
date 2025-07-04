import os
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# === GLOBAL SETTINGS ===
WINDOW_TITLE = 'Face Recognition'
YAML_PATH = resource_path('database/people.yml')

# Performance settings
FRAME_SKIP = 3
MAX_THREADS = 4
THRESHOLD = 0.5
FACE_DETECTION_SCALE = 0.5
RECOGNITION_INTERVAL = 10
MAX_FACES_CACHE = 100

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# Model paths
SHAPE_PREDICTOR_PATH = resource_path('models/shape_predictor_68_face_landmarks.dat')
FACE_RECOGNITION_MODEL_PATH = resource_path('models/dlib_face_recognition_resnet_model_v1.dat')

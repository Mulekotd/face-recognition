import os

from src.common.path import resource_path

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
RECOGNITION_WORKERS = 1
FACE_DESCRIPTOR_NUM_JITTERS = 1
FACE_DESCRIPTOR_PADDING = 0.25

# Tracking settings
TRACK_MAX_MISSED = 8
TRACK_MATCH_MAX_DISTANCE = 120
TRACK_MATCH_MIN_IOU = 0.10
TRACK_SMOOTHING = 0.45
FACE_NAME_BUFFER_SIZE = 8

# Optional dlib CNN/CUDA detector. The HOG detector does not use CUDA; enable
# this only after adding mmod_human_face_detector.dat to models/ or setting the
# DLIB_CNN_FACE_DETECTOR_PATH environment variable.
USE_CUDA_FACE_DETECTOR = os.getenv('USE_CUDA_FACE_DETECTOR', '0') == '1'
DLIB_CNN_UPSAMPLE = 0
DLIB_HOG_UPSAMPLE = 0

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# Display settings
DISPLAY_HOST = os.getenv('DISPLAY_HOST', '127.0.0.1')
DISPLAY_PORT = int(os.getenv('DISPLAY_PORT', '8765'))
DISPLAY_JPEG_QUALITY = int(os.getenv('DISPLAY_JPEG_QUALITY', '82'))

# Model paths
SHAPE_PREDICTOR_PATH = resource_path('models/shape_predictor_68_face_landmarks.dat')
FACE_RECOGNITION_MODEL_PATH = resource_path('models/dlib_face_recognition_resnet_model_v1.dat')
CNN_FACE_DETECTOR_PATH = os.getenv(
    'DLIB_CNN_FACE_DETECTOR_PATH',
    resource_path('models/mmod_human_face_detector.dat')
)

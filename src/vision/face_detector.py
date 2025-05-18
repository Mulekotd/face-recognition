import cv2 as cv
import dlib
import os

from src.common.constants import (
    CNN_FACE_DETECTOR_PATH,
    DLIB_CNN_UPSAMPLE,
    DLIB_HOG_UPSAMPLE,
    FACE_DETECTION_SCALE,
    SHAPE_PREDICTOR_PATH,
    USE_CUDA_FACE_DETECTOR
)


class FaceDetector:
    def __init__(self):
        self.backend = "hog"
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        dlib_uses_cuda = bool(getattr(dlib, "DLIB_USE_CUDA", False))

        if (
            USE_CUDA_FACE_DETECTOR
            and dlib_uses_cuda
            and os.path.exists(CNN_FACE_DETECTOR_PATH)
        ):
            self.face_detector = dlib.cnn_face_detection_model_v1(
                CNN_FACE_DETECTOR_PATH
            )
            self.backend = "cnn"

        self.cuda_enabled = dlib_uses_cuda and self.backend == "cnn"

    def detect_faces(self, frame, scale_factor=FACE_DETECTION_SCALE):
        if scale_factor != 1.0:
            detection_frame = cv.resize(
                frame,
                (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv.INTER_LINEAR
            )
        else:
            detection_frame = frame

        rgb_detection_frame = cv.cvtColor(detection_frame, cv.COLOR_BGR2RGB)

        if self.backend == "cnn":
            return [
                detection.rect
                for detection in self.face_detector(
                    rgb_detection_frame,
                    DLIB_CNN_UPSAMPLE
                )
            ]

        return self.face_detector(rgb_detection_frame, DLIB_HOG_UPSAMPLE)

    def get_face_landmarks(self, rgb_frame, detection):
        return self.shape_predictor(rgb_frame, detection)

    def get_backend_name(self):
        if self.backend == "cnn" and self.cuda_enabled:
            return "dlib-cnn-cuda"

        return f"dlib-{self.backend}"

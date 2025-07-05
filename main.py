import uuid
from threading import Thread

import cv2 as cv

from src.config import (
    WINDOW_TITLE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    YAML_PATH,
    CAMERA_FPS,
    CAMERA_BUFFER_SIZE,
    FRAME_SKIP,
    FACE_DETECTION_SCALE,
    RECOGNITION_INTERVAL,
    MAX_FACES_CACHE,
)
from src.database_manager import DatabaseManager
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.fps_calculator import FPSCalculator
from src.ui_renderer import UIRenderer
from src.utils import match_faces, adjust_detection_coordinates
from src.window_manager import WindowManager


def main():  # noqa: C901 – complexity intentionally accepted
    # --- Component initialization -------------------------------------------------------------
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    database_manager = DatabaseManager()
    fps_calculator = FPSCalculator()
    ui_renderer = UIRenderer()
    window_manager = WindowManager(WINDOW_TITLE, CAMERA_WIDTH, CAMERA_HEIGHT)

    # Load the face database asynchronously
    Thread(
        target=database_manager.load_database_async,
        args=(YAML_PATH,),
        daemon=True,
    ).start()

    # --- Camera setup -------------------------------------------------------------------------
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam.set(cv.CAP_PROP_FPS, CAMERA_FPS)
    cam.set(cv.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

    frame_count = 0
    face_data_list = []

    # --- Main loop -----------------------------------------------------------------------------
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Update FPS counter
        fps_calculator.update()
        current_fps = fps_calculator.get_fps()

        # Check for window resizing (return value not used)
        window_manager.update_window_size()

        # Perform face detection and recognition at intervals
        if frame_count % FRAME_SKIP == 0:
            dets = face_detector.detect_faces(frame)
            adjusted_dets = adjust_detection_coordinates(dets, FACE_DETECTION_SCALE)

            matches = match_faces(face_data_list, adjusted_dets)
            updated_face_data = []
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            for det, matched_face in matches:
                if matched_face:
                    matched_face["rect"] = det
                    face_data = matched_face
                else:
                    if frame_count % RECOGNITION_INTERVAL == 0:
                        shape = face_detector.get_face_landmarks(rgb_frame, det)
                        embedding = face_recognizer.compute_face_descriptor(
                            rgb_frame, shape
                        )
                        name = (
                            face_recognizer.recognize_embedding(
                                embedding, database_manager.get_database()
                            )
                            if database_manager.is_database_loaded()
                            else "Loading..."
                        )
                    else:
                        name = "Processing..."

                    face_data = {
                        "uid": str(uuid.uuid4()),
                        "rect": det,
                        "name": name,
                    }

                updated_face_data.append(face_data)

            face_data_list = updated_face_data

            # Limit face cache size
            if len(face_data_list) > MAX_FACES_CACHE:
                face_data_list = face_data_list[-MAX_FACES_CACHE:]

        # --- Rendering --------------------------------------------------------------------------
        display_frame = frame.copy()

        # Use original rects (no need to scale back)
        ui_renderer.draw_face_rectangles(display_frame, face_data_list)
        ui_renderer.draw_fps(display_frame, current_fps)

        # Resize frame to fit the current window size
        final_frame = window_manager.resize_frame(display_frame)

        # Display the frame
        cv.imshow(WINDOW_TITLE, final_frame)
        frame_count += 1

        # Exit on ESC or window close
        key = cv.waitKey(1) & 0xFF

        if key == 27 or cv.getWindowProperty(WINDOW_TITLE, cv.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

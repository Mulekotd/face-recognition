from concurrent.futures import ThreadPoolExecutor
from threading import Thread

import cv2 as cv

from src.camera.camera_manager import (
    CameraSettings,
    CameraSetupError,
    open_camera
)

from src.common.constants import (
    WINDOW_TITLE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    YAML_PATH,
    CAMERA_FPS,
    CAMERA_BUFFER_SIZE,
    FRAME_SKIP,
    FACE_DETECTION_SCALE,
    RECOGNITION_INTERVAL,
    RECOGNITION_WORKERS
)

from src.common.fps_calculator import FPSCalculator
from src.common.geometry import adjust_detection_coordinates
from src.common.snowflake import generate_embedding_snowflake

from src.storage.database_manager import DatabaseManager
from src.tracking.face_tracker import FaceTrackManager
from src.ui.renderer import UIRenderer
from src.ui.window_manager import WindowManager
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer


def recognize_face_snapshot(
    rgb_frame,
    detection,
    face_detector,
    face_recognizer,
    embedding_matrix,
    embedding_labels
):
    shape = face_detector.get_face_landmarks(rgb_frame, detection)

    embedding = face_recognizer.compute_face_descriptor(rgb_frame, shape)
    embedding_uid = generate_embedding_snowflake(embedding)

    name, distance = face_recognizer.recognize_embedding_from_index(
        embedding,
        embedding_matrix,
        embedding_labels
    )

    return name, distance, embedding_uid


def main():  # noqa: C901 – complexity intentionally accepted
    # --- Component initialization -------------------------------------------------------------
    database_manager = DatabaseManager()

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    face_tracker = FaceTrackManager()
    fps_calculator = FPSCalculator()

    ui_renderer = UIRenderer()
    window_manager = WindowManager(WINDOW_TITLE, CAMERA_WIDTH, CAMERA_HEIGHT)

    recognition_executor = ThreadPoolExecutor(max_workers=RECOGNITION_WORKERS)
    pending_recognitions = {}

    # Load the face database asynchronously
    Thread(
        target=database_manager.load_database_async,
        args=(YAML_PATH,),
        daemon=True
    ).start()

    # --- Camera setup -------------------------------------------------------------------------
    camera_settings = CameraSettings(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps=CAMERA_FPS,
        buffer_size=CAMERA_BUFFER_SIZE
    )

    try:
        camera = open_camera(camera_settings)
    except (CameraSetupError, ValueError) as exc:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
        window_manager.close()

        raise SystemExit(f"Camera setup failed: {exc}") from exc

    cam = camera.capture

    print(f"Using camera source: {camera.label}")

    frame_count = 0

    try:
        # --- Main loop -------------------------------------------------------------------------
        while True:
            ret, frame = cam.read()

            if not ret:
                print(f"Camera read failed from {camera.label}.")
                break

            # Update FPS counter
            fps_calculator.update()
            current_fps = fps_calculator.get_fps()

            # Check for window resizing (return value not used)
            window_manager.update_window_size()

            for uid, future in list(pending_recognitions.items()):
                if not future.done():
                    continue

                try:
                    name, distance, embedding_uid = future.result()
                    face_tracker.apply_recognition_result(
                        uid,
                        name,
                        distance,
                        embedding_uid
                    )
                except Exception as exc:
                    print(f"Recognition failed for track {uid}: {exc}")
                    face_tracker.clear_pending_recognition(uid)
                finally:
                    pending_recognitions.pop(uid, None)

            # Perform face detection at intervals. Recognition jobs are queued
            # asynchronously so slow embeddings do not block rendering.
            if frame_count % FRAME_SKIP == 0:
                dets = face_detector.detect_faces(frame)
                adjusted_dets = adjust_detection_coordinates(
                    dets,
                    FACE_DETECTION_SCALE,
                    frame.shape
                )

                active_tracks = face_tracker.update(adjusted_dets, frame_count)
                database_loaded = database_manager.is_database_loaded()

                embedding_matrix, embedding_labels = (
                    database_manager.get_recognition_index()
                    if database_loaded
                    else (None, None)
                )

                has_embeddings = (
                    embedding_matrix is not None and len(embedding_matrix) > 0
                )

                rgb_frame = None

                for track in active_tracks:
                    if database_loaded and not has_embeddings:
                        track.apply_recognition_result("Unknown")
                        continue

                    if not track.should_recognize(
                        frame_count,
                        RECOGNITION_INTERVAL,
                        database_loaded
                    ): continue

                    if rgb_frame is None:
                        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    track.mark_recognition_pending(frame_count)
                    pending_recognitions[track.uid] = recognition_executor.submit(
                        recognize_face_snapshot,
                        rgb_frame,
                        track.rect,
                        face_detector,
                        face_recognizer,
                        embedding_matrix,
                        embedding_labels
                    )

            # --- Rendering ----------------------------------------------------------------------
            display_frame = frame.copy()
            tracked_faces = face_tracker.get_render_data(frame_count, frame.shape)

            ui_renderer.draw_face_rectangles(display_frame, tracked_faces)
            ui_renderer.draw_fps(display_frame, current_fps)

            if not window_manager.show_frame(display_frame):
                break

            frame_count += 1

            if not window_manager.is_open():
                break

    except KeyboardInterrupt:
        pass
    finally:
        recognition_executor.shutdown(wait=False, cancel_futures=True)
        cam.release()
        window_manager.close()


if __name__ == "__main__":
    main()

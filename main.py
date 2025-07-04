import cv2 as cv
import uuid
from threading import Thread

from src.config import *
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.database_manager import DatabaseManager
from src.fps_calculator import FPSCalculator
from src.ui_renderer import UIRenderer
from src.utils import match_faces, adjust_detection_coordinates
from src.window_manager import WindowManager

def main():
    # Initialize components
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    database_manager = DatabaseManager()
    fps_calculator = FPSCalculator()
    ui_renderer = UIRenderer()
    window_manager = WindowManager(WINDOW_TITLE, CAMERA_WIDTH, CAMERA_HEIGHT)

    # Load database asynchronously
    Thread(target=database_manager.load_database_async, args=(YAML_PATH,), daemon=True).start()

    # Configure camera
    cam = cv.VideoCapture(0)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cam.set(cv.CAP_PROP_FPS, CAMERA_FPS)
    cam.set(cv.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    
    frame_count = 0
    face_data_list = []
    
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        # Update FPS
        fps_calculator.update()
        current_fps = fps_calculator.get_fps()

        # Check for window resize
        window_resized = window_manager.update_window_size()
        
        # Face detection and recognition
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
                        embedding = face_recognizer.compute_face_descriptor(rgb_frame, shape)
                        
                        name = face_recognizer.recognize_embedding(
                            embedding, 
                            database_manager.get_database()
                        ) if database_manager.is_database_loaded() else "Loading..."
                    else:
                        name = "Processing..."
                    
                    face_data = {
                        "uid": str(uuid.uuid4()),
                        "rect": det,
                        "name": name
                    }
                
                updated_face_data.append(face_data)
            
            face_data_list = updated_face_data
            
            if len(face_data_list) > MAX_FACES_CACHE:
                face_data_list = face_data_list[-MAX_FACES_CACHE:]

        # Create display frame
        display_frame = frame.copy()
        
        # Scale face data for display
        scaled_face_data = []
        for face_data in face_data_list:
            scaled_face_data.append({
                "uid": face_data["uid"],
                "rect": face_data["rect"],  # Keep original rect for calculations
                "name": face_data["name"]
            })

        # Render UI on original frame
        ui_renderer.draw_face_rectangles(display_frame, scaled_face_data)
        ui_renderer.draw_fps(display_frame, current_fps)
        ui_renderer.draw_performance_info(display_frame, database_manager.is_database_loaded(), len(face_data_list))
        
        # Add window size info
        scale_factor = window_manager.get_scale_factor()
        window_info = f"Scale: {scale_factor:.2f}x | Size: {window_manager.current_width}x{window_manager.current_height}"
        cv.putText(display_frame, window_info, (10, display_frame.shape[0] - 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Resize frame to fit window
        final_frame = window_manager.resize_frame(display_frame)
        
        # Display frame
        cv.imshow(WINDOW_TITLE, final_frame)
        frame_count += 1
        
        # Check for exit conditions
        key = cv.waitKey(1) & 0xFF

        if key == 27:
            break
        elif cv.getWindowProperty(WINDOW_TITLE, cv.WND_PROP_VISIBLE) < 1:
            break
    
    cam.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

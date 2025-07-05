import cv2 as cv


class UIRenderer:
    @staticmethod
    def draw_fps(frame, fps):
        fps_text = f"FPS: {fps:.1f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Get text size
        (text_width, text_height), baseline = cv.getTextSize(fps_text, font, font_scale, thickness)

        # Position in the top right corner
        x = frame.shape[1] - text_width - 10
        y = text_height + 10

        cv.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (0, 0, 0), -1)
        cv.putText(frame, fps_text, (x, y), font, font_scale, (0, 255, 0), thickness)

    @staticmethod
    def get_face_color(name):
        if name == "Loading...":
            return (0, 255, 255)
        elif name == "Processing...":
            return (255, 255, 0)
        elif name == "Unknown":
            return (0, 0, 255)
        else:
            return (0, 255, 0)

    @staticmethod
    def draw_face_rectangles(frame, face_data_list):
        for face_data in face_data_list:
            det = face_data["rect"]
            x, y, x2, y2 = det.left(), det.top(), det.right(), det.bottom()

            color = UIRenderer.get_face_color(face_data["name"])
            cv.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv.putText(frame, face_data["name"], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

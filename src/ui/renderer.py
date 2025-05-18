import cv2 as cv


class UIRenderer:
    LOADING_NAMES = {"Loading..."}
    PROCESSING_NAMES = {"Processing..."}
    UNKNOWN_NAMES = {"Unknown"}

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
        if name in UIRenderer.LOADING_NAMES:
            return (0, 255, 255)
        elif name in UIRenderer.PROCESSING_NAMES:
            return (255, 255, 0)
        elif name in UIRenderer.UNKNOWN_NAMES:
            return (0, 0, 255)
        else:
            return (0, 255, 0)

    @staticmethod
    def draw_face_rectangles(frame, face_data_list):
        for face_data in face_data_list:
            det = face_data["rect"]

            x = max(0, det.left())
            y = max(0, det.top())

            x2 = min(frame.shape[1] - 1, det.right())
            y2 = min(frame.shape[0] - 1, det.bottom())

            name = face_data["name"]
            color = UIRenderer.get_face_color(name)

            cv.rectangle(frame, (x, y), (x2, y2), color, 2)

            (text_width, text_height), baseline = cv.getTextSize(
                name,
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            label_y = max(text_height + 8, y - 8)

            cv.rectangle(
                frame,
                (x, label_y - text_height - baseline - 4),
                (x + text_width + 8, label_y + baseline),
                (0, 0, 0),
                -1
            )
            cv.putText(
                frame,
                name,
                (x + 4, label_y - 3),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

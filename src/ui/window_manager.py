from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Thread

import cv2 as cv

from src.common.constants import DISPLAY_HOST, DISPLAY_JPEG_QUALITY, DISPLAY_PORT


class WindowManager:
    def __init__(self, window_title, camera_width=640, camera_height=480):
        self.window_title = window_title
        self.current_width = camera_width
        self.current_height = camera_height
        self.scale_factor = 1.0
        self.closed = False
        
        self._latest_jpeg = None

        self._frame_condition = Condition()
        self._frame_version = 0

        handler = self._make_handler()

        try:
            self.server = ThreadingHTTPServer((DISPLAY_HOST, DISPLAY_PORT), handler)
        except OSError:
            self.server = ThreadingHTTPServer((DISPLAY_HOST, 0), handler)

        self.server.daemon_threads = True
        self.server_thread = Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

        self.url = f"http://{DISPLAY_HOST}:{self.server.server_port}/"

        print(f"Face Recognition display: {self.url}")

    def update_window_size(self):
        return False

    def resize_frame(self, frame):
        return frame

    def show_frame(self, frame):
        if self.closed:
            return False

        encode_params = [cv.IMWRITE_JPEG_QUALITY, DISPLAY_JPEG_QUALITY]
        success, encoded_frame = cv.imencode(".jpg", frame, encode_params)

        if not success:
            return True

        with self._frame_condition:
            self._latest_jpeg = encoded_frame.tobytes()
            self._frame_version += 1
            self._frame_condition.notify_all()

        return True

    def is_open(self):
        return not self.closed

    def close(self):
        if self.closed:
            return

        self.closed = True

        with self._frame_condition:
            self._frame_condition.notify_all()

        self.server.shutdown()
        self.server.server_close()

    def get_scale_factor(self):
        return self.scale_factor

    def _make_handler(self):
        manager = self

        class FrameStreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self._send_index()
                    return

                if self.path == "/stream.mjpg":
                    self._send_stream()
                    return

                if self.path == "/health":
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")
                    return

                self.send_error(404)

            def log_message(self, _format, *_args):
                return

            def _send_index(self):
                page = f"""<!doctype html>
                            <html lang="en">
                            <head>
                            <meta charset="utf-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1">
                            <title>{manager.window_title}</title>
                            <style>
                                html, body {{
                                background: #111;
                                height: 100%;
                                margin: 0;
                                overflow: hidden;
                                }}
                                body {{
                                align-items: center;
                                display: flex;
                                justify-content: center;
                                }}
                                img {{
                                height: 100vh;
                                max-width: 100vw;
                                object-fit: contain;
                                width: 100vw;
                                }}
                            </style>
                            </head>
                            <body>
                            <img src="/stream.mjpg" alt="Face Recognition stream">
                            </body>
                            </html>"""
                encoded_page = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded_page)))
                self.end_headers()
                self.wfile.write(encoded_page)

            def _send_stream(self):
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.end_headers()

                last_frame_version = -1

                while manager.is_open():
                    with manager._frame_condition:
                        manager._frame_condition.wait_for(
                            lambda: (
                                manager._frame_version != last_frame_version
                                or not manager.is_open()
                            ),
                            timeout=1.0
                        )

                        if not manager.is_open():
                            break

                        frame = manager._latest_jpeg
                        last_frame_version = manager._frame_version

                    if frame is None:
                        continue

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii")
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

        return FrameStreamHandler

import cv2 as cv
import dlib
import numpy as np
import yaml
import os

from threading import Thread, Lock
from queue import Queue

# === CONFIG ===
WINDOW_TITLE = 'Face Recognition'
YAML_PATH = 'database/people.yaml'

THRESHOLD = 0.5
FRAME_SKIP = 192

# === MODELS ===
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

# === THREADING ===
class RecognitionThread(Thread):
    def __init__(self, database):
        super().__init__()
        self.database = database
        self.input_queue = Queue()
        self.result = "Processando..."
        self.lock = Lock()
        self.daemon = True
        self.start()

    def run(self):
        while True:
            embedding = self.input_queue.get()
            name = "Desconhecido"

            for person in self.database:
                for db_emb in person['embeddings']:
                    if calculate_norm(embedding, db_emb) <= THRESHOLD:
                        name = person['name']
                        break

                if name != "Desconhecido":
                    break

            with self.lock:
                self.result = name

    def recognize(self, embedding):
        self.input_queue.queue.clear()
        self.input_queue.put(embedding)

    def get_result(self):
        with self.lock:
            return self.result

# === UTILS ===
def calculate_norm(embedding, db_emb):
    return round(np.linalg.norm(embedding - db_emb), 2)

# === LOAD DATABASE ===
def process_image(image_path, result_queue):
    full_path = os.path.join('database', image_path)
    img = cv.imread(full_path)

    if img is None:
        result_queue.put(None)
        return
    
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    dets = face_detector(rgb_img)

    if dets:
        shape = shape_predictor(rgb_img, dets[0])
        face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, shape)
        result_queue.put(np.array(face_descriptor))
    else:
        result_queue.put(None)

def load_database(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        people = yaml.safe_load(f)

    for person in people:
        result_queue = Queue()
        threads = []

        for image_path in person['images']:
            thread = Thread(target=process_image, args=(image_path, result_queue))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        embeddings = []

        while not result_queue.empty():
            emb = result_queue.get()
            if emb is not None:
                embeddings.append(emb)

        person['embeddings'] = embeddings

    return people

database = load_database(YAML_PATH)
recognizer = RecognitionThread(database)

# === START CAMERA ===
cam = cv.VideoCapture(0)
frame_count = 0

while True:
    _, frame = cam.read()

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    dets = face_detector(rgb_frame)

    for det in dets:
        x, y, x2, y2 = det.left(), det.top(), det.right(), det.bottom()

        cv.rectangle(frame, (x, y), (x2, y2), (255, 255, 0), 2)

        if frame_count % FRAME_SKIP == 0:
            shape = shape_predictor(rgb_frame, det)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_frame, shape)

            live_embedding = np.array(face_descriptor)
            recognizer.recognize(live_embedding)

        matched_name = recognizer.get_result()
        cv.putText(frame, matched_name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow(WINDOW_TITLE, frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cam.release()
cv.destroyAllWindows()

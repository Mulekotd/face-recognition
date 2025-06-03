import cv2 as cv
import dlib
import numpy as np
import uuid
import yaml

import sys
import os

from threading import Thread
from concurrent.futures import ThreadPoolExecutor

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# === CONFIG ===
WINDOW_TITLE = 'Face Recognition'
YAML_PATH = resource_path('database/people.yaml')

FRAME_SKIP = 256
MAX_THREADS = 4
THRESHOLD = 0.5

# === MODELOS ===
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(resource_path('models/shape_predictor_68_face_landmarks.dat'))
face_rec_model = dlib.face_recognition_model_v1(resource_path('models/dlib_face_recognition_resnet_model_v1.dat'))

# === UTILS ===
def recognize_embedding(embedding, database, threshold=THRESHOLD):
    if not database:
        return "Processando..."

    best_match = None
    best_distance = float('inf')

    for person in database:
        for db_emb in person.get('embeddings', []):
            similarity = np.linalg.norm(embedding - db_emb)
            if similarity < best_distance:
                best_distance = similarity
                best_match = person['name']

    return best_match if best_distance <= threshold else "Desconhecido"


def process_image(image_path):
    full_path = resource_path(os.path.join('database', *image_path.split('/')))
    img = cv.imread(full_path)

    if img is None:
        return None

    img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    dets = face_detector(rgb_img)

    if not dets:
        return None

    shape = shape_predictor(rgb_img, dets[0])
    face_descriptor = face_rec_model.compute_face_descriptor(rgb_img, shape)

    return np.array(face_descriptor)


def load_database(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        people = yaml.safe_load(f)

    for person in people:
        image_paths = person.get('images', [])
        embeddings = []

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            results = executor.map(process_image, image_paths)

            for emb in results:
                if emb is not None:
                    embeddings.append(emb)

        person['embeddings'] = embeddings

        yield person

def match_faces(previous_faces, current_dets, max_distance=50):
    matches = []

    for det in current_dets:
        x, y = det.left(), det.top()
        best_match = None
        best_distance = float('inf')

        for face in previous_faces:
            px, py = face["rect"].left(), face["rect"].top()
            dist = np.linalg.norm([x - px, y - py])

            if dist < best_distance and dist < max_distance:
                best_match = face
                best_distance = dist

        matches.append((det, best_match))

    return matches

# === MAIN ===
database = []
database_loaded = False

def load_database_async():
    global database, database_loaded
    for person in load_database(YAML_PATH):
        database.append(person)
    database_loaded = True

Thread(target=load_database_async).start()

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
face_data_list = []

while True:
    ret, frame = cam.read()

    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    dets = face_detector(rgb_frame)

    matches = match_faces(face_data_list, dets)
    updated_face_data = []

    for det, matched_face in matches:
        x, y, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv.rectangle(frame, (x, y), (x2, y2), (255, 255, 0), 2)

        if matched_face:
            matched_face["rect"] = det
            face_data = matched_face
        else:
            shape = shape_predictor(rgb_frame, det)
            face_descriptor = face_rec_model.compute_face_descriptor(rgb_frame, shape)
            embedding = np.array(face_descriptor)

            name = recognize_embedding(embedding, database) if database_loaded else "Carregando..."

            face_data = {
                "uid": str(uuid.uuid4()),
                "rect": det,
                "name": name
            }

        updated_face_data.append(face_data)
        cv.putText(frame, face_data["name"], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    face_data_list = updated_face_data

    cv.imshow(WINDOW_TITLE, frame)

    if cv.waitKey(1) & 0xFF == 27 or cv.getWindowProperty(WINDOW_TITLE, cv.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv.destroyAllWindows()

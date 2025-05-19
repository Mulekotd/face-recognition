import cv2 as cv
import dlib
import numpy as np
import uuid
import yaml
import os

from queue import Queue
from threading import Thread

# === CONFIG ===
WINDOW_TITLE = 'Face Recognition'
YAML_PATH = 'database/people.yaml'

THRESHOLD = 0.5
FRAME_SKIP = 256

# === MODELS ===
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

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

    if best_distance <= threshold:
        return best_match

    return "Desconhecido"

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
    database = load_database(YAML_PATH)
    database_loaded = True

Thread(target=load_database_async).start()

cam = cv.VideoCapture(0)
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

            if database_loaded:
                name = recognize_embedding(embedding, database)
            else:
                name = "Carregando..."

            face_data = {
                "uid": str(uuid.uuid4()),
                "rect": det,
                "name": name
            }

        updated_face_data.append(face_data)

        cv.putText(frame, face_data["name"], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    face_data_list = updated_face_data

    cv.imshow(WINDOW_TITLE, frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

    frame_count += 1

cam.release()
cv.destroyAllWindows()

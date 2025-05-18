import cv2 as cv
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import yaml

from src.common.constants import MAX_THREADS, resource_path
from src.vision.face_detector import FaceDetector
from src.vision.face_recognizer import FaceRecognizer


class DatabaseManager:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

        self.database = []
        self.database_loaded = False

        self.embedding_matrix = np.empty((0, 128), dtype=np.float32)
        self.embedding_labels = np.empty((0,), dtype=object)

        self._lock = RLock()

    def process_image(self, image_path):
        full_path = resource_path(os.path.join('database', *image_path.split('/')))
        
        img = cv.imread(full_path)
        if img is None:
            return None

        img = cv.resize(img, (0, 0), fx=0.4, fy=0.4)
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        dets = self.face_detector.detect_faces(img, scale_factor=1.0)
        if not dets:
            return None

        shape = self.face_detector.get_face_landmarks(rgb_img, dets[0])
        face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_img, shape)

        return face_descriptor

    def load_database(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            people = yaml.safe_load(f) or []

        for person in people:
            image_paths = person.get('images', [])
            embeddings = []

            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                results = executor.map(self.process_image, image_paths)
                for emb in results:
                    if emb is not None:
                        embeddings.append(emb)

            person['embeddings'] = embeddings
            yield person

    def load_database_async(self, yaml_path):
        loaded_people = []

        for person in self.load_database(yaml_path):
            loaded_people.append(person)

        embedding_matrix, embedding_labels = self._build_recognition_index(
            loaded_people
        )

        with self._lock:
            self.database = loaded_people
            self.embedding_matrix = embedding_matrix
            self.embedding_labels = embedding_labels
            self.database_loaded = True

    def get_database(self):
        with self._lock:
            return list(self.database)

    def is_database_loaded(self):
        with self._lock:
            return self.database_loaded

    def get_recognition_index(self):
        with self._lock:
            return self.embedding_matrix, self.embedding_labels

    def _build_recognition_index(self, people):
        embeddings = []
        labels = []

        for person in people:
            for embedding in person.get('embeddings', []):
                embeddings.append(np.asarray(embedding, dtype=np.float32))
                labels.append(person['name'])

        if not embeddings:
            return (
                np.empty((0, 128), dtype=np.float32),
                np.empty((0,), dtype=object)
            )

        return (
            np.vstack(embeddings).astype(np.float32),
            np.asarray(labels, dtype=object)
        )

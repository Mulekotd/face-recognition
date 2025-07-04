import cv2 as cv
import yaml
import os
from concurrent.futures import ThreadPoolExecutor

from .config import resource_path, MAX_THREADS
from .face_detector import FaceDetector
from .face_recognizer import FaceRecognizer

class DatabaseManager:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.database = []
        self.database_loaded = False
    
    def process_image(self, image_path):
        full_path = resource_path(os.path.join('database', *image_path.split('/')))
        img = cv.imread(full_path)

        if img is None:
            return None

        img = cv.resize(img, (0, 0), fx=0.4, fy=0.4)
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        dets = self.face_detector.face_detector(rgb_img)

        if not dets:
            return None

        shape = self.face_detector.get_face_landmarks(rgb_img, dets[0])
        face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_img, shape)

        return face_descriptor
    
    def load_database(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            people = yaml.safe_load(f)

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
        for person in self.load_database(yaml_path):
            self.database.append(person)

        self.database_loaded = True
    
    def get_database(self):
        return self.database
    
    def is_database_loaded(self):
        return self.database_loaded
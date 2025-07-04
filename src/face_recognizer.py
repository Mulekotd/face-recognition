import cv2 as cv
import dlib
import numpy as np

from .config import FACE_RECOGNITION_MODEL_PATH, THRESHOLD

class FaceRecognizer:
    def __init__(self):
        self.face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    
    def compute_face_descriptor(self, rgb_frame, shape):
        face_descriptor = self.face_rec_model.compute_face_descriptor(rgb_frame, shape)

        return np.array(face_descriptor)
    
    def recognize_embedding(self, embedding, database, threshold=THRESHOLD):
        if not database:
            return "Processando..."

        best_match = None
        best_distance = float('inf')

        # Optimization: use numpy broadcasting for vectorized calculations
        for person in database:
            if not person.get('embeddings'):
                continue
                
            embeddings_array = np.array(person['embeddings'])
            distances = np.linalg.norm(embeddings_array - embedding, axis=1)
            min_distance = np.min(distances)
            
            if min_distance < best_distance:
                best_distance = min_distance
                best_match = person['name']

        return best_match if best_distance <= threshold else "Desconhecido"
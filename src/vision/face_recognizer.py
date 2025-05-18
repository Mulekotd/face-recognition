import dlib
import numpy as np

from src.common.constants import (
    FACE_DESCRIPTOR_NUM_JITTERS,
    FACE_DESCRIPTOR_PADDING,
    FACE_RECOGNITION_MODEL_PATH,
    THRESHOLD
)


class FaceRecognizer:
    def __init__(self):
        self.face_rec_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

    def compute_face_descriptor(self, rgb_frame, shape):
        face_descriptor = self.face_rec_model.compute_face_descriptor(
            rgb_frame,
            shape,
            FACE_DESCRIPTOR_NUM_JITTERS,
            FACE_DESCRIPTOR_PADDING
        )

        return np.asarray(face_descriptor, dtype=np.float32)

    def recognize_embedding(self, embedding, database, threshold=THRESHOLD):
        name, _ = self.recognize_embedding_with_distance(
            embedding,
            database,
            threshold
        )

        return name

    def recognize_embedding_with_distance(self, embedding, database, threshold=THRESHOLD):
        if not database:
            return "Processing...", None

        embeddings = []
        labels = []

        for person in database:
            if not person.get('embeddings'):
                continue

            for person_embedding in person['embeddings']:
                embeddings.append(person_embedding)
                labels.append(person['name'])

        if not embeddings:
            return "Processing...", None

        return self.recognize_embedding_from_index(
            embedding,
            np.asarray(embeddings, dtype=np.float32),
            np.asarray(labels, dtype=object),
            threshold
        )

    def recognize_embedding_from_index(
        self,
        embedding,
        embedding_matrix,
        embedding_labels,
        threshold=THRESHOLD
    ):
        if embedding_matrix is None or len(embedding_matrix) == 0:
            return "Processing...", None

        embedding = np.asarray(embedding, dtype=np.float32)
        
        distances = np.linalg.norm(embedding_matrix - embedding, axis=1)
        best_index = int(np.argmin(distances))
        best_distance = float(distances[best_index])
        if best_distance <= threshold:
            return str(embedding_labels[best_index]), best_distance

        return "Unknown", best_distance

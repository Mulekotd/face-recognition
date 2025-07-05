import dlib


def match_faces(previous_faces, current_dets, max_distance=80):
    matches = []

    for det in current_dets:
        x, y = det.left(), det.top()

        best_match = None
        best_distance = float('inf')

        for face in previous_faces:
            px, py = face["rect"].left(), face["rect"].top()

            # Optimization: Manhattan distance
            dist = abs(x - px) + abs(y - py)

            if dist < best_distance and dist < max_distance:
                best_match = face
                best_distance = dist

        matches.append((det, best_match))

    return matches

def adjust_detection_coordinates(dets, scale_factor):
    adjusted_dets = []

    for det in dets:
        adjusted_det = dlib.rectangle(
            int(det.left() / scale_factor),
            int(det.top() / scale_factor),
            int(det.right() / scale_factor),
            int(det.bottom() / scale_factor)
        )
        adjusted_dets.append(adjusted_det)

    return adjusted_dets

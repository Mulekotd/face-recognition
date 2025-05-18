import math

import dlib


def make_rectangle(left, top, right, bottom):
    return dlib.rectangle(
        int(round(left)),
        int(round(top)),
        int(round(right)),
        int(round(bottom))
    )


def rect_width(rect):
    return max(1, rect.right() - rect.left())


def rect_height(rect):
    return max(1, rect.bottom() - rect.top())


def rect_center(rect):
    return (
        rect.left() + rect_width(rect) / 2,
        rect.top()  + rect_height(rect) / 2
    )


def rect_center_distance(first_rect, second_rect):
    first_x, first_y = rect_center(first_rect)
    second_x, second_y = rect_center(second_rect)

    return math.hypot(first_x - second_x, first_y - second_y)


def rect_iou(first_rect, second_rect):
    left = max(first_rect.left(), second_rect.left())
    top = max(first_rect.top(), second_rect.top())
    right = min(first_rect.right(), second_rect.right())
    bottom = min(first_rect.bottom(), second_rect.bottom())

    intersection_width = max(0, right - left)
    intersection_height = max(0, bottom - top)

    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0

    first_area = rect_width(first_rect) * rect_height(first_rect)
    second_area = rect_width(second_rect) * rect_height(second_rect)
    union_area = first_area + second_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def smooth_rectangle(previous_rect, current_rect, alpha):
    inverse_alpha = 1.0 - alpha

    return make_rectangle(
        previous_rect.left() * inverse_alpha + current_rect.left() * alpha,
        previous_rect.top() * inverse_alpha + current_rect.top() * alpha,
        previous_rect.right() * inverse_alpha + current_rect.right() * alpha,
        previous_rect.bottom() * inverse_alpha + current_rect.bottom() * alpha,
    )


def move_rectangle(rect, offset_x, offset_y):
    return make_rectangle(
        rect.left() + offset_x,
        rect.top() + offset_y,
        rect.right() + offset_x,
        rect.bottom() + offset_y
    )


def clamp_rectangle(rect, frame_shape):
    height, width = frame_shape[:2]
    return make_rectangle(
        max(0, min(rect.left(), width - 1)),
        max(0, min(rect.top(), height - 1)),
        max(0, min(rect.right(), width - 1)),
        max(0, min(rect.bottom(), height - 1))
    )


def scale_rectangle(rect, scale_factor):
    return make_rectangle(
        rect.left() * scale_factor,
        rect.top() * scale_factor,
        rect.right() * scale_factor,
        rect.bottom() * scale_factor
    )


def _extract_rect(face):
    return face["rect"] if isinstance(face, dict) else face.rect


def match_faces(previous_faces, current_dets, max_distance=80, min_iou=0.10):
    matches = []
    candidates = []

    for det_index, det in enumerate(current_dets):
        for face_index, face in enumerate(previous_faces):
            previous_rect = _extract_rect(face)
            iou = rect_iou(previous_rect, det)
            distance = rect_center_distance(previous_rect, det)

            if iou >= min_iou or distance <= max_distance:
                score = distance - (iou * max_distance)
                candidates.append((score, det_index, face_index))

    used_dets = set()
    used_faces = set()
    best_matches = {}

    for _, det_index, face_index in sorted(candidates):
        if det_index in used_dets or face_index in used_faces:
            continue

        used_dets.add(det_index)
        used_faces.add(face_index)
        best_matches[det_index] = previous_faces[face_index]

    for det_index, det in enumerate(current_dets):
        matches.append((det, best_matches.get(det_index)))

    return matches

def adjust_detection_coordinates(dets, scale_factor, frame_shape=None):
    adjusted_dets = []

    for det in dets:
        adjusted_det = scale_rectangle(det, 1 / scale_factor)

        if frame_shape is not None:
            adjusted_det = clamp_rectangle(adjusted_det, frame_shape)

        adjusted_dets.append(adjusted_det)

    return adjusted_dets

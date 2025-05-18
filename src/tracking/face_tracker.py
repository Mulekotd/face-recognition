from collections import Counter, deque
from dataclasses import dataclass, field

from src.common.constants import (
    FACE_NAME_BUFFER_SIZE,
    MAX_FACES_CACHE,
    TRACK_MATCH_MAX_DISTANCE,
    TRACK_MATCH_MIN_IOU,
    TRACK_MAX_MISSED,
    TRACK_SMOOTHING
)

from src.common.geometry import (
    clamp_rectangle,
    move_rectangle,
    rect_center,
    rect_center_distance,
    rect_iou,
    smooth_rectangle
)

from src.common.snowflake import generate_track_snowflake


TRANSIENT_NAMES = {"Loading...", "Processing..."}


@dataclass
class FaceTrack:
    rect: object
    frame_index: int
    uid: str | None = None
    display_rect: object = None
    name: str = "Processing..."
    confidence: float | None = None
    missed_detections: int = 0
    last_detection_frame: int = 0
    last_recognition_frame: int = -100_000
    pending_recognition: bool = False
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    name_buffer: deque = field(
        default_factory=lambda: deque(maxlen=FACE_NAME_BUFFER_SIZE)
    )

    def __post_init__(self):
        if self.uid is None:
            self.uid = generate_track_snowflake(self.frame_index, self.rect)

        self.display_rect = self.rect
        self.last_detection_frame = self.frame_index

    def update_from_detection(self, rect, frame_index):
        previous_center_x, previous_center_y = rect_center(self.rect)
        current_center_x, current_center_y = rect_center(rect)
        frames_elapsed = max(1, frame_index - self.last_detection_frame)

        measured_velocity_x = (current_center_x - previous_center_x) / frames_elapsed
        measured_velocity_y = (current_center_y - previous_center_y) / frames_elapsed
        self.velocity_x = (self.velocity_x * 0.5) + (measured_velocity_x * 0.5)
        self.velocity_y = (self.velocity_y * 0.5) + (measured_velocity_y * 0.5)

        self.rect = rect
        self.display_rect = smooth_rectangle(self.display_rect, rect, TRACK_SMOOTHING)
        self.last_detection_frame = frame_index
        self.missed_detections = 0

    def mark_missed(self):
        self.missed_detections += 1

    def predict_display_rect(self, frame_index, frame_shape):
        frames_since_detection = max(0, frame_index - self.last_detection_frame)

        if frames_since_detection:
            predicted_rect = move_rectangle(
                self.rect,
                self.velocity_x * frames_since_detection,
                self.velocity_y * frames_since_detection
            )

            self.display_rect = smooth_rectangle(
                self.display_rect,
                predicted_rect,
                TRACK_SMOOTHING * 0.5
            )

        self.display_rect = clamp_rectangle(self.display_rect, frame_shape)
        return self.display_rect

    def should_recognize(self, frame_index, interval, database_loaded):
        if self.pending_recognition or self.missed_detections:
            return False

        if not database_loaded:
            self.name = "Loading..."
            return False

        if self.name in TRANSIENT_NAMES:
            return True

        return frame_index - self.last_recognition_frame >= interval

    def mark_recognition_pending(self, frame_index):
        self.pending_recognition = True
        self.last_recognition_frame = frame_index

        if self.name in TRANSIENT_NAMES:
            self.name = "Processing..."

    def apply_recognition_result(self, name, distance=None, embedding_uid=None):
        self.pending_recognition = False

        if not name:
            return

        if embedding_uid:
            self.uid = embedding_uid

        if name in TRANSIENT_NAMES:
            self.name = name
            return

        self.confidence = distance
        self.name_buffer.append(name)
        self.name = self._stable_name(name)

    def _stable_name(self, latest_name):
        if not self.name_buffer:
            return latest_name

        counts = Counter(self.name_buffer)
        best_name, best_count = counts.most_common(1)[0]
        current_count = counts.get(self.name, 0)

        if self.name not in TRANSIENT_NAMES and current_count >= best_count - 1:
            return self.name

        return best_name

    def to_render_data(self, frame_index, frame_shape):
        return {
            "uid": self.uid,
            "rect": self.predict_display_rect(frame_index, frame_shape),
            "name": self.name,
            "confidence": self.confidence,
            "pending_recognition": self.pending_recognition,
            "missed_detections": self.missed_detections,
        }


class FaceTrackManager:
    def __init__(
        self,
        max_tracks=MAX_FACES_CACHE,
        max_missed=TRACK_MAX_MISSED,
        max_distance=TRACK_MATCH_MAX_DISTANCE,
        min_iou=TRACK_MATCH_MIN_IOU,
    ):
        self.max_tracks = max_tracks
        self.max_missed = max_missed
        self.max_distance = max_distance
        self.min_iou = min_iou
        self.tracks = []

    def update(self, detections, frame_index):
        matches, unmatched_detection_indexes, unmatched_track_indexes = self._match(
            detections
        )
        active_tracks = []

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update_from_detection(detections[detection_index], frame_index)
            active_tracks.append(track)

        for track_index in unmatched_track_indexes:
            self.tracks[track_index].mark_missed()

        for detection_index in unmatched_detection_indexes:
            track = FaceTrack(detections[detection_index], frame_index)
            self.tracks.append(track)
            active_tracks.append(track)

        self._trim_tracks()
        active_track_uids = {track.uid for track in self.tracks}
        return [
            track
            for track in active_tracks
            if track.uid in active_track_uids
        ]

    def apply_recognition_result(self, uid, name, distance=None, embedding_uid=None):
        track = self.get_track(uid)

        if track:
            track.apply_recognition_result(name, distance, embedding_uid)

    def clear_pending_recognition(self, uid):
        track = self.get_track(uid)

        if track:
            track.pending_recognition = False

    def get_track(self, uid):
        for track in self.tracks:
            if track.uid == uid:
                return track

        return None

    def get_render_data(self, frame_index, frame_shape):
        return [
            track.to_render_data(frame_index, frame_shape)
            for track in self.tracks
            if track.missed_detections <= self.max_missed
        ]

    def _match(self, detections):
        candidates = []

        for track_index, track in enumerate(self.tracks):
            track_rect = track.display_rect or track.rect

            for detection_index, detection in enumerate(detections):
                iou = rect_iou(track_rect, detection)
                distance = rect_center_distance(track_rect, detection)

                if iou >= self.min_iou or distance <= self.max_distance:
                    score = distance - (iou * self.max_distance)
                    candidates.append((score, track_index, detection_index))

        matches = []
        matched_tracks = set()
        matched_detections = set()

        for _, track_index, detection_index in sorted(candidates):
            if track_index in matched_tracks or detection_index in matched_detections:
                continue

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)
            matches.append((track_index, detection_index))

        unmatched_track_indexes = [
            index
            for index in range(len(self.tracks))
            if index not in matched_tracks
        ]
        unmatched_detection_indexes = [
            index
            for index in range(len(detections))
            if index not in matched_detections
        ]

        return matches, unmatched_detection_indexes, unmatched_track_indexes

    def _trim_tracks(self):
        self.tracks = [
            track
            for track in self.tracks
            if track.missed_detections <= self.max_missed
        ]

        if len(self.tracks) <= self.max_tracks:
            return

        self.tracks.sort(
            key=lambda track: (
                track.missed_detections,
                -track.last_detection_frame,
            ),
        )
        self.tracks = self.tracks[: self.max_tracks]

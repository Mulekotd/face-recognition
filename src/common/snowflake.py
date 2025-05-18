from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import os
import struct
from threading import Lock


CUSTOM_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
TIMESTAMP_BITS = 41
FINGERPRINT_BITS = 12
WORKER_BITS = 5
SEQUENCE_BITS = 6

MAX_TIMESTAMP = (1 << TIMESTAMP_BITS) - 1
MAX_FINGERPRINT = (1 << FINGERPRINT_BITS) - 1
MAX_WORKER_ID = (1 << WORKER_BITS) - 1
MAX_SEQUENCE = (1 << SEQUENCE_BITS) - 1

SEQUENCE_SHIFT = 0
WORKER_SHIFT = SEQUENCE_BITS
FINGERPRINT_SHIFT = WORKER_SHIFT + WORKER_BITS
TIMESTAMP_SHIFT = FINGERPRINT_SHIFT + FINGERPRINT_BITS


@dataclass(frozen=True)
class SnowflakeKey:
    namespace: str
    value: int
    generated_at: datetime
    fingerprint: int

    def __str__(self):
        timestamp = self.generated_at.astimezone(timezone.utc)
        timestamp_text = timestamp.strftime("%Y%m%dT%H%M%S")
        millis = timestamp.microsecond // 1000

        return (
            f"{self.namespace}_{timestamp_text}{millis:03d}Z_"
            f"f{self.fingerprint:03x}_{self.value:016x}"
        )


class SnowflakeGenerator:
    def __init__(self, worker_id=None, epoch=CUSTOM_EPOCH):
        self.epoch = epoch
        self.worker_id = self._resolve_worker_id(worker_id)
        self._lock = Lock()
        self._last_timestamp_ms = -1
        self._sequence = 0

    def generate(self, namespace, payload=None, generated_at=None):
        generated_at = generated_at or datetime.now(timezone.utc)

        timestamp_ms = self._timestamp_ms(generated_at)
        fingerprint = fingerprint_payload(payload)

        with self._lock:
            timestamp_ms = self._next_timestamp_ms(timestamp_ms)
            sequence = self._sequence

        generated_at = self.epoch + timedelta(milliseconds=timestamp_ms)

        value = (
            (timestamp_ms << TIMESTAMP_SHIFT)
            | (fingerprint << FINGERPRINT_SHIFT)
            | (self.worker_id << WORKER_SHIFT)
            | sequence
        )

        return SnowflakeKey(
            namespace=namespace,
            value=value,
            generated_at=generated_at,
            fingerprint=fingerprint,
        )

    def _next_timestamp_ms(self, timestamp_ms):
        if timestamp_ms < self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms

        if timestamp_ms == self._last_timestamp_ms:
            self._sequence = (self._sequence + 1) & MAX_SEQUENCE
            if self._sequence == 0:
                timestamp_ms += 1
        else:
            self._sequence = 0

        self._last_timestamp_ms = timestamp_ms

        return timestamp_ms

    def _timestamp_ms(self, generated_at):
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=timezone.utc)

        timestamp_ms = int(
            (generated_at.astimezone(timezone.utc) - self.epoch).total_seconds()
            * 1000
        )

        if timestamp_ms < 0:
            return 0

        if timestamp_ms > MAX_TIMESTAMP:
            raise ValueError("Snowflake timestamp exceeds the configured bit size")

        return timestamp_ms

    def _resolve_worker_id(self, worker_id):
        if worker_id is None:
            worker_id = os.getenv("SNOWFLAKE_WORKER_ID", "0")

        try:
            worker_id = int(worker_id)
        except (TypeError, ValueError):
            worker_id = 0

        return worker_id & MAX_WORKER_ID


def fingerprint_payload(payload):
    payload_bytes = _payload_to_bytes(payload)
    digest = hashlib.blake2b(payload_bytes, digest_size=4).digest()

    return int.from_bytes(digest, "big") & MAX_FINGERPRINT


def _payload_to_bytes(payload):
    if payload is None:
        return b""

    if isinstance(payload, bytes):
        return payload

    if isinstance(payload, str):
        return payload.encode("utf-8")

    if hasattr(payload, "astype") and hasattr(payload, "tobytes"):
        try:
            return payload.astype("<f4", copy=False).tobytes()
        except (TypeError, ValueError):
            pass

    try:
        return b"".join(struct.pack("<f", float(value)) for value in payload)
    except (TypeError, ValueError):
        return repr(payload).encode("utf-8")


_DEFAULT_GENERATOR = SnowflakeGenerator()


def generate_track_snowflake(frame_index, rect):
    payload = (
        f"{frame_index}:"
        f"{rect.left()}:{rect.top()}:{rect.right()}:{rect.bottom()}"
    )

    return str(_DEFAULT_GENERATOR.generate("trk", payload))


def generate_embedding_snowflake(embedding, generated_at=None):
    return str(
        _DEFAULT_GENERATOR.generate(
            "emb",
            embedding,
            generated_at=generated_at
        )
    )

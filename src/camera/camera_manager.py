from dataclasses import dataclass
from pathlib import Path

import cv2 as cv


PROBE_READ_ATTEMPTS = 1
PROBE_TIMEOUT_MS = 3000
AUTO_FOURCCS = ("MJPG", "YUYV")


class CameraSetupError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraSettings:
    width: int
    height: int
    fps: int
    buffer_size: int


@dataclass(frozen=True)
class CameraCandidate:
    source: object
    label: str
    sort_key: tuple
    device_path: str | None = None
    busnum: int | None = None
    devnum: int | None = None
    busid: str | None = None
    name: str | None = None


@dataclass
class CameraHandle:
    capture: cv.VideoCapture
    source: object
    label: str


def read_sysfs_value(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def usb_number(value):
    try:
        return int(value, 10)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid USB number: {value}") from exc


def video_device_number(video_name):
    try:
        return int(video_name.removeprefix("video"))
    except ValueError:
        return 9999


def find_usb_device_path(video_sysfs_path):
    for path in (video_sysfs_path, *video_sysfs_path.parents):
        if (path / "busnum").exists() and (path / "devnum").exists():
            return path

    return None


def read_video_candidate(video_path):
    usb_device_path = find_usb_device_path(video_path.resolve())

    if usb_device_path is None:
        busnum = None
        devnum = None
        busid = None
    else:
        busnum_value = read_sysfs_value(usb_device_path / "busnum")
        devnum_value = read_sysfs_value(usb_device_path / "devnum")
        busnum = usb_number(busnum_value) if busnum_value is not None else None
        devnum = usb_number(devnum_value) if devnum_value is not None else None
        busid = usb_device_path.name

    video_index_value = read_sysfs_value(video_path / "index")
    video_index = (
        usb_number(video_index_value)
        if video_index_value is not None
        else 9999
    )

    video_number = video_device_number(video_path.name)
    device_path = f"/dev/{video_path.name}"
    camera_name = read_sysfs_value(video_path / "name")

    label = device_path

    if camera_name:
        label = f"{label} ({camera_name})"

    if busnum is not None and devnum is not None:
        label = f"{label}, USB bus {busnum:03d} device {devnum:03d}"

    if busid:
        label = f"{label}, busid {busid}"

    return CameraCandidate(
        source=device_path,
        label=label,
        sort_key=(video_index, video_number),
        device_path=device_path,
        busnum=busnum,
        devnum=devnum,
        busid=busid,
        name=camera_name
    )


def discover_camera_candidates():
    video_root = Path("/sys/class/video4linux")

    if not video_root.exists():
        return []

    candidates = []

    for video_path in sorted(video_root.glob("video*")):
        candidate = read_video_candidate(video_path)
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate.sort_key)

    return candidates


def resolve_camera_candidates():
    candidates = discover_camera_candidates()

    if candidates:
        return candidates

    return [
        CameraCandidate(
            source=0,
            label="0",
            sort_key=(0,)
        )
    ]


def is_v4l2_source(source):
    return (
        isinstance(source, str)
        and source.startswith("/dev/video")
    )


def create_capture(source):
    if is_v4l2_source(source):
        return cv.VideoCapture(source, cv.CAP_V4L2)

    return cv.VideoCapture(source)


def configure_capture(capture, settings, fourcc):
    if fourcc is not None:
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))

    capture.set(cv.CAP_PROP_BUFFERSIZE, settings.buffer_size)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, settings.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, settings.height)
    capture.set(cv.CAP_PROP_FPS, settings.fps)

    open_timeout_prop = getattr(cv, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
    read_timeout_prop = getattr(cv, "CAP_PROP_READ_TIMEOUT_MSEC", None)

    if open_timeout_prop is not None:
        capture.set(open_timeout_prop, PROBE_TIMEOUT_MS)

    if read_timeout_prop is not None:
        capture.set(read_timeout_prop, PROBE_TIMEOUT_MS)


def can_read_frame(capture):
    for _ in range(PROBE_READ_ATTEMPTS):
        ok, frame = capture.read()

        if ok and frame is not None:
            return True

    return False


def source_device_exists(candidate):
    if candidate.device_path is None:
        return True

    return Path(candidate.device_path).exists()


def open_camera_candidate(candidate, settings, fourcc):
    if not source_device_exists(candidate):
        return None, "device node does not exist"

    capture = create_capture(candidate.source)

    if not capture.isOpened():
        capture.release()
        return None, "OpenCV could not open it"

    configure_capture(capture, settings, fourcc)

    if can_read_frame(capture):
        return capture, None

    capture.release()

    return None, "OpenCV opened it, but no frame was read"


def fourcc_label(fourcc):
    return fourcc if fourcc is not None else "default"


def open_camera(settings):
    candidates = resolve_camera_candidates()
    attempts = []

    for candidate in candidates:
        for fourcc in AUTO_FOURCCS:
            capture, error = open_camera_candidate(candidate, settings, fourcc)

            if capture is not None:
                label = f"{candidate.label}, format {fourcc_label(fourcc)}"

                return CameraHandle(
                    capture=capture,
                    source=candidate.source,
                    label=label
                )

            attempts.append(
                f"{candidate.label} with {fourcc_label(fourcc)}: {error}"
            )

    detail = "\n".join(f"- {attempt}" for attempt in attempts)

    raise CameraSetupError(
        "Could not open a camera that returns frames.\n"
        f"Tried:\n{detail}"
    )

from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageSequence


# Working resolution for decoded frames. Sweeps arrive at 1024x1024 and a single
# clip can run to two thousand frames, so holding every frame as full-size RGB
# costs gigabytes and will exhaust a clinical workstation before the study is
# even graded. The model resizes to its manifest input size in grayscale anyway,
# and the quality metrics are computed on grayscale, so nothing that feeds a
# decision is lost by storing frames at the working size. Native dimensions are
# recorded in technical metadata so the audit trail keeps the true acquisition.
DEFAULT_WORKING_EDGE = 512
MIN_WORKING_EDGE = 256

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DICOM_SUFFIXES = {".dcm", ".dicom", ""}


@dataclass(slots=True)
class DecodeLimits:
    """Memory budget for one decoded source.

    ``working_edge`` bounds the longest side of every stored frame. ``max_frames``
    is a hard ceiling that, when exceeded, samples uniformly across the sweep and
    records that it did so, because a sampled study can no longer claim that every
    frame was processed.
    """

    working_edge: int = DEFAULT_WORKING_EDGE
    max_frames: int | None = None

    def __post_init__(self) -> None:
        self.working_edge = max(MIN_WORKING_EDGE, int(self.working_edge))
        if self.max_frames is not None:
            self.max_frames = max(1, int(self.max_frames))

    def stride_for(self, frame_count: int) -> int:
        if not self.max_frames or frame_count <= self.max_frames:
            return 1
        return math.ceil(frame_count / self.max_frames)


def to_working_size(image: Image.Image, working_edge: int) -> Image.Image:
    """Grayscale and bound the longest side, preserving aspect ratio."""
    gray = ImageOps.grayscale(image)
    longest = max(gray.size)
    if longest <= working_edge:
        return gray
    scale = working_edge / longest
    target = (max(1, round(gray.size[0] * scale)), max(1, round(gray.size[1] * scale)))
    return gray.resize(target, Image.BILINEAR)


@dataclass(slots=True)
class MediaFrame:
    source_name: str
    frame_index: int
    image: Image.Image
    media_type: str
    pixel_spacing_mm: tuple[float, float] | None = None
    technical_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestResult:
    frames: list[MediaFrame]
    warnings: list[str] = field(default_factory=list)
    technical_metadata: dict[str, Any] = field(default_factory=dict)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255).astype(np.uint8)


def _pil_from_array(array: np.ndarray) -> Image.Image:
    x = _to_uint8(array)
    if x.ndim == 2:
        return Image.fromarray(x, mode="L").convert("RGB")
    if x.ndim == 3 and x.shape[-1] in (3, 4):
        return Image.fromarray(x[..., :3]).convert("RGB")
    raise ValueError(f"Unsupported pixel array shape: {x.shape}")


def _decode_image(name: str, data: bytes, limits: DecodeLimits) -> IngestResult:
    image = Image.open(io.BytesIO(data))
    source_frame_count = int(getattr(image, "n_frames", 1) or 1)
    native_size: tuple[int, int] | None = None
    frames: list[MediaFrame] = []
    for index, frame in enumerate(ImageSequence.Iterator(image)):
        oriented = ImageOps.exif_transpose(frame)
        if native_size is None:
            native_size = oriented.size
        frames.append(
            MediaFrame(
                source_name=name,
                frame_index=index,
                image=to_working_size(oriented, limits.working_edge),
                media_type="image",
                technical_metadata={"source_frame_count": source_frame_count},
            )
        )
    technical = {
        "source_frame_count": source_frame_count,
        "decoded_frame_count": len(frames),
        "all_frames_processed": len(frames) == source_frame_count,
        "native_size": list(native_size) if native_size else None,
        "working_edge": limits.working_edge,
    }
    return IngestResult(frames=frames, technical_metadata=technical)


def _decode_dicom(name: str, data: bytes, limits: DecodeLimits) -> IngestResult:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM support requires pydicom.") from exc

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    array = ds.pixel_array
    spacing = None
    raw_spacing = getattr(ds, "PixelSpacing", None)
    if raw_spacing and len(raw_spacing) >= 2:
        spacing = (float(raw_spacing[0]), float(raw_spacing[1]))

    technical = {
        "modality": str(getattr(ds, "Modality", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "model": str(getattr(ds, "ManufacturerModelName", "")),
        "rows": int(getattr(ds, "Rows", 0) or 0),
        "columns": int(getattr(ds, "Columns", 0) or 0),
        "number_of_frames": int(getattr(ds, "NumberOfFrames", 1) or 1),
    }
    warnings: list[str] = []
    if spacing is None:
        warnings.append("DICOM pixel spacing is absent. Automated metric measurements must remain disabled.")

    if array.ndim == 2:
        arrays = [(0, array)]
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        arrays = [(0, array)]
    elif array.ndim in (3, 4):
        arrays = [(index, array[index]) for index in range(array.shape[0])]
    else:
        arrays = [(0, array)]

    technical["working_edge"] = limits.working_edge
    technical["decoded_frame_count"] = len(arrays)
    technical["all_frames_processed"] = len(arrays) == technical["number_of_frames"]
    if not technical["all_frames_processed"]:
        warnings.append(
            "Decoded frame count does not match the DICOM NumberOfFrames value. The examination is incomplete."
        )

    frames = [
        MediaFrame(
            source_name=name,
            frame_index=index,
            image=to_working_size(_pil_from_array(frame), limits.working_edge),
            media_type="dicom",
            pixel_spacing_mm=spacing,
            technical_metadata=technical,
        )
        for index, frame in arrays
    ]
    return IngestResult(frames=frames, warnings=warnings, technical_metadata=technical)


def _decode_video(name: str, data: bytes, limits: DecodeLimits) -> IngestResult:
    suffix = Path(name).suffix or ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(data)
            path = Path(temp.name)
        return _decode_video_path(name, path, limits)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _decode_video_path(name: str, path: Path, limits: DecodeLimits) -> IngestResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video support requires opencv-python-headless.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The video container could not be opened.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    stride = limits.stride_for(total) if total > 0 else 1

    frames: list[MediaFrame] = []
    native_size: tuple[int, int] | None = None
    source_index = 0
    read_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        read_count += 1
        if stride > 1 and source_index % stride:
            source_index += 1
            continue
        if native_size is None:
            native_size = (int(frame.shape[1]), int(frame.shape[0]))
        # Convert straight to grayscale at the working size. Going through a
        # full-resolution RGB PIL image first is what made a single sweep
        # cost gigabytes.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        longest = max(gray.shape)
        if longest > limits.working_edge:
            scale = limits.working_edge / longest
            gray = cv2.resize(
                gray,
                (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(
            MediaFrame(
                source_name=name,
                frame_index=source_index,
                image=Image.fromarray(gray, mode="L"),
                media_type="video",
                technical_metadata={"fps": fps, "source_frame_count": total},
            )
        )
        source_index += 1
    capture.release()
    if not frames:
        raise ValueError("No readable frames were found in the clip.")

    sampled = stride > 1
    complete = (total <= 0 or read_count == total) and not sampled
    warnings = []
    if total > 0 and read_count != total:
        warnings.append(
            f"The container reports {total} frames, but only {read_count} frames decoded "
            "before the stream ended."
        )
    if sampled:
        warnings.append(
            f"Only every {stride} frame was kept, {len(frames)} of {read_count}, to stay "
            "inside the frame budget. Sequential processing of every frame is not confirmed "
            "for this source."
        )
    return IngestResult(
        frames=frames,
        warnings=warnings,
        technical_metadata={
            "fps": fps,
            "source_frame_count": total,
            "read_frame_count": read_count,
            "decoded_frame_count": len(frames),
            "all_frames_processed": complete,
            "sampling_stride": stride,
            "sampled": sampled,
            "native_size": list(native_size) if native_size else None,
            "working_edge": limits.working_edge,
        },
    )


def decode_media_path(
    path: str | Path, limits: DecodeLimits | None = None
) -> IngestResult:
    """Decode a file already on disk.

    Reading a clip into memory and writing it back out to a temporary file costs
    twice the file size before a single frame is decoded, which for a 640 MB raw
    sweep is most of a small workstation's headroom. Batch tooling and any caller
    holding a real path should come through here. The Streamlit uploader hands
    over bytes, so :func:`decode_media` remains for that case.
    """
    limits = limits or DecodeLimits()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No such media file: {source}")
    suffix = source.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return _decode_video_path(source.name, source, limits)
    return decode_media(source.name, source.read_bytes(), limits)


def decode_media(
    name: str, data: bytes, limits: DecodeLimits | None = None
) -> IngestResult:
    limits = limits or DecodeLimits()
    if not data:
        raise ValueError("The uploaded file is empty.")
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return _decode_image(name, data, limits)
    if suffix in VIDEO_SUFFIXES:
        return _decode_video(name, data, limits)
    if suffix in DICOM_SUFFIXES:
        return _decode_dicom(name, data, limits)
    raise ValueError(f"Unsupported file type: {suffix or 'no extension'}")


def quality_metrics(image: Image.Image) -> dict[str, float | int | str]:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32) / 255.0
    gx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    sharpness = float(math.sqrt(float(np.mean(gx * gx)) + float(np.mean(gy * gy))))
    contrast = float(np.std(gray))
    mean_brightness = float(np.mean(gray))
    clipping = float(np.mean((gray < 0.01) | (gray > 0.99)))
    if min(gray.shape) < 256:
        quality = "low resolution"
    elif contrast < 0.08:
        quality = "low contrast"
    elif sharpness < 0.025:
        quality = "possible blur"
    else:
        quality = "reviewable"
    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mean_brightness": round(mean_brightness, 4),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 4),
        "clipped_fraction": round(clipping, 4),
        "quality_flag": quality,
    }

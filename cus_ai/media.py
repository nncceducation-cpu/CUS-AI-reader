from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, ImageSequence


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DICOM_SUFFIXES = {".dcm", ".dicom", ""}


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


def _decode_image(name: str, data: bytes) -> IngestResult:
    image = Image.open(io.BytesIO(data))
    source_frame_count = int(getattr(image, "n_frames", 1) or 1)
    frames: list[MediaFrame] = []
    for index, frame in enumerate(ImageSequence.Iterator(image)):
        frames.append(
            MediaFrame(
                source_name=name,
                frame_index=index,
                image=ImageOps.exif_transpose(frame).convert("RGB"),
                media_type="image",
                technical_metadata={"source_frame_count": source_frame_count},
            )
        )
    technical = {
        "source_frame_count": source_frame_count,
        "decoded_frame_count": len(frames),
        "all_frames_processed": len(frames) == source_frame_count,
    }
    return IngestResult(frames=frames, technical_metadata=technical)


def _decode_dicom(name: str, data: bytes) -> IngestResult:
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
            image=_pil_from_array(frame),
            media_type="dicom",
            pixel_spacing_mm=spacing,
            technical_metadata=technical,
        )
        for index, frame in arrays
    ]
    return IngestResult(frames=frames, warnings=warnings, technical_metadata=technical)


def _decode_video(name: str, data: bytes) -> IngestResult:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video support requires opencv-python-headless.") from exc

    suffix = Path(name).suffix or ".mp4"
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(data)
            path = Path(temp.name)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("The video container could not be opened.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frames: list[MediaFrame] = []
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(
                MediaFrame(
                    source_name=name,
                    frame_index=source_index,
                    image=Image.fromarray(rgb),
                    media_type="video",
                    technical_metadata={"fps": fps, "source_frame_count": total},
                )
            )
            source_index += 1
        capture.release()
        if not frames:
            raise ValueError("No readable frames were found in the clip.")
        complete = total <= 0 or len(frames) == total
        warnings = []
        if not complete:
            warnings.append(
                f"The container reports {total} frames, but only {len(frames)} frames decoded before the stream ended."
            )
        return IngestResult(
            frames=frames,
            warnings=warnings,
            technical_metadata={
                "fps": fps,
                "source_frame_count": total,
                "decoded_frame_count": len(frames),
                "all_frames_processed": complete,
            },
        )
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def decode_media(name: str, data: bytes) -> IngestResult:
    if not data:
        raise ValueError("The uploaded file is empty.")
    suffix = Path(name).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return _decode_image(name, data)
    if suffix in VIDEO_SUFFIXES:
        return _decode_video(name, data)
    if suffix in DICOM_SUFFIXES:
        return _decode_dicom(name, data)
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

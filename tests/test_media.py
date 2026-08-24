import io

import numpy as np
import pytest
from PIL import Image

from cus_ai.media import decode_media, quality_metrics


def make_png(size=(512, 512)):
    buffer = io.BytesIO()
    Image.new("L", size, color=128).save(buffer, format="PNG")
    return buffer.getvalue()


def make_gif(frame_count=9, size=(320, 320)):
    buffer = io.BytesIO()
    frames = [Image.new("L", size, color=20 + index * 10) for index in range(frame_count)]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=40, loop=0)
    return buffer.getvalue()


def test_single_image_ingestion():
    result = decode_media("study.png", make_png())
    assert len(result.frames) == 1
    assert result.frames[0].image.mode == "RGB"


def test_every_multiframe_image_frame_is_decoded_in_order():
    result = decode_media("sweep.gif", make_gif(frame_count=9))
    assert len(result.frames) == 9
    assert [frame.frame_index for frame in result.frames] == list(range(9))
    assert result.technical_metadata["source_frame_count"] == 9
    assert result.technical_metadata["all_frames_processed"] is True


def test_every_video_frame_is_decoded_sequentially(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "sweep.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 320))
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    for index in range(7):
        writer.write(np.full((320, 320, 3), index * 20, dtype=np.uint8))
    writer.release()

    result = decode_media(path.name, path.read_bytes())
    assert len(result.frames) == 7
    assert [frame.frame_index for frame in result.frames] == list(range(7))
    assert result.technical_metadata["all_frames_processed"] is True


def test_quality_metrics_are_bounded_and_identify_dimensions():
    image = Image.open(io.BytesIO(make_png((320, 280))))
    metrics = quality_metrics(image)
    assert metrics["width"] == 320
    assert metrics["height"] == 280
    assert 0 <= metrics["mean_brightness"] <= 1
    assert metrics["quality_flag"] in {"reviewable", "low contrast", "possible blur", "low resolution"}


def test_unsupported_type_fails_closed():
    try:
        decode_media("notes.txt", b"not an image")
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("unsupported media should fail")

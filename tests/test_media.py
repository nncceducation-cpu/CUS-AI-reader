import io

from PIL import Image

from cus_ai.media import decode_media, quality_metrics


def make_png(size=(512, 512)):
    buffer = io.BytesIO()
    Image.new("L", size, color=128).save(buffer, format="PNG")
    return buffer.getvalue()


def test_single_image_ingestion():
    result = decode_media("study.png", make_png())
    assert len(result.frames) == 1
    assert result.frames[0].image.mode == "RGB"


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


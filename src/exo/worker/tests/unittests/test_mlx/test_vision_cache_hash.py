import base64
import io

import mlx.core as mx
from PIL import Image

from exo.shared.types.text_generation import Base64Image
from exo.worker.engines.mlx.vision import (
    VisionProcessor,
    _find_media_regions,  # pyright: ignore[reportPrivateUsage]
    _image_content_hash,  # pyright: ignore[reportPrivateUsage]
    decode_base64_image,
)


def _shape_collision_images() -> tuple[Image.Image, Image.Image]:
    pixels = bytes(range(72))
    landscape = Image.frombytes("RGB", (6, 4), pixels)
    portrait = Image.frombytes("RGB", (4, 6), pixels)
    assert landscape.tobytes() == portrait.tobytes()
    return landscape, portrait


def _as_base64_image(image: Image.Image) -> Base64Image:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Base64Image(base64.b64encode(buffer.getvalue()).decode())


def _as_base64_image_with_exif_orientation(
    image: Image.Image, orientation: int
) -> Base64Image:
    exif = image.getexif()
    exif[274] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", exif=exif)
    return Base64Image(base64.b64encode(buffer.getvalue()).decode())


def test_content_hash_includes_image_dimensions() -> None:
    landscape, portrait = _shape_collision_images()

    assert _image_content_hash(landscape) != _image_content_hash(portrait)


def test_content_hash_is_stable_for_same_canonical_image() -> None:
    landscape, _ = _shape_collision_images()
    encoded = _as_base64_image(landscape)

    assert _image_content_hash(decode_base64_image(encoded)) == _image_content_hash(
        decode_base64_image(encoded)
    )


def test_decode_applies_exif_orientation_and_removes_metadata() -> None:
    landscape, _ = _shape_collision_images()

    decoded = decode_base64_image(
        _as_base64_image_with_exif_orientation(landscape, orientation=6)
    )

    assert decoded.size == (4, 6)
    assert decoded.mode == "RGB"
    assert decoded.info == {}
    assert decoded.getexif().get(274) is None


def test_feature_cache_key_includes_image_dimensions() -> None:
    landscape, portrait = _shape_collision_images()

    landscape_key = VisionProcessor._image_cache_key(  # pyright: ignore[reportPrivateUsage]
        [_as_base64_image(landscape)]
    )
    portrait_key = VisionProcessor._image_cache_key(  # pyright: ignore[reportPrivateUsage]
        [_as_base64_image(portrait)]
    )

    assert landscape_key != portrait_key


def test_feature_cache_key_includes_image_order() -> None:
    landscape, portrait = _shape_collision_images()
    a = _as_base64_image(landscape)
    b = _as_base64_image(portrait)

    forward = VisionProcessor._image_cache_key(  # pyright: ignore[reportPrivateUsage]
        [a, b]
    )
    reverse = VisionProcessor._image_cache_key(  # pyright: ignore[reportPrivateUsage]
        [b, a]
    )

    assert forward != reverse


def test_media_region_hash_includes_image_dimensions() -> None:
    landscape, portrait = _shape_collision_images()
    prompt_tokens = mx.array([1, 99, 99, 2])

    landscape_regions = _find_media_regions(
        prompt_tokens, [_as_base64_image(landscape)], image_token_id=99
    )
    portrait_regions = _find_media_regions(
        prompt_tokens, [_as_base64_image(portrait)], image_token_id=99
    )

    assert len(landscape_regions) == len(portrait_regions) == 1
    assert landscape_regions[0].content_hash != portrait_regions[0].content_hash

from pathlib import Path

from PIL import Image
import pytest

from app.ml.scrfd import ML_ROOT, Scrfd


TEST_IMAGE_DIR = Path("tests/test_data/test_image/scrfd")


class TestScrfd:
    def test_init_sets_expected_weight_path(self) -> None:
        scrfd = Scrfd()

        assert scrfd.weight_path == ML_ROOT / "weights" / "scrfd" / "scrfd.onnx"

    @pytest.mark.parametrize(
        ("image_name", "expected_face_count"),
        [
            ("multi_faces.jpg", 5),
            ("not_has_faces.jpeg", 0),
            ("one_face.png", 1),
        ],
    )
    def test_get_face_counts(
        self, image_name: str, expected_face_count: int
    ) -> None:
        image = Image.open(TEST_IMAGE_DIR / image_name).convert("RGB")

        face_count = Scrfd().get_face_counts(image=image)

        assert face_count == expected_face_count

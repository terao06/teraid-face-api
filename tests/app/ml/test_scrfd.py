from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from app.ml.scrfd import ML_ROOT, Scrfd


TEST_IMAGE_DIR = Path("tests/test_data/images/scrfd")
WEIGHT_PATH = Path("tests/test_data/s3/buckets/weights/scrfd/scrfd.onnx")


class TestScrfd:
    @staticmethod
    def _load_weight_bytes(path: Path) -> BytesIO:
        return BytesIO(path.read_bytes())

    def test_init_sets_expected_weight_bytes(self) -> None:
        scrfd = Scrfd(weight_bytes=self._load_weight_bytes(WEIGHT_PATH))

        assert scrfd.weight_bytes.getbuffer().nbytes == WEIGHT_PATH.stat().st_size

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

        face_count = Scrfd(weight_bytes=self._load_weight_bytes(WEIGHT_PATH)).get_face_counts(image=image)

        assert face_count == expected_face_count

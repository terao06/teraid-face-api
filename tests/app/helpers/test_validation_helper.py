from PIL import Image
import pytest

from app.core.exceptions import FaceNotFoundException, MultipleFacesDetectionException
from app.helpers.validation_helper import ValidationHelper


class TestValidationHelper:
    def test_validation_with_face_returns_none_when_single_face_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = Image.new("RGB", (8, 8))
        warnings: list[str] = []

        monkeypatch.setattr(
            "app.ml.scrfd.Scrfd.get_face_counts",
            lambda self, image: 1,
        )
        monkeypatch.setattr(
            "app.helpers.validation_helper.TeraidFaceApiLog.warning",
            lambda message: warnings.append(message),
        )

        result = ValidationHelper.validation_with_face(image=image)

        assert result is None
        assert warnings == []

    def test_validation_with_face_raises_when_multiple_faces_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = Image.new("RGB", (8, 8))
        warnings: list[str] = []

        monkeypatch.setattr(
            "app.ml.scrfd.Scrfd.get_face_counts",
            lambda self, image: 5,
        )
        monkeypatch.setattr(
            "app.helpers.validation_helper.TeraidFaceApiLog.warning",
            lambda message: warnings.append(message),
        )

        with pytest.raises(MultipleFacesDetectionException):
            ValidationHelper.validation_with_face(image=image)

        assert len(warnings) == 1
        assert warnings[0].endswith("5")

    def test_validation_with_face_raises_when_face_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        image = Image.new("RGB", (8, 8))
        warnings: list[str] = []

        monkeypatch.setattr(
            "app.ml.scrfd.Scrfd.get_face_counts",
            lambda self, image: 0,
        )
        monkeypatch.setattr(
            "app.helpers.validation_helper.TeraidFaceApiLog.warning",
            lambda message: warnings.append(message),
        )

        with pytest.raises(FaceNotFoundException):
            ValidationHelper.validation_with_face(image=image)

        assert len(warnings) == 1
        assert warnings[0].endswith("0")

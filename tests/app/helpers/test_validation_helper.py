from unittest.mock import MagicMock, patch

from PIL import Image
import pytest

from app.core.exceptions import FaceNotFoundException, MultipleFacesDetectionException
from app.core.logging import TeraidFaceApiLog
from app.helpers.validation_helper import ValidationHelper
from app.ml.scrfd import Scrfd


class TestValidationHelper:
    @patch.object(TeraidFaceApiLog, "warning")
    @patch.object(Scrfd, "get_face_counts", return_value=1)
    def test_validation_with_face_returns_none_when_single_face_detected(
        self,
        _mock_get_face_counts: MagicMock,
        mock_warning: MagicMock,
    ) -> None:
        image = Image.new("RGB", (8, 8))

        result = ValidationHelper.validation_with_face(image=image)

        assert result is None
        mock_warning.assert_not_called()
        _mock_get_face_counts.assert_called_once()


    @patch.object(TeraidFaceApiLog, "warning")
    @patch.object(Scrfd, "get_face_counts", return_value=5)
    def test_validation_with_face_raises_when_multiple_faces_detected(
        self,
        _mock_get_face_counts: MagicMock,
        mock_warning: MagicMock,
    ) -> None:
        image = Image.new("RGB", (8, 8))

        with pytest.raises(MultipleFacesDetectionException):
            ValidationHelper.validation_with_face(image=image)

        mock_warning.assert_called_once()
        _mock_get_face_counts.assert_called_once()
        assert mock_warning.call_args.args[0].endswith("5")

    @patch.object(TeraidFaceApiLog, "warning")
    @patch.object(Scrfd, "get_face_counts", return_value=0)
    def test_validation_with_face_raises_when_face_is_not_found(
        self,
        _mock_get_face_counts: MagicMock,
        mock_warning: MagicMock,
    ) -> None:
        image = Image.new("RGB", (8, 8))

        with pytest.raises(FaceNotFoundException):
            ValidationHelper.validation_with_face(image=image)

        mock_warning.assert_called_once()
        _mock_get_face_counts.assert_called_once()
        assert mock_warning.call_args.args[0].endswith("0")

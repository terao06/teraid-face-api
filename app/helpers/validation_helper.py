from PIL import Image

from app.core.exceptions import FaceNotFoundException, MultipleFacesDetectionException
from app.core.logging import TeraidFaceApiLog
from app.ml.scrfd import Scrfd


class ValidationHelper:
    @classmethod
    def validation_with_face(cls, image: Image.Image) -> None:
        face_counts = Scrfd().get_face_counts(image=image)

        if face_counts > 1:
            TeraidFaceApiLog.warning(
                f"顔が複数検出されました。顔検出数 = {face_counts}"
            )
            raise MultipleFacesDetectionException(
                "顔が複数検出されました。"
            )

        if face_counts == 0:
            TeraidFaceApiLog.warning(
                f"顔を検出できませんでした。顔検出数 = {face_counts}"
            )
            raise FaceNotFoundException("顔を検出できませんでした。")

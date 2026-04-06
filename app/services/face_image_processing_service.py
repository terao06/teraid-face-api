import base64
from io import BytesIO
import io

import numpy as np
from PIL import Image

from app.ml.retinexformer import Retinexformer
from app.ml.gfpgan import Gfpgan
from app.ml.realesrgan import RealEsrGan
from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.helpers.validation_helper import ValidationHelper


class FaceImageProcessingService:
    def processing(
            self,
            content: str,
            extension: ExtensionType,
            use_brightness_adjustment_lm: bool,
            use_correction_lm: bool,
            use_resolution_lm: bool,
            ) -> FaceImageProcessingResponse:

        image_bytes = base64.b64decode(content)
        target_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # バリデーション実施
        ValidationHelper.validation_with_face(image=target_image)

        target_image_np = np.asarray(target_image)
        processing_image_np = target_image_np.copy()


        normalized_image_np = target_image_np.astype(np.float32) / 255.0
        if use_brightness_adjustment_lm:
            retinexformer = Retinexformer()
            processing_image_np = retinexformer.processing(image_np=normalized_image_np)

        if use_correction_lm:
            gfpgan = Gfpgan()
            processing_image_np = gfpgan.processing(image_np=processing_image_np)

        if use_resolution_lm:
            realesrgan = RealEsrGan()
            processing_image_np = realesrgan.processing(image_np=processing_image_np)

        processing_image = Image.fromarray(processing_image_np)
        base64_processing_image = self._pil_image_to_base64(image=processing_image, extension=extension)
        return FaceImageProcessingResponse(
            content=base64_processing_image,
            extension=extension,
            size_bytes=self._get_image_size_bytes(img=processing_image, format=extension)
        )

    def _pil_image_to_base64(
        self,
        image: Image.Image,
        extension: ExtensionType,
        quality: int = 95
    ) -> str:
        """
        PIL.Image → base64文字列
        """

        try:
            buffer = BytesIO()

            if extension == ExtensionType.JPEG:
                image = image.convert("RGB")
                image.save(buffer, format=extension.value, quality=quality)
            else:
                image.save(buffer, format=extension.value)

            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        except Exception as e:
            raise ValueError(f"Failed to encode image: {e}")

    def _get_image_size_bytes(self, img: Image.Image, format: ExtensionType):
        buffer = io.BytesIO()
        img.save(buffer, format=format.value)
        size_bytes = buffer.tell()
        size_mb = size_bytes
        return size_mb

import base64
from io import BytesIO
import io

import numpy as np
from PIL import Image

from app.models.requests.face_image_processing_request import ExtensionType
from app.models.responses.face_image_processing_response import FaceImageProcessingResponse
from app.helpers.validation_helper import ValidationHelper
from app.ml.retinexformer import Retinexformer
from app.ml.gfpgan import Gfpgan
from app.ml.realesrgan import RealEsrGan
from app.core.aws.ssm_client import SsmClient
from app.core.aws.s3_client import S3Client


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
        ssm_params = SsmClient()
        s3_client = S3Client(s3_endpoint=ssm_params.s3_endpoint)
        scrfd_weight_bytes = s3_client.get_object(
            bucket_name=ssm_params.llm_weight_bucket,
            key=ssm_params.scrfd_weight
        )

        # バリデーション実施
        ValidationHelper.validation_with_face(image=target_image, scrfd_weight_bytes=scrfd_weight_bytes)

        target_image_np = np.asarray(target_image)
        processing_image_np = target_image_np.copy()

        normalized_image_np = target_image_np.astype(np.float32) / 255.0
        if use_brightness_adjustment_lm:
            retinex_weight_bytes = s3_client.get_object(
                bucket_name=ssm_params.llm_weight_bucket,
                key=ssm_params.retinexformer_weight
            )

            retinexformer = Retinexformer(weight_bytes=BytesIO(retinex_weight_bytes))
            processing_image_np = retinexformer.processing(image_np=normalized_image_np)

        if use_correction_lm:
            gfp_weight_bytes = s3_client.get_object(
                bucket_name=ssm_params.llm_weight_bucket,
                key=ssm_params.gfpgan_nv_weight
            )
            resnet_weight_bytes = s3_client.get_object(
                bucket_name=ssm_params.llm_weight_bucket,
                key=ssm_params.gfpgan_resnet_weight
            )
            parsing_wight_bytes = s3_client.get_object(
                bucket_name=ssm_params.llm_weight_bucket,
                key=ssm_params.gfpgan_parsenet_weight
            )

            gfpgan = Gfpgan(
                weight_bytes=BytesIO(gfp_weight_bytes),
                resnet_weight_bytes=BytesIO(resnet_weight_bytes),
                parsing_wight_bytes=BytesIO(parsing_wight_bytes)
            )

            processing_image_np = gfpgan.processing(image_np=processing_image_np)

        if use_resolution_lm:
            parsing_wight_bytes = s3_client.get_object(
                bucket_name=ssm_params.llm_weight_bucket,
                key=ssm_params.realesrgan_weight
            )
            realesrgan = RealEsrGan(weight_bytes=BytesIO(parsing_wight_bytes))
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

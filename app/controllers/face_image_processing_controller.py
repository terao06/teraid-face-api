from fastapi import HTTPException

from app.services.face_image_processing_service import FaceImageProcessingService
from app.models.requests.face_image_processing_request import FaceImageProcessingRequest
from app.core.exceptions import (
    FaceNotFoundException,
    MultipleFacesDetectionException,
    FaceAlignmentError
)
from app.core.messages import ValidationMessages


class FaceImageProcessingController:
    def processing(self, request: FaceImageProcessingRequest):
        try:
            face_image_processing_service = FaceImageProcessingService()
            return face_image_processing_service.processing(
                content=request.content,
                extension=request.extension,
                use_angle_correction=request.use_angle_correction,
                use_brightness_adjustment_lm=request.use_brightness_adjustment_lm,
                use_correction_lm=request.use_correction_lm,
                use_resolution_lm=request.use_resolution_lm
            )
        except FaceNotFoundException:
            raise HTTPException(status_code=404, detail=ValidationMessages.FACE_NOT_FOUND)

        except FaceAlignmentError:
            raise HTTPException(status_code=404, detail=ValidationMessages.ANGLE_CORRECTION_ERROR)

        except MultipleFacesDetectionException:
            raise HTTPException(status_code=409, detail=ValidationMessages.MULTIPLE_FACES_DETECTED)

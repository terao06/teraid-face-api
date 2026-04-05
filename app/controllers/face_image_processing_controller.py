from app.services.face_image_processing_service import FaceImageProcessingService
from app.models.requests.face_image_processing_request import FaceImageProcessingRequest


class FaceImageProcessingController:
    def processing(self, request: FaceImageProcessingRequest):
        face_image_processing_service = FaceImageProcessingService()
        return face_image_processing_service.processing(
            content=request.content,
            extension=request.extension,
            use_brightness_adjustment=request.use_brightness_adjustment,
            use_correction=request.use_correction
        )


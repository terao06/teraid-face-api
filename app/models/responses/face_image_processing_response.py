from pydantic import BaseModel, Field
from app.models.requests.face_image_processing_request import ExtensionType


class FaceImageProcessingResponse(BaseModel):
    content: str = Field(..., description="加工対象の顔画像")
    extension: ExtensionType = Field(..., description="画像拡張子")

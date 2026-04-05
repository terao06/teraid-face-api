from enum import Enum
from pydantic import BaseModel, Field


class ExtensionType(Enum):
    JPEG = "jpeg"
    PNG = "png"

class FaceImageProcessingRequest(BaseModel):
    content: str = Field(..., description="加工対象の顔画像")
    extension: ExtensionType = Field(..., description="画像拡張子")
    use_brightness_adjustment: bool = Field(..., description="明るさ調整機能利用有無")
    use_correction: bool = Field(..., description="補正機能利用有無")

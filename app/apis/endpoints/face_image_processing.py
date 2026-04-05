from fastapi import APIRouter
from app.middlewares.request_wrapper import request_rapper
from app.middlewares.response_wrapper import response_rapper
from app.controllers.face_image_processing_controller import FaceImageProcessingController
from app.models.requests.face_image_processing_request import FaceImageProcessingRequest


face_image_router = APIRouter()

@face_image_router.post("/")
@response_rapper()
@request_rapper()
def face_image_process(request: FaceImageProcessingRequest):
    """
    顔画像修正を加工する。
    
    response_wrapper()デコレーターによって、
    レスポンスは自動的に {"status": "success", "data": {...}} の形式にラップされます。
    """

    return FaceImageProcessingController().processing(
        request=request
    )

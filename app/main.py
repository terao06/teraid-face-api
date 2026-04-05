from fastapi import FastAPI

from app.core.logging import TeraidFaceApiLog
from app.apis.endpoints.face_image_processing import face_image_router
from fastapi.middleware.cors import CORSMiddleware

# ロギング初期設定
TeraidFaceApiLog.setup(
    log_level='INFO',
    enable_file_logging=False  # 必要に応じてTrueに変更
)

app = FastAPI(
    title="teraid Face API",
    description="顔画像修正用APIです",
    version="1.0.0",
)

origins = [
    "http://localhost:3000", # 例えばローカル開発
    # 他の必要なオリジンもここに追加
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # 許可するオリジンのリスト
    allow_credentials=True,       # Cookie等の資格情報も許可する場合に設定
    allow_methods=["GET", "POST", "PUT", "DELETE"],     # 許可するHTTPメソッド（"GET", "POST" など）; "*" は全てを許可
    allow_headers=["*"],          # 許可するHTTPヘッダー; "*" は全てを許可
)

app.include_router(prefix="/face-image-process", router=face_image_router)

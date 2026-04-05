from functools import wraps
from pydantic import BaseModel
from fastapi import HTTPException


def response_rapper(data_key: str = "data"):
    """
    レスポンスを統一された形式でラップするデコレーター
    成功時のデータとエラーメッセージを標準化
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Pydantic モデルのインスタンスの場合
                if isinstance(result, BaseModel):
                    result = result.model_dump()

                # リスト内の Pydantic モデルも辞書に変換
                elif isinstance(result, list):
                    result = [item.model_dump() if isinstance(item, BaseModel) else item for item in result]

                # 成功した場合のレスポンス
                response = {
                    "status": "success"
                }
                if result is not None:
                    response[data_key] = result
                return response

            except HTTPException as e:
                raise e
            except Exception as e:
                raise e
        return wrapper
    return decorator

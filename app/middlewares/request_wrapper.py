from functools import wraps
from app.core.logging import TeraidFaceApiLog
import json


def _convert_to_dict(obj):
    """
    オブジェクトを辞書に変換するヘルパー関数
    """
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return obj


def request_rapper():
    """
    リクエストボディを指定されたPydanticモデルでバリデーションするデコレーター
    FastAPIを使用する場合、バリデーションとデータの注入は依存性注入(Depends)によって行われるため、
    このラッパーは主に互換性の維持や追加の同期処理のために使用されます。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            TeraidFaceApiLog.info(f"Function: {func_name}")
            # Pydanticモデルや基本型のパラメータをログ出力
            log_params = {}
            for key, value in kwargs.items():
                # Requestオブジェクトなどの大きなオブジェクトは除外
                if key not in ['request', 'response', 'db', 'session']:
                    try:
                        log_params[key] = _convert_to_dict(value)
                    except Exception as e:
                        log_params[key] = str(type(value))
            
            if log_params:
                # センシティブな情報をマスキング
                masked_params = TeraidFaceApiLog.mask_sensitive_data(log_params)
                # JSON形式で出力
                try:
                    json_str = json.dumps(masked_params, ensure_ascii=False, indent=None)
                    TeraidFaceApiLog.info(f"Request parameters: {json_str}")
                except (TypeError, ValueError) as e:
                    # JSON化できない場合は文字列化
                    TeraidFaceApiLog.info(f"Request parameters: {masked_params}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

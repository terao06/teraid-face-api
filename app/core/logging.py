import logging
import sys
import os
from typing import Optional, Any
from datetime import datetime
import json
import inspect


class TeraidFaceApiLog:
    """
    アプリケーション全体で使用するロギングクラス
    ベストプラクティスに基づいた設定と機能を提供
    クラスメソッドとして直接ログ出力可能
    """
    
    _loggers = {}
    _initialized = False
    _default_logger = None
    
    # センシティブな情報をマスキングするためのキーワードリスト
    SENSITIVE_KEYS = {'client_id', 'secret_hash', 'client_secret'}
    
    @classmethod
    def mask_sensitive_data(cls, data: Any, mask_string: str = "***") -> Any:
        """
        センシティブな情報をマスキングする
        
        Args:
            data: マスキング対象のデータ
            mask_string: マスキング文字列（デフォルト: "***"）
            
        Returns:
            マスキング済みのデータ
        """
        if isinstance(data, dict):
            masked_data = {}
            for key, value in data.items():
                # キー名が小文字でセンシティブキーワードに一致するか確認
                key_lower = key.lower()
                is_sensitive = any(sensitive_key in key_lower for sensitive_key in cls.SENSITIVE_KEYS)
                
                if is_sensitive:
                    masked_data[key] = mask_string
                elif isinstance(value, (dict, list)):
                    # ネストされた構造も再帰的にマスキング
                    masked_data[key] = cls.mask_sensitive_data(value, mask_string)
                else:
                    masked_data[key] = value
            return masked_data
        elif isinstance(data, list):
            return [cls.mask_sensitive_data(item, mask_string) for item in data]
        else:
            return data
    
    @classmethod
    def setup(
        cls,
        log_level: Optional[str] = None,
        log_format: Optional[str] = None,
        enable_file_logging: bool = False,
        log_file_path: Optional[str] = None
    ):
        """
        ロギングの初期設定を行う（アプリケーション起動時に一度だけ呼び出す）
        
        Args:
            log_level: ログレベル (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: ログフォーマット
            enable_file_logging: ファイルへのログ出力を有効化
            log_file_path: ログファイルのパス
        """
        if cls._initialized:
            return
        
        # 環境変数から設定を取得（デフォルト値も設定）
        log_level = log_level or os.getenv('LOG_LEVEL', 'INFO')
        log_format = log_format or os.getenv(
            'LOG_FORMAT',
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # ルートロガーの設定
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 既存のハンドラーをクリア
        root_logger.handlers.clear()
        
        # コンソール出力ハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # ファイル出力ハンドラー（オプション）
        if enable_file_logging:
            log_file_path = log_file_path or os.getenv('LOG_FILE_PATH', 'logs/app.log')
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        
        cls._initialized = True
        cls._default_logger = root_logger
    
    @classmethod
    def _get_caller_logger(cls) -> logging.Logger:
        """
        呼び出し元のモジュール名からロガーを取得
        """
        if not cls._initialized:
            cls.setup()
        
        # 呼び出し元のフレームを取得（2つ上のスタックフレーム）
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            module_name = caller_frame.f_globals.get('__name__', '__main__')
            
            if module_name not in cls._loggers:
                cls._loggers[module_name] = logging.getLogger(module_name)
            
            return cls._loggers[module_name]
        finally:
            del frame
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        指定された名前のロガーを取得
        
        Args:
            name: ロガー名（通常は__name__を使用）
            
        Returns:
            設定済みのロガーインスタンス
        """
        if not cls._initialized:
            cls.setup()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def debug(cls, message: str, *args, **kwargs):
        """デバッグレベルのログ出力"""
        logger = cls._get_caller_logger()
        logger.debug(message, *args, **kwargs)
    
    @classmethod
    def info(cls, message: str, *args, **kwargs):
        """情報レベルのログ出力"""
        logger = cls._get_caller_logger()
        logger.info(message, *args, **kwargs)
    
    @classmethod
    def warning(cls, message: str, *args, **kwargs):
        """警告レベルのログ出力"""
        logger = cls._get_caller_logger()
        logger.warning(message, *args, **kwargs)
    
    @classmethod
    def error(cls, message: str, *args, **kwargs):
        """エラーレベルのログ出力"""
        logger = cls._get_caller_logger()
        logger.error(message, *args, **kwargs)
    
    @classmethod
    def critical(cls, message: str, *args, **kwargs):
        """クリティカルレベルのログ出力"""
        logger = cls._get_caller_logger()
        logger.critical(message, *args, **kwargs)
    
    @classmethod
    def exception(cls, message: str, *args, **kwargs):
        """例外情報付きエラーログ出力"""
        logger = cls._get_caller_logger()
        logger.exception(message, *args, **kwargs)
    
    @classmethod
    def request(cls, method: str, path: str, params: dict):
        """
        リクエスト情報をログ出力（構造化ログ）
        センシティブな情報は自動的にマスキングされます
        
        Args:
            method: HTTPメソッド
            path: リクエストパス
            params: リクエストパラメータ
        """
        logger = cls._get_caller_logger()
        # センシティブな情報をマスキング
        masked_params = cls.mask_sensitive_data(params)
        log_data = {
            'type': 'request',
            'method': method,
            'path': path,
            'params': masked_params,
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.info(f"REQUEST: {json.dumps(log_data, ensure_ascii=False, default=str)}")
    
    @classmethod
    def response(cls, method: str, path: str, status_code: int, duration_ms: float):
        """
        レスポンス情報をログ出力（構造化ログ）
        
        Args:
            method: HTTPメソッド
            path: リクエストパス
            status_code: HTTPステータスコード
            duration_ms: 処理時間（ミリ秒）
        """
        logger = cls._get_caller_logger()
        log_data = {
            'type': 'response',
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.info(f"RESPONSE: {json.dumps(log_data, ensure_ascii=False, default=str)}")
    
    @classmethod
    def error_detail(cls, error: Exception, context: Optional[dict] = None):
        """
        エラー情報をログ出力（構造化ログ）
        センシティブな情報は自動的にマスキングされます
        
        Args:
            error: 例外オブジェクト
            context: 追加のコンテキスト情報
        """
        logger = cls._get_caller_logger()
        # センシティブな情報をマスキング
        masked_context = cls.mask_sensitive_data(context or {})
        log_data = {
            'type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': masked_context,
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.error(f"ERROR: {json.dumps(log_data, ensure_ascii=False, default=str)}", exc_info=True)
    
    @classmethod
    def business(cls, event: str, data: dict):
        """
        ビジネスロジックのイベントをログ出力（構造化ログ）
        センシティブな情報は自動的にマスキングされます
        
        Args:
            event: イベント名
            data: イベントデータ
        """
        logger = cls._get_caller_logger()
        # センシティブな情報をマスキング
        masked_data = cls.mask_sensitive_data(data)
        log_data = {
            'type': 'business',
            'event': event,
            'data': masked_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.info(f"BUSINESS: {json.dumps(log_data, ensure_ascii=False, default=str)}")

import boto3
import json
import os
from botocore.exceptions import ClientError
from typing import Any, Dict, Union

class SsmClient:
    s3_endpoint: str
    llm_weight_bucket: str
    gfpgan_nv_weight: str
    gfpgan_resnet_weight: str
    gfpgan_parsenet_weight: str
    realesrgan_weight: str
    retinexformer_weight: str
    scrfd_weight: str

    def __init__(self):
        """
        SSM クライアントの初期化
        """
        self.region_name = os.getenv("AWS_REGION", "ap-northeast-1")
        self.endpoint_url = os.getenv("SSM_ENDPOINT", "http://localstack:4566")

        self.client = boto3.client(
            service_name='ssm',
            region_name=self.region_name,
            endpoint_url=self.endpoint_url
        )

        self.s3_endpoint = self._get_string_parameter(name="s3_endpoint")
        self.llm_weight_bucket = self._get_string_parameter(name="llm_weight_bucket")
        self.gfpgan_nv_weight = self._get_string_parameter(name="gfpgan_nv_weight")
        self.gfpgan_resnet_weight = self._get_string_parameter(name="gfpgan_resnet_weight")
        self.gfpgan_parsenet_weight = self._get_string_parameter(name="gfpgan_parsenet_weight")
        self.realesrgan_weight = self._get_string_parameter(name="realesrgan_weight")
        self.retinexformer_weight = self._get_string_parameter(name="retinexformer_weight")
        self.scrfd_weight = self._get_string_parameter(name="scrfd_weight")

    def _get_parameter(self, name: str, with_decryption: bool = True) -> Union[str, Dict[str, Any]]:
        """
        指定されたパラメータ名で値を取得する。
        JSON形式の場合は辞書型にパースして返す。
        """
        try:
            response = self.client.get_parameter(
                Name=name,
                WithDecryption=with_decryption
            )
            value = response['Parameter']['Value']
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except ClientError as e:
            raise e

    def _get_string_parameter(self, name: str, with_decryption: bool = True) -> str:
        value = self._get_parameter(name=name, with_decryption=with_decryption)
        if not isinstance(value, str):
            raise ValueError(f"SSM parameter '{name}' must be a string.")

        return value

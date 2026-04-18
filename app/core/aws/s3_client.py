import boto3
import os


class S3Client:
    def __init__(self, s3_endpoint: str):
        region_name = os.getenv("AWS_REGION", "ap-northeast-1")
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "dummy")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "dummy123")

        self.client = boto3.client(
            service_name='s3',
            region_name=region_name,
            endpoint_url=s3_endpoint,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def get_object(
        self,
        bucket_name: str,
        key: str) -> bytes:

        response = self.client.get_object(Bucket=bucket_name, Key=key)
        return response["Body"].read()

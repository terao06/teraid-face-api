from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import ClientError


BASE_DIR = Path(__file__).resolve().parent
BUCKETS_DIR = BASE_DIR / "buckets"
ENDPOINT_URL = "http://localhost:9002"
REGION_NAME = "ap-northeast-1"
AWS_ACCESS_KEY_ID = "dummy"
AWS_SECRET_ACCESS_KEY = "dummy123"


def iter_bucket_files(buckets_dir: Path) -> list[tuple[str, Path, str]]:
    uploads: list[tuple[str, Path, str]] = []

    for bucket_dir in sorted(path for path in buckets_dir.iterdir() if path.is_dir()):
        bucket_name = bucket_dir.name
        for file_path in sorted(path for path in bucket_dir.rglob("*") if path.is_file()):
            object_key = file_path.relative_to(bucket_dir).as_posix()
            uploads.append((bucket_name, file_path, object_key))

    return uploads


def upload_mock_s3(
    buckets_dir: Path = BUCKETS_DIR,
) -> None:
    s3_client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    created_buckets: set[str] = set()
    for bucket_name, file_path, object_key in iter_bucket_files(buckets_dir=buckets_dir):
        if bucket_name not in created_buckets:
            create_bucket_params: dict[str, object] = {"Bucket": bucket_name}
            if REGION_NAME != "us-east-1":
                create_bucket_params["CreateBucketConfiguration"] = {
                    "LocationConstraint": REGION_NAME,
                }
            try:
                s3_client.create_bucket(**create_bucket_params)
            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]
                if error_code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                    raise
            created_buckets.add(bucket_name)

        s3_client.upload_file(str(file_path), bucket_name, object_key)


def main() -> None:
    upload_mock_s3()


if __name__ == "__main__":
    main()

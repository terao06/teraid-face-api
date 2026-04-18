from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3


BASE_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = BASE_DIR / "face_api_setting.json"
ENDPOINT_URL = os.getenv("SSM_ENDPOINT", "http://localhost:4566")
REGION_NAME = os.getenv("AWS_REGION", "ap-northeast-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "dummy")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "dummy123")


def load_parameters(settings_file: Path) -> dict[str, str]:
    with settings_file.open(encoding="utf-8") as file:
        settings: dict[str, Any] = json.load(file)

    parameters: dict[str, str] = {}
    for name, value in settings.items():
        if isinstance(value, dict):
            parameters[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            continue

        parameters[name] = str(value)

    return parameters


def put_mock_ssm_parameters(settings_file: Path = SETTINGS_FILE) -> None:
    ssm_client = boto3.client(
        "ssm",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION_NAME,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    for name, value in load_parameters(settings_file=settings_file).items():
        ssm_client.put_parameter(
            Name=name,
            Value=value,
            Type="String",
            Overwrite=True,
        )


def main() -> None:
    put_mock_ssm_parameters()


if __name__ == "__main__":
    main()

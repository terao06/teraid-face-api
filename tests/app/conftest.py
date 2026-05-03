import os
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import app.services.face_image_processing_service as face_image_processing_service_module
from tests.test_data.s3.build_s3 import (
    AWS_ACCESS_KEY_ID as S3_AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY as S3_AWS_SECRET_ACCESS_KEY,
    ENDPOINT_URL as S3_ENDPOINT_URL,
    REGION_NAME as S3_REGION_NAME,
    upload_mock_s3,
)
from tests.test_data.ssm.build_ssm import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    ENDPOINT_URL,
    REGION_NAME,
    put_mock_ssm_parameters,
)


@pytest.fixture(scope="session")
def initialize_ssm() -> Iterator[None]:
    original_values = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": os.environ.get("AWS_REGION"),
        "SSM_ENDPOINT": os.environ.get("SSM_ENDPOINT"),
    }
#
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    os.environ["AWS_REGION"] = REGION_NAME
    os.environ["SSM_ENDPOINT"] = ENDPOINT_URL

    put_mock_ssm_parameters()

    yield

    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
            continue

        os.environ[key] = value


@pytest.fixture(scope="session")
def initialize_s3() -> Iterator[None]:
    original_values = {
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": os.environ.get("AWS_REGION"),
    }

    os.environ["AWS_ACCESS_KEY_ID"] = S3_AWS_ACCESS_KEY_ID
    os.environ["AWS_SECRET_ACCESS_KEY"] = S3_AWS_SECRET_ACCESS_KEY
    os.environ["AWS_REGION"] = S3_REGION_NAME

    upload_mock_s3()

    yield

    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
            continue

        os.environ[key] = value


@pytest.fixture
def mock_ssm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    ssm_client_mock = MagicMock(
        return_value=SimpleNamespace(
            s3_endpoint=S3_ENDPOINT_URL,
            llm_weight_bucket="weights",
            gfpgan_nv_weight="gfpgan/GFPGANv.pth",
            gfpgan_resnet_weight="gfpgan/detection_Resnet50_Final.pth",
            gfpgan_parsenet_weight="gfpgan/parsing_parsenet.pth",
            realesrgan_weight="realesrgan/RealESRGAN_x2plus.pth",
            retinexformer_weight="retinexformer/MST_Plus_Plus_8x1150.pth",
            scrfd_weight="scrfd/scrfd.onnx",
            face_alignment_weight="facealignment/face_landmarker.task",
        )
    )
    monkeypatch.setattr(face_image_processing_service_module, "SsmClient", ssm_client_mock)
    return ssm_client_mock

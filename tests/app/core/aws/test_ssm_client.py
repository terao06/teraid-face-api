import pytest
from botocore.exceptions import ClientError

from app.core.aws.ssm_client import SsmClient


@pytest.mark.usefixtures("initialize_ssm")
class TestSsmClient:
    def test_init(self) -> None:
        ssm_client = SsmClient()

        assert ssm_client.s3_endpoint == "http://teraid-face-api-s3:9000"
        assert ssm_client.llm_weight_bucket == "weights"
        assert ssm_client.gfpgan_nv_weight == "gfpgan/GFPGANv.pth"
        assert ssm_client.gfpgan_resnet_weight == "gfpgan/detection_Resnet50_Final.pth"
        assert ssm_client.gfpgan_parsenet_weight == "gfpgan/parsing_parsenet.pth"
        assert ssm_client.realesrgan_weight == "realesrgan/RealESRGAN_x2plus.pth"
        assert ssm_client.retinexformer_weight == "retinexformer/MST_Plus_Plus_8x1150.pth"
        assert ssm_client.scrfd_weight == "scrfd/scrfd.onnx"

    def test_get_parameter(self) -> None:
        SsmClient()._get_parameter("s3_endpoint") == "http://teraid-face-api-s3:9000"

    def test_get_parameter_with_client_error(self) -> None:
        ssm_client = SsmClient()

        with pytest.raises(ClientError) as exc_info:
            ssm_client._get_parameter(name="/missing-parameter")

        assert exc_info.value.response["Error"]["Code"] == "ParameterNotFound"

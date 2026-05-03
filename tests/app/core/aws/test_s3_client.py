import pytest
from botocore.exceptions import ClientError

from app.core.aws.s3_client import S3Client
from tests.test_data.s3.build_s3 import ENDPOINT_URL


@pytest.mark.usefixtures("initialize_s3")
class TestS3Client:
    def test_init(self) -> None:
        s3_client = S3Client(s3_endpoint=ENDPOINT_URL)

        assert s3_client.client.meta.endpoint_url == ENDPOINT_URL

    def test_get_object(self) -> None:
        s3_client = S3Client(s3_endpoint=ENDPOINT_URL)

        content = s3_client.get_object(
            bucket_name="weights",
            key="gfpgan/Readme.md",
        )

        assert content.startswith(b"# GFPGAN weights")

    def test_get_object_with_missing_key(self) -> None:
        s3_client = S3Client(s3_endpoint=ENDPOINT_URL)

        with pytest.raises(ClientError) as exc_info:
            s3_client.get_object(bucket_name="weights", key="missing-object")

        assert exc_info.value.response["Error"]["Code"] == "NoSuchKey"

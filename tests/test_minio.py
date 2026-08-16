import uuid

import pytest

from api.client.minio_client import MinioClient  # adjust import if needed

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture(scope="module")
def minio_client():
    bucket_name = f"test-bucket-{uuid.uuid4().hex}"
    dummy_config = {
        "endpoint": "localhost:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "bucket_name": bucket_name,
        "secure": False,
    }

    client = MinioClient(config_dict=dummy_config)

    yield client


# -----------------------------
# Tests
# -----------------------------


def test_upload_and_list(minio_client):
    data = b"hello world"
    object_name = "test.txt"

    uploaded_object_name = minio_client.upload_bytes(data, object_name)

    objects = minio_client.list_objects()
    assert uploaded_object_name in objects


def test_download_file(minio_client, tmp_path):
    data = b"download test"
    object_name = "download.txt"

    uploaded_object_name = minio_client.upload_bytes(data, object_name)

    download_path = tmp_path / "downloaded.txt"
    minio_client.download_file(uploaded_object_name, str(download_path))

    assert download_path.exists()
    assert download_path.read_bytes() == data


def test_delete_object(minio_client):
    data = b"delete test"
    object_name = "delete.txt"

    uploaded_object_name = minio_client.upload_bytes(data, object_name)
    minio_client.delete_object(uploaded_object_name)

    objects = minio_client.list_objects()
    assert object_name not in objects

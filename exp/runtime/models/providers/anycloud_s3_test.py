"""S3 trainer artifact store tests with a stub signing client."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from exp.runtime.models.providers.anycloud_s3 import S3TrainerArtifactStore, parse_s3_location


class _SigningClient:
    """Record signing requests and return deterministic fake signed URLs."""

    def __init__(self) -> None:
        """Initialize an empty request journal."""
        self.calls: list[tuple[str, dict[str, str], int]] = []

    def generate_presigned_url(
        self, ClientMethod: str, Params: Mapping[str, str], ExpiresIn: int
    ) -> str:
        """Return a fake signed URL for one bucket and key."""
        self.calls.append((ClientMethod, dict(Params), ExpiresIn))
        return f"https://{Params['Bucket']}.example/{Params['Key']}?X-Amz-Signature=secret"


def _store(client: _SigningClient) -> S3TrainerArtifactStore:
    return S3TrainerArtifactStore(
        "s3://trainer-bucket/projects/p1/", region="us-west-2", client=client
    )


def test_upload_returns_opaque_identity_below_the_prefix_and_keeps_url_secret() -> None:
    client = _SigningClient()

    upload = _store(client).begin_upload(kind="state", name="step-000001.pt")

    assert upload.resource_id == "s3://trainer-bucket/projects/p1/state/step-000001.pt"
    assert "secret" not in upload.resource_id
    assert "secret" not in repr(upload)
    assert upload.upload_url.get_secret_value().endswith("X-Amz-Signature=secret")
    assert client.calls == [
        (
            "put_object",
            {
                "Bucket": "trainer-bucket",
                "Key": "projects/p1/state/step-000001.pt",
                "ContentType": "application/octet-stream",
            },
            3600,
        )
    ]


@pytest.mark.parametrize("name", ["", ".", "..", "nested/name"])
def test_upload_rejects_names_that_escape_one_path_component(name: str) -> None:
    with pytest.raises(ValueError, match="one safe path component"):
        _store(_SigningClient()).begin_upload(kind="sampling", name=name)


def test_download_signs_only_identities_below_the_configured_prefix() -> None:
    client = _SigningClient()
    store = _store(client)

    url = store.download_url("s3://trainer-bucket/projects/p1/sampling/final.tar.gz")

    assert url.get_secret_value().endswith("X-Amz-Signature=secret")
    assert client.calls[0][:2] == (
        "get_object",
        {"Bucket": "trainer-bucket", "Key": "projects/p1/sampling/final.tar.gz"},
    )
    for outside in (
        "s3://other-bucket/projects/p1/state/x.pt",
        "s3://trainer-bucket/projects/p2/state/x.pt",
        "s3://trainer-bucket/projects/p1",
    ):
        with pytest.raises(ValueError, match="outside the configured artifact prefix"):
            store.download_url(outside)


@pytest.mark.parametrize(
    "location",
    [
        "https://trainer-bucket/key",
        "s3://trainer-bucket",
        "s3://trainer-bucket/",
        "s3://trainer-bucket/key?versionId=1",
        "s3://user:pass@trainer-bucket/key",
    ],
)
def test_parse_rejects_non_s3_or_authorized_locations(location: str) -> None:
    with pytest.raises(ValueError, match="S3 locations"):
        parse_s3_location(location)

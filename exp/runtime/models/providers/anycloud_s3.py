"""S3 artifact store that authorizes AnyCloud trainer transfers with short-lived URLs.

The store signs one upload or download at a time with the caller's ambient AWS credentials. It
returns only opaque ``s3://bucket/key`` identities for Experiential's SFT evidence, while the
signed URLs stay in memory and reach the trainer service only for one transfer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from pydantic import SecretStr

_URL_LIFETIME_SECONDS = 3600


@dataclass(frozen=True)
class S3ArtifactUpload:
    """One opaque S3 identity paired with its short-lived upload URL.

    Attributes:
        resource_id: Non-secret ``s3://bucket/key`` identity retained in evidence.
        upload_url: Signed URL used once by the uploader and never persisted.
    """

    resource_id: str
    upload_url: SecretStr


class S3PresignClient(Protocol):
    """The one S3 client operation the store needs: signing a bounded transfer URL."""

    def generate_presigned_url(
        self, ClientMethod: str, Params: Mapping[str, str], ExpiresIn: int
    ) -> str:
        """Return one signed URL for ``ClientMethod`` with ``Params``."""
        ...


def parse_s3_location(location: str) -> tuple[str, str]:
    """Split one ``s3://bucket/key`` identity without accepting URL authorization fields.

    Args:
        location: Opaque S3 identity or prefix.

    Returns:
        The bucket name and the key without leading or trailing slashes.

    Raises:
        ValueError: ``location`` is not an ``s3://bucket/key`` value or carries authorization.
    """
    parsed = urlsplit(location)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("S3 locations must be s3://bucket/key values")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("S3 locations must not contain authorization")
    return parsed.netloc, parsed.path.strip("/")


class S3TrainerArtifactStore:
    """Issue short-lived S3 transfers below one caller-owned private prefix."""

    def __init__(self, prefix: str, *, region: str, client: S3PresignClient | None = None) -> None:
        """Bind one ``s3://bucket/prefix`` location and its bucket region.

        Args:
            prefix: Private S3 prefix that receives every trainer artifact for this project.
            region: Bucket region, used to sign regional virtual-hosted URLs.
            client: Optional preconfigured S3 client. Tests inject a stub here.

        Raises:
            ValueError: ``prefix`` is not a valid S3 location.
        """
        self._bucket, self._prefix = parse_s3_location(prefix)
        self._client: S3PresignClient = client or boto3.client(
            "s3",
            region_name=region,
            endpoint_url=f"https://s3.{region}.amazonaws.com",
            config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
        )

    def begin_upload(self, *, kind: Literal["state", "sampling"], name: str) -> S3ArtifactUpload:
        """Authorize one artifact upload without exposing AWS credentials.

        Args:
            kind: Resumable optimizer state or final sampling weights.
            name: One safe path component chosen by the SFT runner.

        Returns:
            The opaque resource identity and a short-lived upload URL.

        Raises:
            ValueError: ``name`` is not one safe path component.
        """
        if "/" in name or name in {"", ".", ".."}:
            raise ValueError("artifact names must be one safe path component")
        key = f"{self._prefix}/{kind}/{name}"
        upload_url = self._client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ContentType": "application/octet-stream",
            },
            ExpiresIn=_URL_LIFETIME_SECONDS,
        )
        return S3ArtifactUpload(
            resource_id=f"s3://{self._bucket}/{key}",
            upload_url=SecretStr(upload_url),
        )

    def download_url(self, resource_id: str) -> SecretStr:
        """Authorize a short-lived download for one identity below this store's prefix.

        Args:
            resource_id: Opaque identity previously returned by ``begin_upload``.

        Returns:
            A short-lived download URL.

        Raises:
            ValueError: ``resource_id`` is malformed or outside this store's prefix.
        """
        bucket, key = parse_s3_location(resource_id)
        if bucket != self._bucket or not key.startswith(f"{self._prefix}/"):
            raise ValueError("resource identity is outside the configured artifact prefix")
        return SecretStr(
            self._client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=_URL_LIFETIME_SECONDS,
            )
        )

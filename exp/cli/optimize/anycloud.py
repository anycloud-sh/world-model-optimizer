"""AnyCloud trainer selection, validation, and backend composition for `exp optimize model`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import httpx
import typer

from exp.common.core.artifacts import Sha256, sha256_json
from exp.common.models import ConnectionConfig, NumericMeasurement, load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.model.sft import (
    AnyCloudTrainerBackend,
    SFTModelOptimizationPreflightError,
    TrainerBackend,
    anycloud_step_cost_bound,
)
from exp.runtime.models.credentials import read_connection_api_key
from exp.runtime.models.providers.anycloud_s3 import S3TrainerArtifactStore, parse_s3_location

_ANYCLOUD_REQUEST_TIMEOUT_SECONDS = 1800.0


class Trainer(StrEnum):
    """Managed trainer that executes the W13 optimizer steps."""

    TINKER = "tinker"
    ANYCLOUD = "anycloud"


@dataclass(frozen=True)
class AnyCloudRun:
    """Per-invocation AnyCloud storage and cost settings for a frozen anycloud connection.

    Attributes:
        artifact_prefix: Private ``s3://bucket/prefix`` that receives checkpoints and adapters.
        artifact_region: AWS region of the artifact bucket.
        price_per_hour_usd: Confirmed hourly price of the GPU running the trainer service.
        maximum_step_seconds: Wall-clock ceiling for one optimizer step, used for cost bounds.
    """

    artifact_prefix: str
    artifact_region: str
    price_per_hour_usd: float
    maximum_step_seconds: float

    def step_cost_bound(self) -> NumericMeasurement:
        """Return the conservative cost of one optimizer step."""
        return anycloud_step_cost_bound(
            price_per_hour_usd=self.price_per_hour_usd,
            maximum_step_seconds=self.maximum_step_seconds,
        )


def connection_provider(store: ProjectStore, connection_name: str) -> str:
    """Return the provider of the connection frozen into the selected config.

    Args:
        store: Project whose local catalog holds the connection.
        connection_name: Connection frozen into the selected optimization config.

    Returns:
        The connection's provider identifier.

    Raises:
        typer.BadParameter: The connection is no longer in the local catalog.
    """
    connection = load_model_catalog(store.model_catalog_path).connections.get(connection_name)
    if connection is None:
        raise typer.BadParameter(
            f"selected model-optimization connection {connection_name!r} is not configured"
        )
    return connection.provider


def resolve_anycloud_run(
    provider: str,
    *,
    trainer: Trainer | None,
    artifact_prefix: str | None,
    artifact_region: str | None,
    price_per_hour_usd: float | None,
    maximum_step_seconds: float,
) -> AnyCloudRun | None:
    """Validate per-invocation AnyCloud flags against the frozen connection's provider.

    Args:
        provider: Provider of the connection frozen into the selected config.
        trainer: Optional explicit trainer flag, which must agree with ``provider``.
        artifact_prefix: S3 prefix for trainer artifacts.
        artifact_region: Region of the artifact bucket.
        price_per_hour_usd: Hourly GPU price.
        maximum_step_seconds: Wall-clock ceiling for one optimizer step.

    Returns:
        Validated AnyCloud settings, or ``None`` for a Tinker connection.

    Raises:
        typer.BadParameter: The flags disagree with the connection or are incomplete.
        ValueError: The artifact prefix is not an S3 location.
    """
    if trainer is not None and trainer != provider:
        raise typer.BadParameter(
            f"--trainer {trainer.value} differs from the selected config's {provider!r} connection"
        )
    values = (
        ("--anycloud-artifact-prefix", artifact_prefix),
        ("--anycloud-artifact-region", artifact_region),
        ("--anycloud-price-per-hour-usd", price_per_hour_usd),
    )
    if provider != Trainer.ANYCLOUD:
        supplied = [name for name, value in values if value is not None]
        if supplied:
            raise typer.BadParameter(", ".join(supplied) + " apply only to an anycloud trainer")
        return None
    missing = [name for name, value in values if value is None]
    if missing:
        raise typer.BadParameter("an AnyCloud trainer run requires " + ", ".join(missing))
    assert artifact_prefix is not None
    assert artifact_region is not None
    assert price_per_hour_usd is not None
    parse_s3_location(artifact_prefix)
    return AnyCloudRun(
        artifact_prefix=artifact_prefix,
        artifact_region=artifact_region,
        price_per_hour_usd=price_per_hour_usd,
        maximum_step_seconds=maximum_step_seconds,
    )


def frozen_connection(
    store: ProjectStore,
    connection_name: str,
    expected_connection_config_sha256: Sha256,
    *,
    provider: str,
) -> ConnectionConfig:
    """Load the selected connection and require its metadata to match the frozen digest.

    Args:
        store: Project whose secret-free catalog holds the connection.
        connection_name: Exact connection frozen into the optimization config.
        expected_connection_config_sha256: Frozen connection metadata digest.
        provider: Provider the composed backend requires.

    Returns:
        The unchanged connection metadata.

    Raises:
        SFTModelOptimizationPreflightError: The connection is missing, uses another provider, or
            drifted after the config was created.
    """
    catalog = load_model_catalog(store.model_catalog_path)
    connection = catalog.connections.get(connection_name)
    if connection is None or connection.provider != provider:
        raise SFTModelOptimizationPreflightError(
            f"selected connection {connection_name!r} is not a configured {provider} connection"
        )
    current_connection_config_sha256 = sha256_json(
        {
            "provider": connection.provider,
            "base_url": connection.base_url,
            "api_key_env": connection.api_key_env,
        }
    )
    if current_connection_config_sha256 != expected_connection_config_sha256:
        raise SFTModelOptimizationPreflightError(
            f"selected {provider} connection metadata drifted before credential resolution"
        )
    return connection


def compose_anycloud_backend(
    store: ProjectStore,
    connection_name: str,
    expected_connection_config_sha256: Sha256,
    run: AnyCloudRun,
) -> TrainerBackend:
    """Resolve the trainer token and compose the AnyCloud HTTP adapter with S3 artifacts.

    Args:
        store: Project whose secret-free catalog selects the trainer connection.
        connection_name: Exact AnyCloud connection frozen into the optimization config.
        expected_connection_config_sha256: Frozen connection metadata digest that must match
            before credential access.
        run: Validated artifact storage and cost settings for this invocation.

    Returns:
        Concrete backend that does not call the trainer until W13 invokes ``open``.

    Raises:
        ModelCredentialError: The selected token environment variable is absent.
        SFTModelOptimizationPreflightError: The selected connection is invalid or drifted.
    """
    connection = frozen_connection(
        store, connection_name, expected_connection_config_sha256, provider="anycloud"
    )
    if connection.base_url is None:
        raise SFTModelOptimizationPreflightError(
            f"AnyCloud connection {connection_name!r} has no trainer URL"
        )
    token = read_connection_api_key(connection, connection_id=connection_name)
    client = httpx.Client(
        base_url=connection.base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=httpx.Timeout(_ANYCLOUD_REQUEST_TIMEOUT_SECONDS, connect=60),
    )
    return AnyCloudTrainerBackend(
        client,
        S3TrainerArtifactStore(run.artifact_prefix, region=run.artifact_region),
        price_per_hour_usd=run.price_per_hour_usd,
        maximum_step_seconds=run.maximum_step_seconds,
    )

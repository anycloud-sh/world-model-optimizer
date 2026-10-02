"""AnyCloud trainer flag validation and backend composition tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from exp.cli.optimize.anycloud import (
    AnyCloudRun,
    Trainer,
    compose_anycloud_backend,
    resolve_anycloud_run,
)
from exp.common.core.artifacts import sha256_json
from exp.common.models import ConnectionConfig, ModelCatalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.model.sft import AnyCloudTrainerBackend, SFTModelOptimizationPreflightError
from exp.optimize.model.sft.training_test import _persisted_dataset
from exp.runtime.models.credentials import ModelCredentialError

_CONNECTION = ConnectionConfig(
    provider="anycloud",
    base_url="https://exp-trainer.anycloud.sh",
    api_key_env="TRAINER_TOKEN",
)
_CONNECTION_SHA256 = sha256_json(
    {
        "provider": _CONNECTION.provider,
        "base_url": _CONNECTION.base_url,
        "api_key_env": _CONNECTION.api_key_env,
    }
)


def _store(tmp_path: Path) -> ProjectStore:
    store = _persisted_dataset(tmp_path).store
    write_model_catalog(
        store.model_catalog_path,
        ModelCatalog(connections={"trainer": _CONNECTION}, models={}),
    )
    return store


def _run() -> AnyCloudRun:
    run = resolve_anycloud_run(
        "anycloud",
        trainer=None,
        artifact_prefix="s3://trainer-bucket/p1",
        artifact_region="us-west-2",
        price_per_hour_usd=1.29,
        maximum_step_seconds=120,
    )
    assert run is not None
    return run


def test_run_settings_price_one_step_by_gpu_hour() -> None:
    run = resolve_anycloud_run(
        "anycloud",
        trainer=Trainer.ANYCLOUD,
        artifact_prefix="s3://trainer-bucket/p1",
        artifact_region="us-west-2",
        price_per_hour_usd=1.29,
        maximum_step_seconds=120,
    )

    assert run is not None
    assert run.step_cost_bound().value == pytest.approx(1.29 * 120 / 3600)
    assert run.step_cost_bound().provenance == "estimated"


def test_tinker_connection_needs_no_anycloud_settings() -> None:
    assert (
        resolve_anycloud_run(
            "tinker",
            trainer=None,
            artifact_prefix=None,
            artifact_region=None,
            price_per_hour_usd=None,
            maximum_step_seconds=600,
        )
        is None
    )


def test_artifact_prefix_must_be_an_s3_location() -> None:
    with pytest.raises(ValueError, match="S3 locations"):
        resolve_anycloud_run(
            "anycloud",
            trainer=None,
            artifact_prefix="https://trainer-bucket/p1",
            artifact_region="us-west-2",
            price_per_hour_usd=1.29,
            maximum_step_seconds=600,
        )


def test_compose_reads_token_and_builds_the_http_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("TRAINER_TOKEN", "trainer-secret")

    backend = compose_anycloud_backend(store, "trainer", _CONNECTION_SHA256, _run())

    assert isinstance(backend, AnyCloudTrainerBackend)


def test_compose_requires_the_token_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.delenv("TRAINER_TOKEN", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "no-stored-credentials"))

    with pytest.raises(ModelCredentialError, match="TRAINER_TOKEN"):
        compose_anycloud_backend(store, "trainer", _CONNECTION_SHA256, _run())


def test_compose_rejects_connection_drift_before_reading_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    monkeypatch.setenv("TRAINER_TOKEN", "trainer-secret")
    drifted = sha256_json({"provider": "anycloud", "base_url": "https://other", "api_key_env": "X"})

    with pytest.raises(SFTModelOptimizationPreflightError, match="drifted"):
        compose_anycloud_backend(store, "trainer", drifted, _run())
    with pytest.raises(SFTModelOptimizationPreflightError, match="not a configured anycloud"):
        compose_anycloud_backend(store, "missing", _CONNECTION_SHA256, _run())


def test_explicit_trainer_must_match_the_frozen_connection() -> None:
    with pytest.raises(typer.BadParameter, match="differs from the selected config"):
        resolve_anycloud_run(
            "tinker",
            trainer=Trainer.ANYCLOUD,
            artifact_prefix=None,
            artifact_region=None,
            price_per_hour_usd=None,
            maximum_step_seconds=600,
        )

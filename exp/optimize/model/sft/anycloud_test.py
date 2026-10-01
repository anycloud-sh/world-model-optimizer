"""Local AnyCloud adapter tests with caller-owned HTTP and artifact fakes."""

from __future__ import annotations

import json
from typing import Literal, cast

import httpx
import pytest
from pydantic import SecretStr

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import AssistantAction
from exp.optimize.model.sft.anycloud import (
    AnyCloudArtifactUpload,
    AnyCloudSFTDatum,
    AnyCloudTrainerBackend,
)
from exp.optimize.model.sft.contracts import (
    SFTExample,
    SFTMessage,
    TraceExampleSource,
)
from exp.optimize.model.sft.training import TinkerSFTError, TinkerSFTSpec

_DIGEST = "a" * 64


def _spec() -> TinkerSFTSpec:
    """Build one compact frozen specification for local adapter tests."""
    return TinkerSFTSpec(
        base_model="Qwen/Qwen3.5-4B",
        lora_rank=8,
        learning_rate=0.0002,
        batch_size=1,
        epochs=1,
        checkpoint_every_steps=1,
        maximum_datum_tokens=128,
    )


def _example() -> SFTExample:
    """Build one complete target that exercises remote conversation rendering."""
    return SFTExample(
        example_id="example-anycloud",
        leakage_group_id="lineage-anycloud",
        task="Respond with the requested color.",
        history=(SFTMessage(role="user", content="What color is the sky?"),),
        target=AssistantAction(content="Blue."),
        source=TraceExampleSource(
            trace_id="trace-anycloud",
            acceptance_evidence=ArtifactInput(
                artifact_id="acceptance-evidence",
                sha256=_DIGEST,
            ),
        ),
        source_step_index=0,
    )


class _ArtifactStore:
    """Issue deterministic signed URLs while journaling caller-owned authorization."""

    def __init__(self) -> None:
        """Initialize empty upload and download journals."""
        self.uploads: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    def begin_upload(
        self, *, kind: Literal["state", "sampling"], name: str
    ) -> AnyCloudArtifactUpload:
        """Return one opaque S3 resource identity and test-only signed URL."""
        self.uploads.append((kind, name))
        return AnyCloudArtifactUpload(
            resource_id=f"s3://exp-trainer-test/{kind}/{name}",
            upload_url=SecretStr(f"https://upload.invalid/{kind}/{name}?signature=upload-secret"),
        )

    def download_url(self, resource_id: str) -> SecretStr:
        """Return one test-only signed download URL for the requested resource."""
        self.downloads.append(resource_id)
        return SecretStr("https://download.invalid/state.pt?signature=download-secret")


class _TrainerService:
    """Provide deterministic HTTP responses and retain parsed request payloads."""

    def __init__(self) -> None:
        """Initialize an empty request journal."""
        self.requests: list[tuple[str, dict[str, object]]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Respond to the complete AnyCloud trainer surface used by WMO."""
        payload = json.loads(request.content)
        self.requests.append((request.url.path, payload))
        if request.url.path == "/v1/sessions":
            return httpx.Response(200, json={"session_id": "session-1"})
        if request.url.path.endswith("/datums:render"):
            datums = [
                {
                    "datum_id": f"datum-{index}",
                    "example_id": example["example_id"],
                    "supervised_token_count": 2,
                }
                for index, example in enumerate(payload["examples"], start=1)
            ]
            return httpx.Response(200, json={"datums": datums})
        if request.url.path.endswith("/batches:train"):
            return httpx.Response(
                200,
                json={
                    "loss": 1.25,
                    "gradient_norm": 0.75,
                    "input_token_count": 14,
                    "supervised_token_count": 2,
                },
            )
        if request.url.path.endswith("/artifacts:save"):
            return httpx.Response(200, json={"sha256": _DIGEST, "size_bytes": 128})
        return httpx.Response(404)


def _backend(
    service: _TrainerService,
    store: _ArtifactStore,
) -> tuple[AnyCloudTrainerBackend, httpx.Client]:
    """Bind the concrete adapter to local caller-owned HTTP and artifact fakes."""
    client = httpx.Client(
        base_url="https://trainer.invalid",
        headers={"Authorization": "Bearer caller-secret"},
        transport=httpx.MockTransport(service.handle),
    )
    backend = AnyCloudTrainerBackend(
        client,
        store,
        price_per_hour_usd=1.09,
        maximum_step_seconds=120,
        model_revision="b" * 40,
    )
    return backend, client


def test_adapter_completes_render_train_and_direct_artifact_upload_contract() -> None:
    """The adapter preserves Experiential order and returns only opaque artifact identities."""
    service = _TrainerService()
    store = _ArtifactStore()
    backend, client = _backend(service, store)
    with client:
        session = backend.open(_spec(), None)
        (datum,) = session.render_examples((_example(),))
        result = session.train_batch((datum,), learning_rate=0.0002)
        state_path = session.save_state("step-000001.pt")
        sampling_path = session.save_sampling_handle("final-adapter.tar")

    assert isinstance(datum, AnyCloudSFTDatum)
    assert datum.example_id == "example-anycloud"
    assert datum.supervised_token_count == 2
    assert result.loss == 1.25
    assert result.gradient_norm == 0.75
    assert state_path == "s3://exp-trainer-test/state/step-000001.pt"
    assert sampling_path == "s3://exp-trainer-test/sampling/final-adapter.tar"
    assert store.uploads == [
        ("state", "step-000001.pt"),
        ("sampling", "final-adapter.tar"),
    ]
    assert service.requests[0][1]["model_revision"] == "b" * 40
    render_payload = service.requests[1][1]
    examples = cast("list[dict[str, object]]", render_payload["examples"])
    messages = cast("list[dict[str, object]]", examples[0]["messages"])
    assert messages[-1] == {
        "role": "assistant",
        "content": "Blue.",
    }
    upload_url = cast(str, service.requests[3][1]["upload_url"])
    assert upload_url.endswith("signature=upload-secret")
    assert "upload-secret" not in state_path


def test_adapter_restores_from_caller_authorized_download_without_persisting_url() -> None:
    """Resume sends the short-lived URL only to the service and retains the opaque ID."""
    service = _TrainerService()
    store = _ArtifactStore()
    backend, client = _backend(service, store)
    resource_id = "s3://exp-trainer-test/state/step-000001.pt"
    with client:
        backend.open(_spec(), resource_id)

    assert store.downloads == [resource_id]
    resume_url = cast(str, service.requests[0][1]["resume_download_url"])
    assert resume_url.endswith("download-secret")


def test_adapter_cost_bound_uses_confirmed_vm_price_and_step_ceiling() -> None:
    """The pre-dispatch budget check conservatively prices the whole VM time bound."""
    service = _TrainerService()
    store = _ArtifactStore()
    backend, client = _backend(service, store)
    with client:
        cost = backend.conservative_step_cost(_spec(), batch_example_count=1)

    assert cost is not None
    assert cost.value == pytest.approx(1.09 * 120 / 3600)
    assert cost.provenance == "estimated"
    with pytest.raises(ValueError, match="batch_example_count must be positive"):
        backend.conservative_step_cost(_spec(), batch_example_count=0)

    with pytest.raises(ValueError, match="model_revision must be an exact"):
        AnyCloudTrainerBackend(
            client,
            store,
            price_per_hour_usd=1.09,
            maximum_step_seconds=120,
            model_revision="main",
        )


def test_adapter_rejects_foreign_session_datums_before_remote_dispatch() -> None:
    """A datum identity cannot cross trainer sessions even when its wire shape matches."""
    service = _TrainerService()
    store = _ArtifactStore()
    backend, client = _backend(service, store)
    with client:
        session = backend.open(_spec(), None)
        foreign = AnyCloudSFTDatum(
            example_id="example-anycloud",
            supervised_token_count=2,
            datum_id="datum-foreign",
            session_id="session-foreign",
        )
        with pytest.raises(TinkerSFTError, match="datum from another session"):
            session.train_batch((foreign,), learning_rate=0.0002)

    assert [path for path, _payload in service.requests] == ["/v1/sessions"]


def test_adapter_errors_do_not_expose_remote_body_or_signed_url() -> None:
    """Safe status-only errors omit service bodies and short-lived authorization material."""
    secret = "signed-download-secret"

    def fail(request: httpx.Request) -> httpx.Response:
        """Return a failure body containing material that must never enter Experiential errors."""
        return httpx.Response(500, text=f"internal failure: {secret}")

    store = _ArtifactStore()
    client = httpx.Client(
        base_url="https://trainer.invalid",
        transport=httpx.MockTransport(fail),
    )
    backend = AnyCloudTrainerBackend(
        client,
        store,
        price_per_hour_usd=1.09,
        maximum_step_seconds=120,
    )
    with client, pytest.raises(TinkerSFTError) as raised:
        backend.open(_spec(), None)

    assert str(raised.value) == "AnyCloud trainer could not open session: HTTP 500"
    assert secret not in str(raised.value)

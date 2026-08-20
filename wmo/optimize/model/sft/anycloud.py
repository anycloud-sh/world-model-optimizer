"""AnyCloud HTTP adapter for the existing managed SFT backend seam.

The adapter accepts a caller-owned authenticated HTTP client and artifact store. It does not read
credentials, launch compute, or create storage. Application composition retains those authority
boundaries, matching the concrete Tinker adapter's caller-owned client contract.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar
from urllib.parse import quote

import httpx
from pydantic import Field, SecretStr, ValidationError, field_validator

from wmo.common.core.artifacts import ContractModel, JsonObject, Sha256
from wmo.common.models import NumericMeasurement
from wmo.optimize.model.sft.contracts import SFTExample
from wmo.optimize.model.sft.provider_resources import validate_provider_resource_id
from wmo.optimize.model.sft.tinker import tinker_messages_from_example
from wmo.optimize.model.sft.training_contracts import (
    TinkerSFTError,
    TinkerSFTSpec,
    TrainerBatchResult,
    TrainerDatum,
    TrainerSession,
)


class AnyCloudArtifactUpload(ContractModel):
    """One opaque resource identity paired with a short-lived upload URL."""

    resource_id: str = Field(min_length=1, max_length=2048)
    upload_url: SecretStr


class AnyCloudTrainerArtifactStore(Protocol):
    """Caller-owned authorization for immutable trainer artifact transfers."""

    def begin_upload(
        self, *, kind: Literal["state", "sampling"], name: str
    ) -> AnyCloudArtifactUpload:
        """Return an opaque resource identity and short-lived upload URL."""
        ...

    def download_url(self, resource_id: str) -> SecretStr:
        """Authorize a short-lived download for one previously returned resource identity."""
        ...


@dataclass(frozen=True)
class AnyCloudSFTDatum:
    """One WMO example rendered and retained by a live AnyCloud trainer session."""

    example_id: str
    supervised_token_count: int
    datum_id: str
    session_id: str


class _OpenSessionResponse(ContractModel):
    """Opaque identity returned for one live trainer session."""

    session_id: str = Field(min_length=1, max_length=128)


class _RenderedDatum(ContractModel):
    """Remote datum metadata returned in exact example order."""

    datum_id: str = Field(min_length=1, max_length=128)
    example_id: str = Field(min_length=1, max_length=256)
    supervised_token_count: int = Field(gt=0)


class _RenderExamplesResponse(ContractModel):
    """Ordered metadata for a completed remote render call."""

    datums: tuple[_RenderedDatum, ...] = Field(min_length=1)


class _TrainBatchResponse(ContractModel):
    """Finite observations returned for one completed optimizer update."""

    loss: float
    gradient_norm: float
    input_token_count: int = Field(gt=0)
    supervised_token_count: int = Field(gt=0)

    @field_validator("loss", "gradient_norm")
    @classmethod
    def _require_finite_metric(cls, value: float) -> float:
        """Reject nonfinite accelerator observations before WMO records them."""
        if not math.isfinite(value):
            raise ValueError("AnyCloud trainer metrics must be finite")
        return value


class _SaveArtifactResponse(ContractModel):
    """Digest and byte count for one completed direct artifact upload."""

    sha256: Sha256
    size_bytes: int = Field(gt=0)


_ResponseT = TypeVar("_ResponseT", bound=ContractModel)


class AnyCloudTrainerBackend:
    """Open stateful remote LoRA sessions through an injected AnyCloud service client."""

    def __init__(
        self,
        client: httpx.Client,
        artifact_store: AnyCloudTrainerArtifactStore,
        *,
        price_per_hour_usd: float,
        maximum_step_seconds: float,
        model_revision: str | None = None,
    ) -> None:
        """Bind caller-owned service, storage, and conservative cost authority.

        Args:
            client: Authenticated client with the trainer service as its base URL.
            artifact_store: Caller-owned short-lived transfer authorization.
            price_per_hour_usd: Confirmed hourly price for the selected AnyCloud compute.
            maximum_step_seconds: Caller-provided conservative duration bound for one step.
            model_revision: Optional exact 40-hex Hugging Face model revision.

        Raises:
            ValueError: Either cost input is negative, nonfinite, or the step bound is zero.
        """
        if not math.isfinite(price_per_hour_usd) or price_per_hour_usd < 0:
            raise ValueError("price_per_hour_usd must be finite and nonnegative")
        if not math.isfinite(maximum_step_seconds) or maximum_step_seconds <= 0:
            raise ValueError("maximum_step_seconds must be finite and positive")
        if model_revision is not None and re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
            raise ValueError("model_revision must be an exact lowercase 40-hex revision")
        self._client = client
        self._artifact_store = artifact_store
        self._price_per_hour_usd = price_per_hour_usd
        self._maximum_step_seconds = maximum_step_seconds
        self._model_revision = model_revision

    def conservative_step_cost(
        self, spec: TinkerSFTSpec, *, batch_example_count: int
    ) -> NumericMeasurement | None:
        """Bound one remote step from the selected VM price and wall-clock ceiling.

        Args:
            spec: Frozen WMO training settings. The AnyCloud bound is time based.
            batch_example_count: Exact positive scheduled example count.

        Returns:
            The finite upper cost bound for one allocated remote step.

        Raises:
            ValueError: ``batch_example_count`` is not positive.
        """
        del spec
        if batch_example_count <= 0:
            raise ValueError("batch_example_count must be positive")
        return NumericMeasurement(
            value=self._price_per_hour_usd * self._maximum_step_seconds / 3600,
            provenance="estimated",
        )

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> TrainerSession:
        """Create one session, restoring remote optimizer state when requested.

        Args:
            spec: Frozen base model, LoRA rank, seed, and token ceiling.
            resume_state_path: Opaque artifact resource identity from a prior checkpoint.

        Returns:
            A session bound to the returned remote identity.
        """
        payload: JsonObject = {
            "base_model": spec.base_model,
            "lora_rank": spec.lora_rank,
            "seed": spec.seed,
            "maximum_datum_tokens": spec.maximum_datum_tokens,
        }
        if self._model_revision is not None:
            payload["model_revision"] = self._model_revision
        if resume_state_path is not None:
            resource_id = validate_provider_resource_id(
                resume_state_path,
                label="AnyCloud resume state",
            )
            download_url = self._artifact_store.download_url(resource_id)
            payload["resume_download_url"] = download_url.get_secret_value()
        response = self._post(
            "/v1/sessions",
            payload,
            _OpenSessionResponse,
            operation="open session",
        )
        return AnyCloudTrainerSession(
            client=self._client,
            artifact_store=self._artifact_store,
            session_id=response.session_id,
        )

    def _post(
        self,
        path: str,
        payload: JsonObject,
        response_type: type[_ResponseT],
        *,
        operation: str,
    ) -> _ResponseT:
        """Post one validated request and parse a bounded response without exposing bodies."""
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            return response_type.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer could not {operation}: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer could not {operation}: remote outcome is unknown"
            ) from exc
        except (ValidationError, ValueError) as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer returned an invalid response while attempting to {operation}"
            ) from exc


class AnyCloudTrainerSession:
    """One remote session that satisfies WMO's existing managed trainer contract."""

    def __init__(
        self,
        *,
        client: httpx.Client,
        artifact_store: AnyCloudTrainerArtifactStore,
        session_id: str,
    ) -> None:
        """Bind a caller-owned client, artifact store, and exact remote session identity."""
        self._client = client
        self._artifact_store = artifact_store
        self._session_id = session_id

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[TrainerDatum, ...]:
        """Render complete WMO conversations on the selected remote base model.

        Args:
            examples: Ordered frozen W12 training examples.

        Returns:
            One remote datum identity per example in exact input order.

        Raises:
            TinkerSFTError: The service fails, returns malformed data, or changes example order.
        """
        if not examples:
            return ()
        payload_examples: list[JsonObject] = []
        for example in examples:
            messages = tinker_messages_from_example(example)
            payload_examples.append(
                {
                    "example_id": example.example_id,
                    "messages": messages,
                }
            )
        response = self._post(
            "datums:render",
            {"examples": payload_examples},
            _RenderExamplesResponse,
            operation="render examples",
        )
        expected_ids = tuple(example.example_id for example in examples)
        observed_ids = tuple(datum.example_id for datum in response.datums)
        if observed_ids != expected_ids:
            raise TinkerSFTError("AnyCloud trainer returned rendered examples in a different order")
        return tuple(
            AnyCloudSFTDatum(
                example_id=datum.example_id,
                supervised_token_count=datum.supervised_token_count,
                datum_id=datum.datum_id,
                session_id=self._session_id,
            )
            for datum in response.datums
        )

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        """Dispatch exactly one remote cross-entropy optimizer update.

        Args:
            datums: Remote datum identities returned by this exact session.
            learning_rate: Frozen Adam learning rate for the scheduled step.

        Returns:
            Backend-reported finite loss and gradient norm.

        Raises:
            TinkerSFTError: A datum belongs to another backend or session, or dispatch fails.
        """
        datum_ids: list[str] = []
        if not datums:
            raise TinkerSFTError("AnyCloud trainer batches must contain at least one datum")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise TinkerSFTError("AnyCloud trainer learning_rate must be finite and positive")
        for datum in datums:
            if not isinstance(datum, AnyCloudSFTDatum):
                raise TinkerSFTError("AnyCloud trainer received a datum from another backend")
            if datum.session_id != self._session_id:
                raise TinkerSFTError("AnyCloud trainer received a datum from another session")
            datum_ids.append(datum.datum_id)
        response = self._post(
            "batches:train",
            {"datum_ids": datum_ids, "learning_rate": learning_rate},
            _TrainBatchResponse,
            operation="train batch",
        )
        return TrainerBatchResult(
            loss=response.loss,
            gradient_norm=response.gradient_norm,
        )

    def save_state(self, checkpoint_name: str) -> str:
        """Upload one resumable optimizer state through caller-authorized storage.

        Args:
            checkpoint_name: Unique immutable name for the scheduled run step.

        Returns:
            A non-secret opaque artifact resource identity.
        """
        return self._save_artifact(kind="state", name=checkpoint_name)

    def save_sampling_handle(self, model_name: str) -> str:
        """Upload final portable PEFT weights through caller-authorized storage.

        Args:
            model_name: Unique immutable name for the completed model.

        Returns:
            A non-secret opaque artifact resource identity.
        """
        return self._save_artifact(kind="sampling", name=model_name)

    def _save_artifact(self, *, kind: Literal["state", "sampling"], name: str) -> str:
        """Authorize and complete one direct immutable trainer artifact upload."""
        target = self._artifact_store.begin_upload(kind=kind, name=name)
        resource_id = validate_provider_resource_id(
            target.resource_id,
            label=f"AnyCloud {kind} artifact",
        )
        self._post(
            "artifacts:save",
            {
                "kind": kind,
                "name": name,
                "upload_url": target.upload_url.get_secret_value(),
            },
            _SaveArtifactResponse,
            operation=f"save {kind} artifact",
        )
        return resource_id

    def _post(
        self,
        suffix: str,
        payload: JsonObject,
        response_type: type[_ResponseT],
        *,
        operation: str,
    ) -> _ResponseT:
        """Post beneath the exact escaped session path and parse a validated response."""
        session_id = quote(self._session_id, safe="")
        path = f"/v1/sessions/{session_id}/{suffix}"
        try:
            response = self._client.post(path, json=payload)
            response.raise_for_status()
            return response_type.model_validate(response.json())
        except httpx.HTTPStatusError as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer could not {operation}: HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer could not {operation}: remote outcome is unknown"
            ) from exc
        except (ValidationError, ValueError) as exc:
            raise TinkerSFTError(
                f"AnyCloud trainer returned an invalid response while attempting to {operation}"
            ) from exc

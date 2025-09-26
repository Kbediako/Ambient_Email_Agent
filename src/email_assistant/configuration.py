from __future__ import annotations

import os
from dataclasses import dataclass

from langchain.chat_models import init_chat_model

_DEFAULT_MODEL = "gemini-2.5-pro"
_DEFAULT_PROVIDER = "google_genai"


@dataclass(frozen=True)
class ModelSpec:
    """Normalised representation of the chat model + provider."""

    provider: str
    model: str

    @property
    def identifier(self) -> str:
        """
        Produce the model identifier prefixed by the provider when present.
        
        If the ModelSpec has a non-empty provider, the identifier is "provider:model"; otherwise it is just "model".
        
        Returns:
            identifier (str): The provider-prefixed identifier or the model name when no provider is set.
        """

        return f"{self.provider}:{self.model}" if self.provider else self.model


def _default_model() -> str:
    """
    Selects the default model name for the assistant.
    
    Checks the environment variables `EMAIL_ASSISTANT_MODEL` then `GEMINI_MODEL` and returns the first non-empty value; if neither is set returns the module default `_DEFAULT_MODEL`.
    
    Returns:
        str: The resolved model identifier.
    """
    return (
        os.environ.get("EMAIL_ASSISTANT_MODEL")
        or os.environ.get("GEMINI_MODEL")
        or _DEFAULT_MODEL
    )


def _default_provider() -> str:
    """
    Selects the default model provider for the email assistant.
    
    Reads the `EMAIL_ASSISTANT_MODEL_PROVIDER` environment variable and returns its value if set; otherwise returns the module default `_DEFAULT_PROVIDER`.
    
    Returns:
        provider (str): The resolved model provider name.
    """
    return os.environ.get("EMAIL_ASSISTANT_MODEL_PROVIDER", _DEFAULT_PROVIDER)


def normalize_model_spec(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    default_model: str | None = None,
    default_provider: str | None = None,
) -> ModelSpec:
    """
    Normalize and resolve a provider and model name for Gemini chat models.
    
    Constructs a ModelSpec by resolving an explicit `model` and optional `model_provider`
    against provided defaults and environment defaults. Accepts `provider:model`
    prefixes and Vertex-style `models/<id>` paths; when a provider is omitted the
    default provider is used.
    
    Parameters:
        model (str | None): Optional model identifier; may include a provider prefix
            like `provider:model` or a path like `models/<id>`.
        model_provider (str | None): Optional explicit provider override.
        default_model (str | None): Optional fallback model to use when `model` is
            empty or resolves to an empty fragment.
        default_provider (str | None): Optional fallback provider to use when no
            provider is specified.
    
    Returns:
        ModelSpec: A ModelSpec with `provider` set to the resolved provider and
            `model` set to the resolved model identifier.
    """

    effective_model = (model or "").strip() or default_model or _default_model()
    provider = (
        model_provider
        or (default_provider or _default_provider())
        or ""
    )

    if ":" in effective_model:
        candidate_provider, candidate_model = effective_model.split(":", 1)
        candidate_provider = candidate_provider.strip()
        candidate_model = candidate_model.strip()
        if candidate_provider:
            provider = candidate_provider
        effective_model = candidate_model or default_model or _default_model()

    if effective_model.startswith("models/"):
        remainder = effective_model.split("/", 1)[1]
        if remainder:
            effective_model = remainder
        else:
            effective_model = default_model or _default_model()

    if not effective_model:
        effective_model = default_model or _default_model()

    return ModelSpec(provider=provider, model=effective_model)


def format_model_identifier(model: str | None = None, *, provider: str | None = None) -> str:
    """
    Produce a provider-prefixed model identifier suitable for logging.
    
    Returns:
        identifier (str): The identifier in the form "provider:model" when a provider is present, or "model" when no provider is set.
    """

    spec = normalize_model_spec(model, model_provider=provider)
    return spec.identifier


def model_spec(model: str | None = None, *, provider: str | None = None) -> ModelSpec:
    """
    Return a normalized ModelSpec for the given model and optional provider.
    
    Parameters:
        model (str | None): Optional model identifier. May include a provider prefix (e.g., "provider:model") or a "models/..." path; if None, environment or built-in defaults are used.
        provider (str | None): Optional explicit provider override.
    
    Returns:
        ModelSpec: A ModelSpec with the resolved `provider` and `model` values.
    """

    return normalize_model_spec(model, model_provider=provider)


def get_llm(temperature: float = 0.0, **kwargs):
    """
    Create a LangChain BaseChatModel configured for a Gemini chat model.
    
    This function resolves the effective provider and model (accepting overrides via the
    `model` and `model_provider` kwargs), ensures `convert_system_message_to_human` is set
    to False by default, and returns an initialized chat model instance.
    
    Parameters:
        temperature (float): Sampling temperature for the model.
        **kwargs: Additional keyword arguments forwarded to LangChain's `init_chat_model`.
            Recognized overrides include:
              - model: explicit model string (may include provider prefix or "models/..." paths)
              - model_provider: explicit provider name
    
    Returns:
        BaseChatModel: A configured LangChain chat model instance.
    """

    # Allow explicit model override via kwargs, fallback to env var (2.5 series by default)
    raw_model = kwargs.pop("model", None)
    provider_override = kwargs.pop("model_provider", None)
    spec = normalize_model_spec(raw_model, model_provider=provider_override)

    # Gemini now natively supports system messages; disable the legacy conversion layer to
    # avoid the noisy "Convert_system_message_to_human will be deprecated" warning.
    kwargs.setdefault("convert_system_message_to_human", False)

    return init_chat_model(
        spec.model,
        model_provider=spec.provider,
        temperature=temperature,
        **kwargs,
    )

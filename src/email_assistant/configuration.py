import os
from typing import Literal, Optional
from langchain_google_genai import ChatGoogleGenerativeAI


Role = Optional[Literal["router", "tool", "memory"]]


def resolve_model(role: Role = None) -> str:
    """Resolve model name from environment with role-specific overrides.

    Precedence (first non-empty wins):
    - EMAIL_ASSISTANT_<ROLE>_MODEL where ROLE in {ROUTER, TOOL, MEMORY}
    - EMAIL_ASSISTANT_MODEL (global default for all roles)
    - GEMINI_MODEL (fallback)
    - GEMINI_MODEL_AGENT (legacy fallback used in some environments)
    - "gemini-2.5-pro" (final default)
    """

    if role == "router":
        specific = os.getenv("EMAIL_ASSISTANT_ROUTER_MODEL")
        if specific:
            return specific
    elif role == "tool":
        specific = os.getenv("EMAIL_ASSISTANT_TOOL_MODEL")
        if specific:
            return specific
    elif role == "memory":
        specific = os.getenv("EMAIL_ASSISTANT_MEMORY_MODEL")
        if specific:
            return specific

    return (
        os.getenv("EMAIL_ASSISTANT_MODEL")
        or os.getenv("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL_AGENT")
        or "gemini-2.5-pro"
    )


def _normalise_model_name(model_name: str) -> str:
    """Normalise provider-prefixed or path-prefixed model identifiers."""
    if ":" in model_name:
        _, model_name = model_name.split(":", 1)
    if model_name.startswith("models/"):
        model_name = model_name.split("/", 1)[1]
    return model_name


def get_llm(temperature: float = 0.0, *, role: Role = None, **kwargs):
    """Return a configured Gemini LLM.

    Args:
        temperature: The temperature to use for the LLM.
        role: Optional role hint ("router", "tool", or "memory") to resolve
              the model from environment in a single place.
        **kwargs: Additional arguments to pass to the LLM (e.g., model).

    Returns:
        A configured ChatGoogleGenerativeAI instance.
    """
    # Prefer explicit model override, otherwise resolve from env using role
    model_name = kwargs.pop("model", resolve_model(role))
    model_name = _normalise_model_name(model_name)

    # Gemini now natively supports system messages; disable the legacy conversion layer
    # to avoid the noisy "Convert_system_message_to_human will be deprecated" warning.
    kwargs.setdefault("convert_system_message_to_human", False)
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, **kwargs)


__all__ = ["get_llm", "resolve_model"]

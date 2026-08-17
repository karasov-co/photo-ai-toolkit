"""One place that finds configuration, so no entry point has to remember to.

The bug this exists to prevent: `main.py` called `load_dotenv()` and `cli.py`
did not. A user who put `OPENAI_API_KEY` in `.env` and ran the new CLI got a
credentials error from deep inside the semantic pass -- after every photograph
had already been decoded -- and the run then carried on to a summary that looked
like a success.

Three rules follow from that:

- **Loading happens once, at the entry point, before anything reads an
  environment variable.** Not lazily inside the module that needs it, because
  that is exactly how one caller ends up configured and another does not.

- **The project `.env` is found relative to the source, not the shell.** A user
  running `python /path/to/repo/cli.py` from their Pictures folder is still
  running this project, and its `.env` is still the one that applies. A `.env`
  in the working directory is also honoured, and the resolution order is
  reported rather than guessed at.

- **A real environment variable always wins.** `override=False` throughout: a
  key exported for one command must not be silently replaced by a stale one in a
  file.

Nothing here ever logs, prints, or returns the key -- not its value, not its
prefix, not its length. `credential_status()` returns where it came from and
nothing else, because "sk-proj-abc…" in a bug report is still a leaked key.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

# src/photoai/bootstrap.py -> src/photoai -> src -> the checkout, which is where
# `.env` sits. Two levels further up than it used to be, and the same directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV = PROJECT_ROOT / ".env"

API_KEY_VAR = "OPENAI_API_KEY"
# xAI's own variable, checked first. OPENAI_API_KEY still works, because most
# people arrive with one already in their .env and refusing it would be
# pedantry -- a key that does not work is caught by the preflight in one
# request, which is a better place to find out than a config error.
XAI_KEY_VAR = "XAI_API_KEY"
MODEL_VAR = "OPENAI_MODEL"
PROVIDER_VAR = "PHOTO_AI_PROVIDER"

# The model this toolkit runs on. One name, defined once, used by Stage 2 and
# Stage 3 alike, and verified against the account before any photograph is
# opened.
#
# It is overridable, because a model available to one account is not available
# to another. It is never *substituted*: if the configured model is unavailable
# the run stops and says so. Quietly dropping to an older family would change
# every judgement in the report while the report went on claiming to be the
# analysis that was asked for -- the artistic read in particular is the whole
# product, and it is not portable across model generations.
DEFAULT_MODEL = "grok-4.6"
DEFAULT_PROVIDER = "grok"

# The old name, kept because it is what several call sites and tests say. One
# value, two spellings, and no chance of them drifting apart.
DEFAULT_SEMANTIC_MODEL = DEFAULT_MODEL

# Families this toolkit will not fall back to, and will not name as a
# workaround. Listed so a test can assert their absence from the fallback paths
# rather than trusting that nobody adds one back.
LEGACY_MODEL_PREFIXES = (
    "gpt-4",
    "gpt-3",
    "gpt-5-",
    "chatgpt-4o",
    "o1",
    "o3",
)


def is_legacy_model(name: str) -> bool:
    """Whether a name belongs to a superseded family.

    Used to refuse a *silent* substitution, never to block an operator who has
    typed one deliberately: `--model` is an explicit instruction from somebody
    who can see what they asked for. What it must never be is a default, a
    fallback, or a suggestion in an error message.
    """
    lowered = str(name or "").strip().lower()
    return any(lowered.startswith(prefix) for prefix in LEGACY_MODEL_PREFIXES)


# Cost estimation moved to `llm_provider`, which reads data/pricing.json --
# prices go stale on the vendor's schedule and belong in a file with the date
# somebody last checked them, not in a constant nobody revisits.


def estimate_cost(model: str, photographs: int) -> float:
    """Kept as the name every caller already uses. See llm_provider."""
    from photoai import llm_provider

    return llm_provider.estimate_cost(model, photographs)


_loaded_from: Path | None = None
_load_attempted = False


class CredentialSource(StrEnum):
    ENVIRONMENT = "environment"
    PROJECT_ENV = "project_env"
    WORKING_DIR_ENV = "working_dir_env"
    MISSING = "missing"


@dataclass(frozen=True)
class EnvironmentReport:
    """What was loaded and from where. Never what the value is."""

    loaded_files: tuple[Path, ...] = ()
    key_present: bool = False
    source: CredentialSource = CredentialSource.MISSING

    @property
    def env_path(self) -> Path | None:
        return self.loaded_files[0] if self.loaded_files else None


def load_project_environment(*, working_dir: Path | None = None) -> EnvironmentReport:
    """Load `.env` files, then report where the API key came from.

    Order, highest priority first:

      1. a variable already set in the real environment
      2. `<project root>/.env`
      3. `<working directory>/.env`, when that is a different directory

    `override=False` at every step, so nothing already set is replaced.
    Idempotent: calling it from several entry points is harmless.
    """
    global _loaded_from, _load_attempted

    key_was_in_environment = any(
        os.environ.get(name, "").strip() for name in (XAI_KEY_VAR, API_KEY_VAR)
    )

    loaded: list[Path] = []
    for candidate in _candidates(working_dir):
        if _load_one(candidate):
            loaded.append(candidate)

    _load_attempted = True
    _loaded_from = loaded[0] if loaded else None

    if key_was_in_environment:
        source = CredentialSource.ENVIRONMENT
    elif api_key():
        source = (
            CredentialSource.PROJECT_ENV
            if loaded and loaded[0] == PROJECT_ENV
            else CredentialSource.WORKING_DIR_ENV
        )
    else:
        source = CredentialSource.MISSING

    return EnvironmentReport(
        loaded_files=tuple(loaded),
        key_present=source is not CredentialSource.MISSING,
        source=source,
    )


def _candidates(working_dir: Path | None) -> list[Path]:
    """Project root first, then the working directory if it differs."""
    paths = [PROJECT_ENV]
    cwd_env = (Path(working_dir) if working_dir else Path.cwd()).resolve() / ".env"
    if cwd_env != PROJECT_ENV:
        paths.append(cwd_env)
    return paths


def _load_one(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
        logger.warning("python-dotenv is not installed; %s was not loaded", path.name)
        return False
    # override=False: an exported variable outranks anything in a file.
    load_dotenv(path, override=False)
    return True


# --- credentials ------------------------------------------------------------


def api_key() -> str | None:
    """The key, or None. Callers pass it straight to the client and never log it.

    xAI's variable wins where both are set: somebody who has written
    XAI_API_KEY has said which service they mean.
    """
    for name in (XAI_KEY_VAR, API_KEY_VAR):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def has_credentials() -> bool:
    return api_key() is not None


def credential_source() -> CredentialSource:
    if not has_credentials():
        return CredentialSource.MISSING
    if _loaded_from is None:
        return CredentialSource.ENVIRONMENT
    return (
        CredentialSource.PROJECT_ENV
        if _loaded_from == PROJECT_ENV
        else CredentialSource.WORKING_DIR_ENV
    )


def credential_status(language: str = "en") -> str:
    """A one-line, key-free statement of where the credential came from."""
    from photoai.i18n import t

    source = credential_source()
    if source is CredentialSource.MISSING:
        return t("creds.missing", language)
    if source is CredentialSource.ENVIRONMENT:
        return t("creds.environment", language)
    return t("creds.dotenv", language, path=str(_loaded_from))


# --- model ------------------------------------------------------------------


def resolve_provider(cli_provider: str | None = None) -> str:
    """CLI, then the environment, then the documented default.

    Same precedence as the model, in the same place, so the two cannot drift.
    """
    if cli_provider:
        return cli_provider
    return os.environ.get(PROVIDER_VAR, "").strip() or DEFAULT_PROVIDER


def resolve_model(cli_model: str | None = None) -> str:
    """CLI, then `OPENAI_MODEL`, then the documented default.

    The precedence is fixed here rather than at each call site so that the CLI
    help, the README and the behaviour cannot drift apart. There is no fourth
    step: if the resolved model does not work, the run stops rather than trying
    a different one.
    """
    if cli_model:
        return cli_model
    from_env = os.environ.get(MODEL_VAR, "").strip()
    return from_env or DEFAULT_MODEL


def model_source(cli_model: str | None = None) -> str:
    if cli_model:
        return "--model"
    if os.environ.get(MODEL_VAR, "").strip():
        return MODEL_VAR
    return "default"


# --- the client -------------------------------------------------------------


class SemanticCredentialsMissing(RuntimeError):
    """No API key. Raised before any file is decoded, never after."""


class SemanticUnavailable(RuntimeError):
    """The key exists but the API refused it, or the model is not reachable."""

    def __init__(self, message: str, *, kind: str = "unknown") -> None:
        super().__init__(message)
        self.kind = kind


def make_client(*, api_key_override: str | None = None):
    """Build an OpenAI client with the key passed explicitly.

    Explicit rather than relying on the SDK to read the environment: that
    implicit read is what made the original failure invisible until the client
    was constructed, halfway through a long run.
    """
    key = api_key_override or api_key()
    if not key:
        raise SemanticCredentialsMissing(
            f"{API_KEY_VAR} is not set in the environment or in a .env file"
        )
    import openai

    return openai.OpenAI(api_key=key)


def classify_api_error(error: Exception) -> tuple[str, str]:
    """Turn an SDK exception into (kind, safe message).

    The message is rebuilt rather than passed through, because an SDK error
    string can quote the request -- including headers.
    """
    name = type(error).__name__
    text = str(error).lower()

    if "authenticationerror" in name.lower() or "invalid_api_key" in text or "401" in text:
        return "authentication", "the API rejected the key"
    if "permissiondenied" in name.lower() or "403" in text:
        return "permission", "the key does not have access to this model"
    if "notfound" in name.lower() or "404" in text or "does not exist" in text:
        return "model_not_found", "the model does not exist or is not available to this key"
    if "insufficient_quota" in text or "no credits remaining" in text or "credit_balance" in text:
        return "quota", "the account has no remaining credits"
    if "billing" in text or "payment" in text:
        return "billing", "the account has a billing problem"
    if "ratelimit" in name.lower() or "429" in text:
        return "rate_limit", "the API rate limit was reached"
    if any(word in name.lower() for word in ("connection", "timeout", "apiconnection")):
        return "network", "the API could not be reached"
    return "unknown", f"the API call failed ({name})"


# Failures that will hit every remaining request identically. A parse error is
# about one group; an exhausted balance is about the rest of the run, and
# carrying on spends minutes to produce a report that is wrong about every
# photograph after the point the credit ran out.
FATAL_API_KINDS = frozenset(
    {"authentication", "permission", "model_not_found", "quota", "billing"}
)


def is_fatal_api_error(error: Exception) -> bool:
    kind, _ = classify_api_error(error)
    return kind in FATAL_API_KINDS

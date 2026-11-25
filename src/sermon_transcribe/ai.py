"""Provider adapters and bounded generation retries for sermon processing."""

import json
import logging
import math
import re
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache

from sermon_transcribe.settings import COMPATIBLE_PROVIDER, PROVIDERS, compatible_base_url, compatible_opener

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MAX_OUTPUT_TOKENS = 16384
MAX_ATTEMPTS = 4
CONTEXT_HEADROOM = 256
MAX_RETRY_DELAY = 60
logger = logging.getLogger(__name__)

PROFILES = {
    "cleanup": (0.1, None),
    "summary": (0.2, 4096),
    "summary_notes": (0.2, 2048),
}
THINKING_BLOCK = re.compile(r"<(think|thinking|reasoning|analysis)(?:\s[^>]*)?>.*?</\1\s*>", re.I | re.S)


class GenerationError(RuntimeError):
    def __init__(self, kind, message, *, parameter=None, retry_after=None):
        super().__init__(message)
        self.kind = kind
        self.parameter = parameter
        self.retry_after = retry_after


class CreditBalanceError(GenerationError):
    def __init__(self, message):
        super().__init__("credits", message)


def configure_ai_logging():
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ModelLimits:
    context: int | None = None
    output: int | None = None
    parameters: frozenset = frozenset()
    llama_cpp: bool = False


@dataclass
class Usage:
    attempts: int = 0
    retries: int = 0
    totals: dict = field(default_factory=dict)

    def record(self, attempt, metrics):
        self.attempts += 1
        self.retries += attempt > 1
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.totals[name] = self.totals.get(name, 0) + value

    def fields(self):
        return dict(attempts=self.attempts, retries=self.retries, **self.totals)


def estimate_tokens(text: str) -> int:
    # A conservative estimate for ordinary sermon prose, including non-ASCII text.
    return math.ceil(len(text.encode("utf-8")) / 3)


def _positive_int(value):
    return value if type(value) is int and value > 0 else None


def _object(value):
    return value if isinstance(value, dict) else {}


def _first_choice(payload):
    choices = payload.get("choices")
    return _object(choices[0]) if isinstance(choices, list) and choices else {}


def _log_label(value):
    return value if isinstance(value, str) and re.fullmatch(r"[\w. /:@-]{1,120}", value) else None


@lru_cache(maxsize=2)
def _openrouter_models(cache_hour):
    # This public catalogue is shared across jobs. Never cache account credentials.
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=5) as response:
        return {
            model["id"]: model for model in json.load(response)["data"]
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        }


def _limits(model):
    top = _object(model.get("top_provider"))
    contexts = [
        limit for limit in (_positive_int(model.get("context_length")), _positive_int(top.get("context_length")))
        if limit is not None
    ]
    return ModelLimits(
        context=min(contexts, default=None),
        output=_positive_int(top.get("max_completion_tokens")),
        parameters=frozenset(model.get("supported_parameters") or ()),
    )


def _retry_delay(value):
    if not value:
        return None
    try:
        return max(0, int(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _api_error(service, status, error, headers, settings):
    status = status if type(status) is int else None
    message = str(error.get("message", "")).lower()
    code = str(error.get("type") or error.get("code") or "").lower()
    parameter = str(error.get("param", "")).lower()
    retry_after = _retry_delay(headers.get("Retry-After")) if headers else None
    if (status == 402 and retry_after is None) or "credit balance is too low" in message:
        return CreditBalanceError(f"{service} reported insufficient credit or budget.")
    if status in (408, 429, 500, 502, 503, 504, 529) or (status == 402 and retry_after is not None):
        return GenerationError("transient", f"{service} returned HTTP {status}.", retry_after=retry_after)
    if status in (401, 403) and not any(word in code or word in message for word in ("guardrail", "moderation", "content policy")):
        return GenerationError("provider", f"{service} rejected the API key or model access (HTTP {status}).")
    raw_status = f"HTTP {status}" if status is not None else "an API error"
    if any(word in code or word in message for word in ("context_length_exceeded", "context window", "context length", "context size", "exceeds the context")):
        return GenerationError("context", f"The request exceeded the context window on {service}.")
    if any(word in code or word in message for word in ("guardrail", "moderation", "content policy", "content_filter", "safety filter")):
        return GenerationError("filtered", f"{service} blocked the request under its content policy.")
    if code in {"max_tokens", "max_output_tokens", "length"}:
        return GenerationError("length", f"{service} reached its output token limit.")
    rejection = ("unsupported", "not support", "unknown", "unrecognized", "invalid", "not allowed", "only the default")
    if status in (400, 422) and any(word in code or word in message for word in rejection):
        for name, aliases in (
            ("reasoning", ("reasoning", "effort")),
            ("chat_template_kwargs", ("chat_template_kwargs", "enable_thinking")),
            ("temperature", ("temperature",)),
        ):
            if name in settings and any(alias in parameter or alias in message for alias in aliases):
                return GenerationError("unsupported", f"{service} rejected {name}.", parameter=name)
    if status == 404:
        return GenerationError("provider", f"{service} could not find the configured model or endpoint (HTTP 404).")
    return GenerationError("provider", f"{service} rejected the request ({raw_status}).")


def _completion(payload, anthropic):
    reasoning_present = False
    if anthropic:
        blocks = payload.get("content") or []
        finish = payload.get("stop_reason")
        native = None
    else:
        choice = _first_choice(payload)
        finish, native = choice.get("finish_reason"), choice.get("native_finish_reason")
        message = _object(choice.get("message"))
        blocks = message.get("content")
        reasoning_present = bool(message.get("reasoning") or message.get("reasoning_content") or message.get("reasoning_details"))
        if message.get("refusal"):
            finish = "refusal"
    usage = _object(payload.get("usage"))
    details = _object(usage.get("completion_tokens_details"))
    metrics = {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
        "reasoning_tokens": details.get("reasoning_tokens", usage.get("reasoning_tokens")),
        "cost": usage.get("cost"),
    }
    if isinstance(blocks, list):
        blocks = "\n".join(
            block["text"] for block in blocks
            if isinstance(block, dict) and block.get("type") in ("text", "output_text") and isinstance(block.get("text"), str)
        )
    metrics = {name: value if type(value) in (int, float) else None for name, value in metrics.items()}
    return blocks, finish, native, metrics, reasoning_present


def _visible_text(content, finish, native, metrics, budget):
    reasons = {str(reason).lower() for reason in (finish, native) if reason}
    if reasons & {"model_context_window_exceeded", "context_length_exceeded"}:
        raise GenerationError("context", "AI completion reached the model's context window.")
    if reasons & {"length", "max_tokens", "max_output_tokens", "limit"}:
        raise GenerationError("length", f"AI completion reached the output token limit ({budget} tokens).")
    if reasons & {"content_filter", "safety", "blocked", "refusal", "recitation", "blocklist", "prohibited_content", "spii"}:
        raise GenerationError("filtered", "The AI provider filtered the completion.")
    if finish and str(finish).lower() not in {"stop", "end_turn", "stop_sequence", "eos", "eog"}:
        raise GenerationError("provider", "The AI provider ended the completion without a normal stop.")
    output = metrics.get("output_tokens")
    if not reasons and isinstance(output, int) and output >= budget:
        raise GenerationError("length", f"AI completion used its full output budget ({budget} tokens) without a finish reason.")
    text = THINKING_BLOCK.sub("", content).strip() if isinstance(content, str) else ""
    if re.search(r"</?(?:think|thinking|reasoning|analysis)\b|<\|channel\|>\s*analysis", text, re.I):
        text = ""
    if not text:
        reasoning = metrics.get("reasoning_tokens")
        if isinstance(reasoning, int) and reasoning > 0 and isinstance(output, int) and output >= budget - 16:
            raise GenerationError("length", f"AI reasoning exhausted the output budget ({budget} tokens).")
        raise GenerationError("empty", "The AI provider returned no usable text.")
    return text


class AIClient:
    def __init__(self, provider, model, api_key, *, base_url=None, enable_thinking=None, job_id=None, cancelled=None, timeout=120):
        if provider not in PROVIDERS or not model:
            raise GenerationError("provider", "Select a supported AI provider and model in Settings.")
        self.provider = provider
        self.model = model
        self.job_id = job_id
        self.cancelled = cancelled
        self.timeout = timeout
        self.total = Usage()
        self.stage_usage = Usage()
        self.rejected = set()
        self.model_limits = None
        self.anthropic = provider == "anthropic"
        self.open_request = urllib.request.urlopen
        self.headers = {"Content-Type": "application/json"}
        self.enable_thinking = enable_thinking
        self.base_url = None
        if self.anthropic:
            self.service, self.url = "Anthropic", ANTHROPIC_API_URL
            self.headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        else:
            if api_key:
                self.headers["Authorization"] = f"Bearer {api_key}"
            if provider == "openrouter":
                self.service, self.url = "OpenRouter", OPENROUTER_API_URL
                self.headers["X-Title"] = "Sermon Transcribe"
            elif provider == COMPATIBLE_PROVIDER:
                self.base_url = compatible_base_url(base_url)
                self.service, self.url = "OpenAI-compatible server", self.base_url + "/chat/completions"
                self.open_request = compatible_opener.open
            else:
                self.service, self.url = "OpenAI", OPENAI_API_URL

    @property
    def limits(self):
        if self.model_limits is not None:
            return self.model_limits
        limits = ModelLimits()
        try:
            if self.provider == "openrouter":
                models = _openrouter_models(int(time.time() // 3600))
                limits = _limits(models.get(self.model, {}))
            elif self.base_url:
                # llama.cpp exposes the configured slot context at /props, not the
                # training context reported by the loaded model at /v1/models.
                root = self.base_url.removesuffix("/v1")
                request = urllib.request.Request(root + "/props", headers=self.headers)
                with self.open_request(request, timeout=3) as response:
                    settings = json.load(response).get("default_generation_settings") or {}
                if settings:
                    limits = ModelLimits(context=_positive_int(settings.get("n_ctx")), llama_cpp=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            logger.info("AI model limits unavailable job=%s provider=%s model=%s", self.job_id, self.provider, self.model)
        self.model_limits = limits
        return limits

    @contextmanager
    def report(self, stage=None):
        start = time.monotonic()
        previous_usage = self.stage_usage
        usage = self.stage_usage = Usage() if stage else self.total
        status = "failed"
        try:
            yield
            status = "completed"
        except GenerationError as exc:
            status = exc.kind
            raise
        finally:
            self.stage_usage = previous_usage
            logger.info(
                "AI %s job=%s provider=%s model=%s status=%s seconds=%.2f %s",
                stage or "sermon", self.job_id, self.provider, self.model, status, time.monotonic() - start, usage.fields(),
            )

    def chunk_chars(self, configured, *, cleanup=False, output_reserve=1024):
        # Leave room for instructions and a corrective prompt. Cleaning output
        # scales with its input. Summary notes can use a fixed output reserve.
        if self.limits.context is not None:
            available = self.limits.context - CONTEXT_HEADROOM - 512
            allowance = available // 2 if cleanup else available - output_reserve
            configured = min(configured, max(256, allowance * 3))
        if cleanup and self.limits.output is not None:
            configured = min(configured, self.limits.output * 2)
        return configured

    def _parameters(self, temperature):
        parameters = {"temperature": temperature}
        if self.provider == "openrouter" and self.limits.parameters:
            if "temperature" not in self.limits.parameters:
                parameters.pop("temperature")
            if "reasoning" in self.limits.parameters:
                parameters["reasoning"] = {"effort": "minimal", "exclude": True}
        if self.base_url and self.enable_thinking is not None:
            parameters["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        return {name: value for name, value in parameters.items() if name not in self.rejected}

    def _request(self, prompt, parameters, budget):
        token_field = "max_completion_tokens" if self.provider == "openai" else "max_tokens"
        message = [{"type": "text", "text": prompt}] if self.anthropic else prompt
        payload = {"model": self.model, token_field: budget, "messages": [{"role": "user", "content": message}], **parameters}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=self.headers, method="POST")
        try:
            with self.open_request(request, timeout=self.timeout) as response:
                result = json.load(response)
                status = getattr(response, "status", None)
                headers = getattr(response, "headers", None)
        except urllib.error.HTTPError as exc:
            status, headers = exc.code, exc.headers
            with exc:
                try:
                    result = json.load(exc)
                except (ValueError, OSError):
                    result = {}
            result = _object(result)
            return result, _api_error(self.service, status, _object(result.get("error")), headers, parameters)
        except OSError:
            return {}, GenerationError("transient", f"Could not reach {self.service} or the request timed out.")
        except ValueError:
            return {}, GenerationError("provider", f"{self.service} returned invalid JSON.")
        if not isinstance(result, dict):
            return {}, GenerationError("provider", f"{self.service} returned an invalid response.")
        if isinstance(error := result.get("error"), dict):
            return result, _api_error(self.service, error.get("code", status), error, headers, parameters)
        if isinstance(error := _first_choice(result).get("error"), dict):
            return result, _api_error(self.service, error.get("code"), error, headers, parameters)
        return result, None

    def generate(self, prompt, task, *, source=None, validate=None, max_tokens=None, temperature=None):
        profile_temp, profile_budget = PROFILES[task]
        # Punctuation, formatting and limited reasoning can expand the input.
        min_cleanup = estimate_tokens(source or "") if task == "cleanup" else 0
        budget = max_tokens or profile_budget or max(2048, math.ceil(min_cleanup * 1.4) + 256)
        parameters = self._parameters(profile_temp if temperature is None else temperature)
        correction = ""
        validation_retry = False
        empty_retry = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self.cancelled and self.cancelled():
                raise GenerationError("cancelled", "Job cancelled by user.")
            request_text = correction + prompt
            limit = min(self.limits.output or MAX_OUTPUT_TOKENS, max_tokens or MAX_OUTPUT_TOKENS)
            if self.limits.context is not None:
                limit = min(limit, self.limits.context - estimate_tokens(request_text) - CONTEXT_HEADROOM)
            if limit < 1 or (max_tokens is None and limit < max(128, min_cleanup)):
                raise GenerationError("context", f"The configured model has insufficient context or output for this {task} request.")
            effective_budget = min(budget, limit)
            start = time.monotonic()
            result, failure = self._request(request_text, parameters, effective_budget)
            content, finish, native, metrics, reasoning_present = _completion(result, self.anthropic)
            self.stage_usage.record(attempt, metrics)
            if self.stage_usage is not self.total:
                self.total.record(attempt, metrics)
            if failure is None:
                try:
                    text = _visible_text(content, finish, native, metrics, effective_budget)
                    issue = validate(text) if validate else None
                    if issue:
                        failure = GenerationError("validation", issue)
                except GenerationError as exc:
                    failure = exc
            logger.info(
                "AI request job=%s stage=%s provider=%s model=%s served_model=%s inference_provider=%s "
                "request=%s attempt=%d seconds=%.2f finish=%s native_finish=%s outcome=%s "
                "max_output_tokens=%d settings=%s usage=%s detail=%s",
                self.job_id, task, self.provider, self.model, _log_label(result.get("model")), _log_label(result.get("provider")),
                _log_label(result.get("id")), attempt, time.monotonic() - start, _log_label(finish), _log_label(native),
                failure.kind if failure else "completed", effective_budget, parameters, metrics, failure or "",
            )
            if failure is None:
                return text
            if attempt == MAX_ATTEMPTS:
                raise failure
            if failure.kind == "length":
                lowered_reasoning = False
                if reasoning_present or (metrics.get("reasoning_tokens") or 0) > 0:
                    if parameters.get("reasoning", {}).get("effort") == "minimal" and "reasoning_none" not in self.rejected:
                        parameters["reasoning"] = {"effort": "none", "exclude": True}
                        lowered_reasoning = True
                    if self.limits.llama_cpp and self.enable_thinking is None and "chat_template_kwargs" not in parameters and "chat_template_kwargs" not in self.rejected:
                        parameters["chat_template_kwargs"] = {"enable_thinking": False}
                        lowered_reasoning = True
                if effective_budget == limit and not lowered_reasoning:
                    raise failure
                budget = min(limit, effective_budget * 2)
            elif failure.kind == "unsupported" and failure.parameter in parameters:
                if failure.parameter == "reasoning" and parameters["reasoning"].get("effort") == "none":
                    self.rejected.add("reasoning_none")
                    parameters["reasoning"] = {"effort": "minimal", "exclude": True}
                else:
                    self.rejected.add(failure.parameter)
                    del parameters[failure.parameter]
            elif failure.kind == "validation" and not validation_retry:
                correction = f"Correct this issue in the next answer: {failure}\n\n"
                validation_retry = True
            elif failure.kind == "empty" and not empty_retry:
                correction = "Return the requested text in the final answer.\n\n" + correction
                empty_retry = True
            elif failure.kind == "transient" and (failure.retry_after or 0) <= MAX_RETRY_DELAY:
                time.sleep(max(2 ** (attempt - 1), failure.retry_after or 0))
            else:
                raise failure

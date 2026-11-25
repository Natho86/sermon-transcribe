import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from sermon_transcribe import ai, cleanup, settings

SUMMARY = "Summary:\nThe preacher discusses hope.\n\nBible references:\n- Psalm 18:2"
TRANSCRIPT = " ".join(
    ["We turn to the Lord for strength when the storms arrive. This message invites the church to pray and trust in God's promises."] * 4
)


def chat(content, finish="stop", **kwargs):
    return {"choices": [{"message": {"content": content, "reasoning": "private chain"}, "finish_reason": finish}], **kwargs}


def http_error(status, message, *, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    body = json.dumps({"error": {"message": message}}).encode()
    return HTTPError("http://fixture", status, "", headers, io.BytesIO(body))


class AITests(unittest.TestCase):
    def client(self, provider="openrouter", *, replies, limits=None, **kwargs):
        client = ai.AIClient(provider, "fixture-model", "fixture-secret", job_id="job-123", **kwargs)
        client.model_limits = limits or ai.ModelLimits()
        sent = []
        replies = iter(replies)

        def request(req, **_):
            sent.append(json.loads(req.data))
            response = next(replies)
            if isinstance(response, Exception):
                raise response
            return io.BytesIO(json.dumps(response).encode())

        client.open_request = request
        return client, sent

    def test_reasoning_exhaustion_adapts_before_format_validation(self):
        truncated = chat(SUMMARY, "length", usage={
            "prompt_tokens": 1851, "completion_tokens": 4096, "completion_tokens_details": {"reasoning_tokens": 4000}, "cost": 0.005,
        }, provider="Google")
        truncated["choices"][0]["native_finish_reason"] = "MAX_TOKENS"
        client, sent = self.client(
            replies=[truncated, chat("<think>hidden reasoning</think>" + SUMMARY, usage={"cost": 0.007})],
            limits=ai.ModelLimits(parameters=frozenset({"temperature", "reasoning"})),
        )
        with self.assertLogs(ai.logger, level="INFO") as captured, client.report():
            self.assertEqual(cleanup.summarize_transcript("Source transcript.", client), SUMMARY)
        self.assertEqual([req["max_tokens"] for req in sent], [4096, 8192])
        self.assertEqual([req["temperature"] for req in sent], [0.2, 0.2])
        self.assertEqual(sent[0]["reasoning"], {"effort": "minimal", "exclude": True})
        self.assertEqual(sent[1]["reasoning"], {"effort": "none", "exclude": True})
        self.assertEqual(sent[0]["messages"], sent[1]["messages"])
        logs = "\n".join(captured.output)
        self.assertIn("MAX_TOKENS", logs)
        self.assertIn("outcome=length", logs)
        self.assertIn("reasoning_tokens", logs)
        self.assertIn("'cost': 0.012", logs)
        for secret in ["fixture-secret", "Source transcript.", "hidden reasoning", "private chain", SUMMARY]:
            self.assertNotIn(secret, logs)

    def test_format_retry_identifies_actual_word_count(self):
        too_long = SUMMARY.replace("The preacher discusses hope.", "word " * 41)
        client, sent = self.client(replies=[chat(too_long), chat(SUMMARY)])
        self.assertEqual(cleanup.summarize_transcript("A source.", client), SUMMARY)
        self.assertIn("41 words", sent[1]["messages"][0]["content"])
        self.assertEqual(sent[0]["max_tokens"], sent[1]["max_tokens"])
        self.assertNotIn("reasoning", sent[0])

    def test_empty_and_filter_are_distinct(self):
        client, sent = self.client(replies=[chat("", finish="content_filter")])
        with self.assertRaisesRegex(ai.GenerationError, "filtered") as caught:
            client.generate("A prompt", "summary")
        self.assertEqual(caught.exception.kind, "filtered")
        self.assertEqual(len(sent), 1)
        client, sent = self.client(replies=[chat("<think>Unclosed"), chat(None)])
        with self.assertRaisesRegex(ai.GenerationError, "no usable text") as caught:
            client.generate("A prompt", "summary")
        self.assertEqual(caught.exception.kind, "empty")
        self.assertEqual(len(sent), 2)

    def test_unsupported_reasoning_parameter_removed_for_later_requests(self):
        client, sent = self.client(
            replies=[http_error(400, "Unsupported parameter reasoning"), chat(SUMMARY), chat(SUMMARY)],
            limits=ai.ModelLimits(parameters=frozenset({"reasoning"})),
        )
        self.assertEqual(client.generate("Prompt", "summary"), SUMMARY)
        self.assertEqual(client.generate("Another prompt", "summary"), SUMMARY)
        self.assertIn("reasoning", sent[0])
        self.assertNotIn("reasoning", sent[1])
        self.assertNotIn("reasoning", sent[2])

    def test_local_thinking_preference_and_anthropic_blocks(self):
        local, sent = self.client(
            settings.COMPATIBLE_PROVIDER, replies=[chat([{"type": "text", "text": SUMMARY}])],
            base_url="http://localhost:8080/v1", enable_thinking=False,
        )
        self.assertEqual(local.generate("Prompt", "summary"), SUMMARY)
        self.assertEqual(sent[0]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning", sent[0])
        anthropic, sent = self.client("anthropic", replies=[{
            "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": SUMMARY}],
            "usage": {"input_tokens": 100, "output_tokens": 40},
        }])
        self.assertEqual(anthropic.generate("Prompt", "summary"), SUMMARY)
        self.assertEqual(sent[0]["messages"][0]["content"], [{"type": "text", "text": "Prompt"}])
        self.assertEqual(anthropic.total.fields()["output_tokens"], 40)

    def test_missing_usage_and_openai_token_parameter(self):
        client, sent = self.client("openai", replies=[chat(SUMMARY, finish=None)])
        self.assertEqual(client.generate("Prompt", "summary"), SUMMARY)
        self.assertEqual(sent[0]["max_completion_tokens"], 4096)
        self.assertEqual(client.total.fields(), {"attempts": 1, "retries": 0})

    def test_local_context_and_output_cap(self):
        client, sent = self.client(
            settings.COMPATIBLE_PROVIDER, replies=[chat(SUMMARY)],
            base_url="http://localhost:8080/v1", limits=ai.ModelLimits(context=4096, output=32768),
        )
        self.assertLess(client.chunk_chars(8000, cleanup=True), 8000)
        self.assertEqual(client.generate("Prompt", "summary"), SUMMARY)
        self.assertLess(sent[0]["max_tokens"], 4096)
        with self.assertRaisesRegex(ai.GenerationError, "insufficient context"):
            client.generate("x" * 15000, "summary")

    def test_model_discovery_uses_advertised_capabilities(self):
        catalogue = {"gemini-fixture": {
            "context_length": 12000,
            "top_provider": {"context_length": 8192, "max_completion_tokens": 6144},
            "supported_parameters": ["max_tokens", "reasoning"],
        }}
        with patch.object(ai, "_openrouter_models", return_value=catalogue):
            cloud = ai.AIClient("openrouter", "gemini-fixture", "fixture")
            self.assertEqual(cloud.limits.context, 8192)
            self.assertEqual(cloud.limits.output, 6144)
            self.assertEqual(cloud._parameters(0.1), {"reasoning": {"effort": "minimal", "exclude": True}})
        local = ai.AIClient(settings.COMPATIBLE_PROVIDER, "fixture", "", base_url="http://localhost:8080/v1")
        payload = io.BytesIO(b'{"default_generation_settings":{"n_ctx":4096}}')
        with patch.object(local, "open_request", return_value=payload) as request:
            self.assertEqual(local.limits.context, 4096)
            self.assertTrue(local.limits.llama_cpp)
            self.assertEqual(request.call_args.args[0].full_url, "http://localhost:8080/props")

    def test_mandatory_reasoning_falls_back_to_minimal_with_larger_budget(self):
        client, sent = self.client(
            replies=[chat("", "length"), http_error(400, "reasoning effort none is unsupported"), chat(SUMMARY)],
            limits=ai.ModelLimits(parameters=frozenset({"reasoning"})),
        )
        self.assertEqual(client.generate("Prompt", "summary"), SUMMARY)
        self.assertEqual([item["reasoning"]["effort"] for item in sent], ["minimal", "none", "minimal"])
        self.assertEqual([item["max_tokens"] for item in sent], [4096, 8192, 8192])

    def test_retry_after_and_retry_exhaustion(self):
        responses = [http_error(429, "rate limited", retry_after=7)]
        responses.extend(http_error(503, "busy") for _ in range(ai.MAX_ATTEMPTS - 1))
        client, sent = self.client(replies=responses)
        with patch.object(ai.time, "sleep") as sleep:
            with self.assertRaisesRegex(ai.GenerationError, "HTTP 503"):
                client.generate("Prompt", "summary")
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [7, 2, 4])
        self.assertEqual(len(sent), ai.MAX_ATTEMPTS)

    def test_token_cap_never_accepts_truncated_text(self):
        client, sent = self.client(
            replies=[chat(SUMMARY, "length")],
            limits=ai.ModelLimits(output=2048),
        )
        with self.assertRaisesRegex(ai.GenerationError, "2048 tokens"):
            cleanup.summarize_transcript("Transcript", client)
        self.assertEqual(len(sent), 1)

    def test_incomplete_cleanup_does_not_publish_output_or_generate_summary(self):
        client, sent = self.client("anthropic", replies=[
            {"content": [{"type": "text", "text": "Short topic."}], "stop_reason": "end_turn"},
            {"content": [{"type": "text", "text": "Still too short."}], "stop_reason": "end_turn"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, cleaned, summary = (root / name for name in ("sermon_raw.txt", "sermon.txt", "sermon_summary.txt"))
            raw.write_text(TRANSCRIPT)
            args = SimpleNamespace(skip_existing=True, reprocess=False, max_chars=8000, max_tokens=None, temperature=None)
            with self.assertRaisesRegex(ai.GenerationError, "substantially shorter"):
                cleanup.process_transcript_file(raw, cleaned, summary, args, client)
            self.assertEqual(raw.read_text(), TRANSCRIPT)
            self.assertFalse(cleaned.exists())
            self.assertFalse(summary.exists())
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0]["temperature"], 0.1)
        self.assertIn("substantially shorter", sent[1]["messages"][0]["content"][0]["text"])

    def test_saved_provider_without_new_fields_is_compatible(self):
        config = {"keys": [{"enabled": True, "key": "fixture", "provider": "openrouter", "model": "model"}]}
        with patch.object(settings, "load_keys", return_value=config):
            self.assertEqual(settings.get_active_provider(), {"key": "fixture", "provider": "openrouter", "model": "model"})
            config["keys"][0].update(provider=settings.COMPATIBLE_PROVIDER, base_url="http://localhost:8080/v1")
            self.assertNotIn("enable_thinking", settings.get_active_provider())


if __name__ == "__main__":
    unittest.main()

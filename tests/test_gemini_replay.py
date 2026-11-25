import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sermon_transcribe.ai import AIClient, ModelLimits
from sermon_transcribe.cleanup import (
    DISCLAIMER_TEXT, SUMMARY_DISCLAIMER_TEXT, normalize_summary, process_transcript_file,
)

CASES = (
    (
        """Summary:
A sermon examining the execution of John the Baptist by Herod Antipas in Matthew 14, contrasting Herod's pride with faithful obedience to Jesus.

Main points:
- The truth that Herod refused to obey || So the first thing we're going to look at is the truth that Herod refused to obey.
- The fear that ruled Herod || But the next thing we look at is the fear that ruled Herod.
- The King John came to announce || And that just kind of brings us to a final point: the King John came to announce.

Bible references:
- Matthew 14:1–12
- Matthew 4:17
- Mark 6:20
- Matthew 17
- Matthew 20""",
        """So the first thing we're going to look at is the truth that Herod refused to obey. Okay, have a look.

But the next thing we look at is the fear that ruled Herod. So Herod was this important man.

Like this poor man, John the Baptist. And that just kind of brings us to a final point: the King John came to announce.""",
    ),
    (
        """Summary:
This sermon considers the feeding of the five thousand in Matthew 14, focusing on Jesus' provision and the role of his disciples.

Main points:
- Jesus is presenting himself in this miracle, first of all, as the compassionate one. || And Jesus is presenting himself in this miracle, first of all, as the compassionate one.
- He's also presenting himself as the head of the household. || But Jesus is presenting himself not just as the compassionate one in this passage; he's also presenting himself as the head of the household.
- He is the abundant supplier of our needs. || But thirdly, he is also—and this is the main point of the passage—he is the abundant supplier of our needs.
- The disciples are unable to provide for the people on their own. || You see, what is being emphasised in this passage, first of all, is that the disciples are unable to provide for the people on their own.
- They are able to provide for people with what Jesus supplies to them. || These disciples are unable to provide for people on their own, and yet, at the same time, they are able to provide for people with what Jesus supplies to them.

Bible references:
- Matthew 14:13–21
- Matthew 9:36
- Exodus 16
- 2 Kings 4:42–44
- Revelation 19:9""",
        """And Jesus is presenting himself in this miracle, first of all, as the compassionate one. Can I invite you to open your Bibles?

But Jesus is presenting himself not just as the compassionate one in this passage; he's also presenting himself as the head of the household. Let me read verses 15 to 19:

But thirdly, he is also—and this is the main point of the passage—he is the abundant supplier of our needs.

You see, what is being emphasised in this passage, first of all, is that the disciples are unable to provide for the people on their own. This multitude before them—totally unable to give them what they really need. These disciples are unable to provide for people on their own, and yet, at the same time, they are able to provide for people with what Jesus supplies to them.""",
    ),
)


class GeminiReplayTests(unittest.TestCase):
    def test_openrouter_completion_through_cleanup_and_saved_output(self):
        for index, (completion, cleaned) in enumerate(CASES):
            with self.subTest(sermon=index), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                raw, text, summary = (path / name for name in ("sermon_raw.txt", "sermon.txt", "sermon_summary.txt"))
                raw.write_text(cleaned.replace("fear that ruled Herod", "fear, the rule, Herod").replace("emphasised", "emphasized"))
                client = AIClient("openrouter", "fixture", "", job_id="gemini-replay")
                client.model_limits = ModelLimits()
                requests = []

                def respond(request, **kwargs):
                    prompt = json.loads(request.data)["messages"][0]["content"]
                    requests.append(prompt)
                    body = cleaned if len(requests) == 1 else completion
                    return io.BytesIO(json.dumps({"choices": [{
                        "finish_reason": "stop", "message": {"content": body},
                    }]}).encode())

                client.open_request = respond
                args = SimpleNamespace(
                    skip_existing=True, reprocess=False, max_chars=8000, max_tokens=None,
                    temperature=None, summary_max_chars=None, summary_max_tokens=None,
                )
                with self.assertLogs("sermon_transcribe.ai", "INFO") as recorded:
                    process_transcript_file(raw, text, summary, args, client)
                published = "\n".join(line.split(" || ", 1)[0] for line in completion.splitlines())
                logs = "\n".join(recorded.output)
                self.assertIn(f"kind=cleaned filename='sermon.txt' chars={len(cleaned)} "
                              f"sha256={hashlib.sha256(cleaned.encode()).hexdigest()}", logs)
                self.assertNotIn(cleaned, logs)
                self.assertEqual(text.read_text(), DISCLAIMER_TEXT + "\n\n" + cleaned + "\n")
                self.assertEqual(summary.read_text(), SUMMARY_DISCLAIMER_TEXT + "\n\n" + published + "\n")
                self.assertEqual(len(requests), 2)
                self.assertTrue(requests[1].endswith(cleaned))
                self.assertEqual(normalize_summary(published), published)
                if index == 0:
                    # The cleaned wording is unsupported when the raw text is
                    # actually the summary source.
                    self.assertNotIn("- The fear that ruled Herod", normalize_summary(completion, transcript=raw.read_text()))

    def test_chunked_replay_verifies_against_full_saved_transcript(self):
        completion, cleaned = CASES[1]
        client = AIClient("openrouter", "fixture", "", job_id="gemini-chunked")
        client.model_limits = ModelLimits()
        seen = []

        def respond(request, **kwargs):
            prompt = json.loads(request.data)["messages"][0]["content"]
            seen.append(prompt)
            body = "Subjects discussed:\n- The feeding of the five thousand." if prompt.startswith("Extract") else completion
            return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": body}}]}).encode())

        client.open_request = respond
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw, text, summary = (path / name for name in ("sermon_raw.txt", "sermon.txt", "sermon_summary.txt"))
            text.write_text(DISCLAIMER_TEXT + "\n\n" + cleaned + "\n")
            args = SimpleNamespace(
                skip_existing=True, reprocess=False, max_chars=8000, max_tokens=None,
                temperature=None, summary_max_chars=450, summary_max_tokens=None,
            )
            process_transcript_file(raw, text, summary, args, client)
            published = "\n".join(line.split(" || ", 1)[0] for line in completion.splitlines())
            self.assertEqual(summary.read_text(), SUMMARY_DISCLAIMER_TEXT + "\n\n" + published + "\n")
            self.assertTrue(all(prompt.startswith("Extract") for prompt in seen[:-1]))
            self.assertIn("consecutive sections", seen[-1])
            self.assertNotIn(cleaned, seen[-1])

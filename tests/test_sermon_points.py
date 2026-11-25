import unittest

from sermon_transcribe.bible_references import normalize_bible_references
from sermon_transcribe.sermon_points import log_point_notes, verified_main_points


class PointRegressionTests(unittest.TestCase):
    def test_matthew_14_parallel_points_and_repetition(self):
        points = [
            "Jesus is the only worthy object of faith",
            "Jesus is the only object of saving faith",
            "Jesus is the only object of living faith",
        ]
        recap = f"So {points[0]},\n{points[1]} and\n{points[2]}."
        source = f"Some context. The first one is: {points[0]}.\n\n{recap}\n\n{recap}"
        proposed = "\n".join(f"- {point}. ||\n  {point}" for point in reversed(points))
        self.assertEqual(verified_main_points(proposed, source), [point + "." for point in points])

    def test_matthew_15_remove_only_recognised_heading_lead_ins(self):
        first = "Firstly, the question of authority. God's truth regulates human tradition."
        second = ("Secondly, we'll think about the question of acceptance.\n"
                  "God's sight exposes the human heart and would have us be cleaned by him.")
        points = [
            "Authority: God’s truth regulates human tradition.",
            "Acceptance: God's sight exposes the human heart and would have us be cleaned by him.",
        ]
        proposed = f"- {points[0]} || {first}\n- {points[1]} || {second}\n"
        proposed += f"- {points[1].replace('cleaned', 'cleansed')} || {second}"
        introduction = "First one relates to the question of authority,\nwhich is that God's truth regulates human tradition."
        self.assertEqual(verified_main_points(proposed, introduction + "\n" + second + "\n" + first), points)
        self.assertEqual(verified_main_points(f"- {points[0]} || {introduction}", introduction), points[:1])

    def test_five_points_in_source_order_and_sensitive_rejections_counted(self):
        source = " ".join(f"{name}, {word} matters." for name, word in zip(
            ("Firstly", "Secondly", "Thirdly", "Fourthly", "Fifthly"), ("Prayer", "Hope", "Love", "Faith", "Grace"),
        ))
        entries = "\n".join(f"- {word} matters. || {word} matters." for word in ("Grace", "Faith", "Love", "Hope", "Prayer"))
        entries += "\n- Missing.\n- Invented. || Hidden quotation.\n- Short. || Prayer matters."
        with self.assertLogs("sermon_transcribe.ai.points", "INFO") as captured:
            self.assertEqual(verified_main_points(entries, source, job_id="job-1"),
                             ["Prayer matters.", "Hope matters.", "Love matters.", "Faith matters.", "Grace matters."])
            self.assertEqual(verified_main_points("None noted.", source), [])
            log_point_notes("Announced points and headings:\n- One || secret evidence", "job-1", "extraction")
        logs = "\n".join(captured.output)
        self.assertIn("generated=8 parsed=7 verified=5 rejected=3", logs)
        self.assertIn("generated=0 parsed=0 verified=0 rejected=0", logs)
        self.assertIn("stage=extraction generated=1 with_evidence=1", logs)
        for reason in ("missing_quotation", "quotation_not_in_transcript", "point_not_in_quotation"):
            self.assertIn(reason, logs)
        for private in ("Hidden quotation", "secret evidence", "Prayer matters"):
            self.assertNotIn(private, logs)

    def test_does_not_drop_qualifiers_from_nonparallel_sentence(self):
        source = "I do not teach that Jesus ignores sadness and Jesus ignores suffering."
        self.assertEqual(verified_main_points("- Jesus ignores suffering. || " + source, source), [])


class ReferenceFormattingTests(unittest.TestCase):
    def test_john_11_format_variants(self):
        supplied = " John 11:17-27\n• **John 11:25**\n2) *Psalm 23*\n3. __Romans__ 8:38–39"
        self.assertEqual(normalize_bible_references(supplied), "- John 11:17–27\n- Psalm 23\n- Romans 8:38–39")

    def test_no_references_variants_and_invalid_content(self):
        for text in ("None noted", "- None noted.", "• **None noted**", "1. None noted"):
            with self.subTest(text=text):
                self.assertEqual(normalize_bible_references(text), "None noted.")
        for text in ("None noted.\nJohn 11:17–27", "• John the Baptist", "John 11:27–17", "John 11:17–27 and Romans 8"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                normalize_bible_references(text)

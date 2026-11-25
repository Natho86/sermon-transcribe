import unittest

from sermon_transcribe.bible_references import normalize_bible_references


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

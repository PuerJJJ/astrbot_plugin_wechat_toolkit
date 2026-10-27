import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).resolve().parent / "image_options.py"
spec = importlib.util.spec_from_file_location("image_options_test", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Reply:
    status = 200

    def __init__(self, data):
        self.content = self
        self.data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def iter_chunked(self, n):
        yield json.dumps(self.data).encode()


class Session:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, *a, **kw):
        self.calls.append((a, kw))
        return Reply(self.data)


class OptionsTests(unittest.IsolatedAsyncioTestCase):
    def test_all_three_presets_and_size_options(self):
        self.assertEqual(m.normalize_model("Flare"), "gpt-image-2.5-flare")
        self.assertEqual(m.normalize_model("gpt2"), "gpt-image-2")
        self.assertEqual(m.normalize_size("9:16"), "864x1536")
        self.assertEqual(m.normalize_size("960×1280"), "960x1280")
        self.assertEqual(len(m.PRESET_MODELS), 3)

    def test_edit_options_preserve_prompt(self):
        mode, prompt, options = m.parse_image_command(
            "改图 --model flare --size 3:4 --quality high 保留人物\n只改背景"
        )
        self.assertEqual(mode, "edit")
        self.assertEqual(prompt, "保留人物\n只改背景")
        self.assertEqual(options["size"], "960x1280")

    def test_bad_duplicate_or_missing_options_rejected(self):
        for text in [
            "生图 --key abc 猫",
            "生图 --size",
            "生图 --size 1:1 --size 9:16 猫",
            "生图 --model flare",
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.parse_image_command(text)

    def test_cache_changes_with_endpoint_or_key(self):
        a = {"api_base": "https://example.com/v1", "api_key": "test-a"}
        self.assertNotEqual(
            m.model_fingerprint(a), m.model_fingerprint(a | {"api_key": "test-b"})
        )

    async def test_fetch_filters_and_deduplicates_model_candidates(self):
        session = Session(
            {
                "data": [
                    {"id": "gpt-image-2"},
                    {"id": "gpt-image-2"},
                    {"id": "gpt-image-2.5-flare"},
                    {"id": "chat-model"},
                    {"id": "custom-vision", "output_modalities": ["image"]},
                    {"id": "bad\nvalue"},
                ]
            }
        )
        with patch.object(m.aiohttp, "ClientSession", return_value=session):
            models = await m.fetch_image_models(
                {"api_base": "https://example.com/v1", "api_key": "test-a"}
            )
        self.assertEqual(
            models, ["gpt-image-2", "gpt-image-2.5-flare", "custom-vision"]
        )
        self.assertEqual(session.calls[0][0][0], "https://example.com/v1/models")
        self.assertFalse(session.calls[0][1]["allow_redirects"])


if __name__ == "__main__":
    unittest.main()

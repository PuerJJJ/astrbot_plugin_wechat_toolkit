import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

path = Path(__file__).resolve().parent / "downloads.py"
spec = importlib.util.spec_from_file_location("download_test", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_copy_is_original_bytes_and_multi_use(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.mp4"
            source.write_bytes(b"original-quality")
            service = SimpleNamespace(
                register_file=AsyncMock(return_value="random-token")
            )
            links = m.DownloadLinks(Path(d) / "plugin", service)
            result = await links.create(source, "https://example.com", 3600)
            pin = Path(service.register_file.await_args.args[0])
            self.assertEqual(pin.read_bytes(), source.read_bytes())
            self.assertFalse(service.register_file.await_args.kwargs["single_use"])
            self.assertEqual(result["url"], "https://example.com/api/file/random-token")
            source.write_bytes(b"changed-source")
            self.assertEqual(pin.read_bytes(), b"original-quality")
            with patch.object(m.time, "time", return_value=result["expires_at"] + 1):
                await links.cleanup()
            self.assertFalse(pin.exists())
            self.assertTrue(source.exists())

    async def test_bad_url_is_rejected_before_registration(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / "source.mp4"
            source.write_bytes(b"x")
            service = SimpleNamespace(register_file=AsyncMock())
            links = m.DownloadLinks(Path(d) / "plugin", service)
            for url in [
                "http://example.com",
                "https://u:p@example.com",
                "https://example.com?key=secret",
            ]:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    await links.create(source, url)
            service.register_file.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

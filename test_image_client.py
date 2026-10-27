import base64
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

path = Path(__file__).resolve().parent / "image_client.py"
spec = importlib.util.spec_from_file_location("image_client_test", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
CONFIG = {
    "api_base": "https://example.com/v1",
    "api_key": "test-secret",
    "model": "gpt-image-2.5-sunburst",
}


class Response:
    def __init__(self, status=200, data=b""):
        self.status = status
        self.data = data
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def iter_chunked(self, n):
        yield self.data


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def png():
    b = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(b, "PNG")
    return b.getvalue()


class ImageTests(unittest.IsolatedAsyncioTestCase):
    def test_credentials_and_options(self):
        for extra in [
            {"api_base": "http://example.com"},
            {"api_base": "https://u:p@example.com"},
            {"api_key": ""},
            {"size": "721x1280"},
            {"output_format": "jpeg", "background": "transparent"},
            {"model": "gpt-image-2", "quality": "max"},
        ]:
            with self.subTest(extra=extra), self.assertRaises(m.ImageRequestError):
                m.validate_options(CONFIG | extra, "generate")
        self.assertEqual(
            m.validate_options(CONFIG | {"size": "1536x864"}, "generate")[2]["size"],
            "1536x864",
        )

    async def test_generation_endpoint_and_validated_image(self):
        response = Response(
            data=json.dumps(
                {"data": [{"b64_json": base64.b64encode(png()).decode()}]}
            ).encode()
        )
        session = Session(response)
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(m.aiohttp, "ClientSession", return_value=session),
        ):
            out = await m.request_image(CONFIG, "test", "generate", [], Path(d))
            self.assertEqual(out.read_bytes(), png())
        self.assertEqual(
            session.calls[0][0], "https://example.com/v1/images/generations"
        )
        self.assertFalse(session.calls[0][1]["allow_redirects"])

    async def test_edit_uses_multipart_and_preserves_source(self):
        response = Response(
            data=json.dumps(
                {"data": [{"b64_json": base64.b64encode(png()).decode()}]}
            ).encode()
        )
        session = Session(response)
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(m.aiohttp, "ClientSession", return_value=session),
        ):
            source = Path(d) / "source.png"
            source.write_bytes(png())
            out = await m.request_image(CONFIG, "edit", "edit", [source], Path(d))
            self.assertNotEqual(out, source)
            self.assertEqual(source.read_bytes(), png())
        self.assertTrue(session.calls[0][0].endswith("/images/edits"))
        self.assertIn("data", session.calls[0][1])

    async def test_error_never_retries_or_leaks_body(self):
        session = Session(Response(401, b"test-secret"))
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(m.aiohttp, "ClientSession", return_value=session),
            self.assertRaises(m.ImageRequestError) as error,
        ):
            await m.request_image(CONFIG, "test", "generate", [], Path(d))
        self.assertEqual(len(session.calls), 1)
        self.assertNotIn("test-secret", str(error.exception))
        self.assertIn("401", str(error.exception))

    def test_rejects_non_image(self):
        with self.assertRaises(m.ImageRequestError):
            m.validate_image(b"<html>bad</html>")

    async def test_three_retries_then_success(self):
        notices = AsyncMock()
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(
                m,
                "request_image",
                new=AsyncMock(
                    side_effect=[m.ImageRequestError("503", True)] * 3
                    + [Path(d) / "ok.png"]
                ),
            ) as post,
            patch.object(m.asyncio, "sleep", new=AsyncMock()),
        ):
            await m.request_with_retries(
                CONFIG, "cat", "generate", [], Path(d), notices
            )
        self.assertEqual(post.await_count, 4)
        self.assertTrue(
            any("第 3/3 次重试" in x.args[0] for x in notices.await_args_list)
        )

    async def test_permanent_error_does_not_retry(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(
                m,
                "request_image",
                new=AsyncMock(side_effect=m.ImageRequestError("401")),
            ) as post,
            self.assertRaises(m.ImageRequestError),
        ):
            await m.request_with_retries(
                CONFIG, "cat", "generate", [], Path(d), AsyncMock()
            )
        self.assertEqual(post.await_count, 1)

    async def test_retry_limit_and_edit_references(self):
        refs = [Path("reference.png")]
        notices = AsyncMock()
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(
                m,
                "request_image",
                new=AsyncMock(side_effect=m.ImageRequestError("timeout", True)),
            ) as post,
            patch.object(m.asyncio, "sleep", new=AsyncMock()),
            self.assertRaises(m.ImageRequestError),
        ):
            await m.request_with_retries(CONFIG, "edit", "edit", refs, Path(d), notices)
        self.assertEqual(post.await_count, 4)
        self.assertTrue(
            all(x.args[2] == "edit" and x.args[3] is refs for x in post.await_args_list)
        )
        self.assertIn("重试上限", notices.await_args_list[-1].args[0])

    async def test_cancellation_never_retries(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(
                m,
                "request_image",
                new=AsyncMock(side_effect=m.asyncio.CancelledError()),
            ) as post,
            self.assertRaises(m.asyncio.CancelledError),
        ):
            await m.request_with_retries(
                CONFIG, "cat", "generate", [], Path(d), AsyncMock()
            )
        self.assertEqual(post.await_count, 1)


if __name__ == "__main__":
    unittest.main()

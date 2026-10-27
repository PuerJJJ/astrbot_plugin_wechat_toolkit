"""Bounded OpenAI-compatible image generation; one POST, no automatic retries."""

import base64
import binascii
import io
import json
import re
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from PIL import Image as PILImage

MAX_REFERENCE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 48 * 1024 * 1024


class ImageRequestError(Exception):
    """A sanitized error safe to display in chat."""


def validate_options(config: dict, mode: str):
    base = str(config.get("api_base", "")).strip().rstrip("/")
    parts = urlsplit(base)
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ImageRequestError("生图API地址必须是无用户名、查询参数的HTTPS基础地址。")
    key = str(config.get("api_key", "")).strip()
    if not key:
        raise ImageRequestError("请先在插件配置里填写生图API Key。")
    model = str(config.get("model", "gpt-image-2.5-sunburst")).strip()
    quality = str(config.get("quality", "low"))
    size = str(
        config.get(
            "edit_size" if mode == "edit" else "size",
            "auto" if mode == "edit" else "1024x1024",
        )
    )
    fmt = str(config.get("output_format", "png"))
    background = str(config.get("background", "auto"))
    if (
        not model
        or quality not in {"auto", "low", "medium", "high", "xhigh", "max"}
        or fmt not in {"png", "jpeg", "webp"}
        or background not in {"auto", "opaque", "transparent"}
    ):
        raise ImageRequestError("生图模型、质量、格式或背景配置无效。")
    if mode not in {"generate", "edit"}:
        raise ImageRequestError("模式只能是generate或edit。")
    if size != "auto":
        match = re.fullmatch(r"(\d{3,4})x(\d{3,4})", size)
        if not match:
            raise ImageRequestError("尺寸格式应为1024x1024或auto。")
        w, h = map(int, match.groups())
        if (
            w % 16
            or h % 16
            or max(w, h) > 3840
            or not 655360 <= w * h <= 8294400
            or not 1 / 3 <= w / h <= 3
        ):
            raise ImageRequestError(
                "尺寸需为16的倍数、每边不超过3840、比例1:3～3:1、像素655360～8294400。"
            )
    modern = model.startswith(("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"))
    if quality in {"xhigh", "max"} and not modern:
        raise ImageRequestError("xhigh/max质量只适用于GPT Image 2.5。")
    if background == "transparent" and (fmt == "jpeg" or not modern):
        raise ImageRequestError("透明背景需GPT Image 2.5及PNG/WebP格式。")
    payload = {
        "model": model,
        "quality": quality,
        "size": size,
        "output_format": fmt,
        "background": background,
        "n": 1,
    }
    if config.get("send_response_format", False):
        payload["response_format"] = "b64_json"
    return base, key, payload


def validate_image(data: bytes):
    if not data or len(data) > MAX_RESPONSE_BYTES:
        raise ImageRequestError("图片为空或超过大小限制。")
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in {"PNG", "JPEG", "WEBP"} or im.width * im.height > 20_000_000:
                raise ImageRequestError("只接受PNG/JPEG/WebP且不超过2000万像素。")
            im.verify()
    except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise ImageRequestError("返回的图片文件无效。") from exc
    return {
        "PNG": ("png", "image/png"),
        "JPEG": ("jpg", "image/jpeg"),
        "WEBP": ("webp", "image/webp"),
    }[fmt]


async def request_image(
    config: dict, prompt: str, mode: str, references: list[Path], output_dir: Path
) -> Path:
    """Make exactly one generation/edit POST and save only a validated image."""
    base, key, payload = validate_options(config, mode)
    prompt = prompt.strip()
    if not prompt or len(prompt) > 8000:
        raise ImageRequestError("提示词须为1～8000字。")
    payload["prompt"] = prompt
    if mode == "edit" and not 1 <= len(references) <= 4:
        raise ImageRequestError("改图需要1～4张参考图，或上一张生成的图片。")
    if mode == "generate" and references:
        raise ImageRequestError("文生图模式不接收参考图。")
    request_kwargs = {}
    if mode == "edit":
        form = aiohttp.FormData()
        for k, v in payload.items():
            form.add_field(k, str(v))
        for path in references:
            if not path.is_file() or path.stat().st_size > MAX_REFERENCE_BYTES:
                raise ImageRequestError("参考图不存在或超过10MB。")
            data = path.read_bytes()
            ext, mime = validate_image(data)
            form.add_field(
                "image[]", data, filename="reference." + ext, content_type=mime
            )
        request_kwargs["data"] = form
    else:
        request_kwargs["json"] = payload
    timeout = aiohttp.ClientTimeout(
        total=max(180, min(600, int(config.get("timeout", 420))))
    )
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(
                base + ("/images/edits" if mode == "edit" else "/images/generations"),
                headers={"Authorization": "Bearer " + key},
                allow_redirects=False,
                **request_kwargs,
            ) as response,
        ):
            if response.status != 200:
                raise ImageRequestError(
                    f"生图服务返回HTTP {response.status}。本次不自动重试，请检查配置或额度。"
                )
            data = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ImageRequestError("生图响应超过48MB限制。")
        body = json.loads(data)
        encoded = body["data"][0].get("b64_json")
        if not isinstance(encoded, str):
            raise ImageRequestError(
                "接口未返回b64_json图片；请使用支持Base64响应的生图接口。"
            )
        raw = base64.b64decode(encoded, validate=True)
        ext, _ = validate_image(raw)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / (uuid.uuid4().hex + "." + ext)
        output.write_bytes(raw)
        return output
    except ImageRequestError:
        raise
    except (TimeoutError, aiohttp.ClientError) as exc:
        raise ImageRequestError(
            "生图请求超时或网络异常，服务端可能仍在处理；本次不自动重试。"
        ) from exc
    except (ValueError, KeyError, IndexError, TypeError, binascii.Error) as exc:
        raise ImageRequestError("生图响应格式不正确；本次不自动重试。") from exc

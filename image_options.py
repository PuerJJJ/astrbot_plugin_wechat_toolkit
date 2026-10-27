"""Provider-neutral image model discovery and chat command options."""

import hashlib
import re
from urllib.parse import urlsplit

import aiohttp

PRESET_MODELS = ("gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare")
MODEL_ALIASES = {
    "sunburst": PRESET_MODELS[1],
    "flare": PRESET_MODELS[2],
    "gpt2": PRESET_MODELS[0],
}
SIZES = {
    "1:1": "1024x1024",
    "16:9": "1536x864",
    "9:16": "864x1536",
    "3:4": "960x1280",
    "4:3": "1280x960",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
    "2:1": "1536x768",
    "1:2": "768x1536",
    "方图": "1024x1024",
    "横图": "1536x1024",
    "竖图": "1024x1536",
}


def connection(config):
    base = str(config.get("api_base", "")).strip().rstrip("/")
    url = urlsplit(base)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("请填写有效的HTTPS API基础地址，不含用户名或查询参数。")
    key = str(config.get("api_key", "")).strip()
    if not key:
        raise ValueError("请先在插件配置中填写生图API Key。")
    return base, key


def normalize_model(value):
    value = str(value).strip()
    value = MODEL_ALIASES.get(value.lower(), value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,149}", value):
        raise ValueError("模型ID格式无效。")
    return value


def normalize_size(value):
    value = str(value).strip().lower().replace("×", "x")
    return SIZES.get(value, value)


def parse_image_command(text):
    """Consume only leading options; preserve the remaining prompt verbatim."""
    text = text.strip().lstrip("/")
    mode = "edit" if text.startswith(("改图", "该图")) else "generate"
    prompt = text[2:].lstrip(" ：:")
    options = {}
    while prompt.startswith("--"):
        m = re.match(
            r"""--(model|size|quality)\s+(?:"([^"]+)"|'([^']+)'|(\S+))(?:\s+|$)""",
            prompt,
        )
        if not m:
            raise ValueError(
                "参数格式：--model flare --size 9:16 --quality high，后接提示词。"
            )
        name = m[1]
        if name in options:
            raise ValueError("同一参数不能重复。")
        value = next(v for v in m.groups()[1:] if v is not None)
        options[name] = (
            normalize_model(value)
            if name == "model"
            else normalize_size(value)
            if name == "size"
            else value.lower()
        )
        prompt = prompt[m.end() :]
    if not prompt.strip():
        raise ValueError("请填写生图或改图提示词。")
    return mode, prompt, options


def model_fingerprint(config):
    base, key = connection(config)
    return hashlib.sha256((base + "\0" + key).encode()).hexdigest()


async def fetch_image_models(config):
    """Fetch model candidates once. Listing is not a generation capability test."""
    base, key = connection(config)
    async with (
        aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session,
        session.get(
            base + "/models",
            headers={"Authorization": "Bearer " + key},
            allow_redirects=False,
        ) as response,
    ):
        if response.status != 200:
            raise ValueError(f"模型列表接口返回HTTP {response.status}。")
        raw = bytearray()
        async for chunk in response.content.iter_chunked(32768):
            raw.extend(chunk)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError("模型列表响应过大。")
    import json

    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("模型列表格式不兼容。")
    models = []
    for item in data["data"]:
        if not isinstance(item, dict):
            continue
        name = item.get("id", "")
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,149}", name
        ):
            continue
        candidate = bool(
            re.search(r"gpt-image|dall-e|flux|imagen|seedream|ideogram", name, re.I)
        )
        modalities = item.get("output_modalities", [])
        candidate = candidate or (
            isinstance(modalities, list) and "image" in modalities
        )
        if candidate and name not in models:
            models.append(name)
        if len(models) >= 100:
            break
    return models

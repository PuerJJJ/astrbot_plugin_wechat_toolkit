"""Read validated public tariffs from the official Ark documentation."""

import asyncio
import hashlib
import json
import re
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp

UTC = timezone.utc

SOURCE = "https://ark.volcengine.com/region:cn-beijing/docs/model-pricing.md"
MODEL_ROWS = {
    "deepseek-v4-1-flash": "deepseek-v4-1-flash",
    "deepseek-v4-1-flash-economy": "deepseek-v4-1-flash经济版",
}


def parse_prices(markdown: str, fetched_at: str | None = None) -> dict:
    """Reject unsupported document shapes rather than guess a price or schedule."""
    section = markdown.split("## 在线推理（常规）", 1)[1].split(
        "## 在线推理（低延迟）", 1
    )[0]
    if "元/百万token" not in section or "北京时间周一至周五" not in section:
        raise ValueError("Unsupported price unit or weekday schedule")
    periods = re.findall(
        r"(\d{2}):(\d{2})[–—-](\d{2}):(\d{2})", section.split("|模型名称", 1)[0]
    )
    if not periods:
        raise ValueError("Missing peak schedule")
    windows = [
        [int(h) * 60 + int(m), int(eh) * 60 + int(em)] for h, m, eh, em in periods
    ]
    if any(not 0 <= a < b <= 1440 for a, b in windows):
        raise ValueError("Invalid peak schedule")
    rows = {}
    current = ""
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cols = [v.strip() for v in line.rstrip(" \\").strip("|").split("|")]
        if len(cols) != 8 or cols[0] == "^^":
            continue
        if cols[0]:
            current = cols[0]
        if current not in MODEL_ROWS.values() or cols[1] not in (
            "空闲时段",
            "高峰时段",
        ):
            continue
        rates = []
        for index in (2, 5, 7):
            value = Decimal(cols[index])
            if not value.is_finite() or not 0 <= value <= 100000:
                raise ValueError("Invalid tariff")
            rates.append(str(value))
        period = "peak" if cols[1] == "高峰时段" else "offpeak"
        if period in rows.setdefault(current, {}):
            raise ValueError("Duplicate tariff row")
        rows[current][period] = rates
    models = {model: rows[row] for model, row in MODEL_ROWS.items() if row in rows}
    if any(set(v) != {"peak", "offpeak"} for v in models.values()) or len(
        models
    ) != len(MODEL_ROWS):
        raise ValueError("Incomplete supported model tariffs")
    return {
        "schema": 1,
        "source": SOURCE,
        "fetched_at": fetched_at or datetime.now(UTC).isoformat(),
        "sha256": hashlib.sha256(markdown.encode()).hexdigest(),
        "currency": "CNY",
        "unit_tokens": 1000000,
        "peak_weekdays": [0, 1, 2, 3, 4],
        "peak_windows": windows,
        "models": models,
    }


def resolve_rates(snapshot: dict, model: str, ended: datetime):
    family = re.sub(r"-\d{6}$", "", model)
    tariff = snapshot.get("models", {}).get(family)
    if not tariff:
        return None
    minutes = ended.hour * 60 + ended.minute
    peak = ended.weekday() in snapshot["peak_weekdays"] and any(
        a <= minutes < b for a, b in snapshot["peak_windows"]
    )
    try:
        rates = [Decimal(v) for v in tariff["peak" if peak else "offpeak"]]
    except (KeyError, InvalidOperation, TypeError):
        return None
    return (
        rates
        if len(rates) == 3 and all(v.is_finite() and v >= 0 for v in rates)
        else None
    )


class PriceStore:
    def __init__(self, root: Path):
        self.path = root / "ark-public-prices.json"
        self.snapshot = None
        self.last_error = ""
        self.lock = asyncio.Lock()
        self.last_attempt = 0.0
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                if (
                    data.get("schema") == 1
                    and data.get("source") == SOURCE
                    and data.get("models")
                ):
                    datetime.fromisoformat(data["fetched_at"])
                    if (
                        data.get("currency") != "CNY"
                        or data.get("unit_tokens") != 1000000
                    ):
                        raise ValueError("Invalid cache currency")
                    if not all(
                        resolve_rates(
                            data, model, datetime.now(ZoneInfo("Asia/Shanghai"))
                        )
                        is not None
                        for model in MODEL_ROWS
                    ):
                        raise ValueError("Incomplete cached rates")
                    self.snapshot = data
            except (OSError, ValueError, KeyError, TypeError, InvalidOperation):
                self.last_error = "invalid_cache"

    async def refresh(self, force=False):
        now = datetime.now(UTC).timestamp()
        if not force and now - self.last_attempt < 300:
            return self.snapshot
        async with self.lock:
            if not force and now - self.last_attempt < 300:
                return self.snapshot
            self.last_attempt = now
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with (
                    aiohttp.ClientSession(timeout=timeout) as session,
                    session.get(SOURCE, allow_redirects=False) as response,
                ):
                    if response.status != 200:
                        raise ValueError("http_" + str(response.status))
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(32768):
                        data.extend(chunk)
                        if len(data) > 1024 * 1024:
                            raise ValueError("document_too_large")
                snapshot = parse_prices(bytes(data).decode("utf-8"))
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
                tmp.replace(self.path)
                self.snapshot = snapshot
                self.last_error = ""
            except (
                TimeoutError,
                aiohttp.ClientError,
                OSError,
                ValueError,
                IndexError,
                InvalidOperation,
                KeyError,
            ) as exc:
                self.last_error = type(exc).__name__
            return self.snapshot

    async def get(self):
        if self.snapshot is None:
            return await self.refresh()
        return self.snapshot

    async def run(self):
        while True:
            await self.refresh()
            await asyncio.sleep(86400)

    def label(self):
        if self.snapshot is None:
            return "官方资费未获取，金额未知"
        stamp = (
            datetime.fromisoformat(self.snapshot["fetched_at"])
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .strftime("%Y-%m-%d %H:%M")
        )
        stale = (
            datetime.now(UTC).timestamp()
            - datetime.fromisoformat(self.snapshot["fetched_at"]).timestamp()
            > 86400
        )
        return f"官方公开价同步：{stamp}" + (
            "（刷新失败，使用旧缓存）" if self.last_error or stale else ""
        )

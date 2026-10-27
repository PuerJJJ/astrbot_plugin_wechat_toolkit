"""Read-only AstrBot run accounting; prices are estimates, not account bills."""

import sqlite3
from datetime import datetime
from decimal import Decimal
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .pricing import resolve_rates

TZ = ZoneInfo("Asia/Shanghai")
MODEL = "deepseek-v4-1-flash-economy-261017"


def estimated_cost(model, uncached, cached, output, ended, snapshot=None):
    """Estimate with the retrieved public price, never a baked-in fallback rate."""
    if snapshot is None or min(uncached, cached, output) < 0:
        return None
    rates = resolve_rates(snapshot, model, datetime.fromtimestamp(ended, TZ))
    if rates is None:
        return None
    return sum(
        Decimal(count) * rate
        for count, rate in zip((uncached, cached, output), rates, strict=True)
    ) / Decimal(1000000)


def usage_report(
    database,
    now=None,
    snapshot=None,
    price_label="",
    eligible_provider_ids=(),
    platform_id=None,
):
    """Aggregate completed runs without modifying AstrBot's database."""
    if not isinstance(platform_id, str) or not platform_id or ":" in platform_id:
        raise ValueError("A valid platform_id is required for isolated usage queries")
    now = now or datetime.now(TZ)
    start = (
        now.astimezone(TZ)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=3) as con:
        rows = con.execute(
            "SELECT provider_model, token_input_other, token_input_cached, token_output, end_time, provider_id FROM provider_stats WHERE status = ? AND end_time <= ? AND substr(umo, 1, instr(umo, ':') - 1) = ?",
            ("completed", now.timestamp(), platform_id),
        ).fetchall()
    totals = []
    for period in ([r for r in rows if r[4] >= start], rows):
        costs = [
            estimated_cost(*r[:5], snapshot=snapshot)
            if r[5] in eligible_provider_ids
            else None
            for r in period
        ]
        cost = (
            "部分模型费率未知"
            if any(c is None for c in costs)
            else f"¥{sum(costs, Decimal(0)):.4f}"
        )
        totals.append((len(period), sum(sum(r[1:4]) for r in period), cost))
    today, alltime = totals
    return (
        "📊 用量统计 · 当前机器人\n"
        "━━━━━━━━━━━━\n"
        f"📅 今日 · {now.astimezone(TZ):%m月%d日}\n"
        f"  调用　{today[0]:,} 次\n"
        f"  Token　{today[1]:,}\n"
        f"  消费　{today[2]}（估算）\n\n"
        "📚 累计\n"
        f"  调用　{alltime[0]:,} 次\n"
        f"  Token　{alltime[1]:,}\n"
        f"  消费　{alltime[2]}（估算）\n"
        "━━━━━━━━━━━━\n"
    )


def eligible_ark_providers(config):
    sources = {item["id"]: item for item in config.get("provider_sources", [])}
    eligible = set()
    for item in config.get("provider", []):
        source = sources.get(item.get("provider_source_id"), item)
        tier = (item.get("custom_extra_body") or {}).get(
            "service_tier",
            (source.get("custom_extra_body") or {}).get("service_tier", "default"),
        )
        if urlsplit(
            source.get("api_base", "")
        ).hostname == "ark.cn-beijing.volces.com" and tier in (None, "default"):
            eligible.add(item["id"])
    return eligible


def turn_usage_text(records, snapshot, eligible_ids, attempted=False):
    unknown = any(r.get("missing") for r in records) or (attempted and not records)
    tokens = sum(
        r.get("input_other", 0) + r.get("input_cached", 0) + r.get("output", 0)
        for r in records
    )
    costs = [
        estimated_cost(
            r["model"],
            r.get("input_other", 0),
            r.get("input_cached", 0),
            r.get("output", 0),
            r["ended"],
            snapshot,
        )
        if r.get("provider_id") in eligible_ids and not r.get("missing")
        else None
        for r in records
    ]
    if unknown or any(x is None for x in costs):
        amount = "费用未知"
    else:
        cost = sum(costs, Decimal(0))
        amount = (
            f"约¥{cost:.4f}" if cost >= Decimal("0.0001") or not cost else "<¥0.0001"
        )
    token_text = (
        (f"已知{tokens:,} Token" if tokens else "Token未知")
        if unknown
        else f"{tokens:,} Token"
    )
    return f"本次对话模型：{token_text} · {amount}"

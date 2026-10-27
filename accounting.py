"""Read-only AstrBot run accounting; prices are estimates, not account bills."""

import sqlite3
from datetime import datetime
from decimal import Decimal
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
    database, now=None, snapshot=None, price_label="", eligible_provider_ids=()
):
    """Aggregate completed runs without modifying AstrBot's database."""
    now = now or datetime.now(TZ)
    start = (
        now.astimezone(TZ)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=3) as con:
        rows = con.execute(
            "SELECT provider_model, token_input_other, token_input_cached, token_output, end_time, provider_id FROM provider_stats WHERE status = ? AND end_time <= ?",
            ("completed", now.timestamp()),
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
        "📊 AstrBot 用量统计\n"
        "━━━━━━━━━━━━\n"
        f"📅 今日 · {now.astimezone(TZ):%m月%d日}\n"
        f"  调用　{today[0]:,} 次\n"
        f"  Token　{today[1]:,}\n"
        f"  消费　{today[2]}\n\n"
        "📚 累计\n"
        f"  调用　{alltime[0]:,} 次\n"
        f"  Token　{alltime[1]:,}\n"
        f"  消费　{alltime[2]}\n"
        "━━━━━━━━━━━━\n"
        "北京时间｜本机已完成对话任务\n"
        f"{price_label}\n"
        "调用按任务计；Token含缓存和推理。\n"
        "费用按当前公开价估算，非历史实扣；不含搜索、语音、记忆总结等及优惠抵扣，非账户余额。"
    )

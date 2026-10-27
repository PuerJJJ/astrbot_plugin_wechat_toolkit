"""Read-only AstrBot run accounting; prices are estimates, not account bills."""

import sqlite3
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")
MODEL = "deepseek-v4-1-flash-economy-261017"


def estimated_cost(model, uncached, cached, output, ended):
    """Estimate token fees using the verified 2026-10-27 Ark public tariff."""
    if model != MODEL or min(uncached, cached, output) < 0:
        return None
    when = datetime.fromtimestamp(ended, TZ)
    peak = when.weekday() < 5 and (9 <= when.hour < 12 or 14 <= when.hour < 18)
    factor = Decimal(2 if peak else 1)
    return (
        (
            Decimal(uncached) * Decimal("0.80")
            + Decimal(cached) * Decimal("0.016")
            + Decimal(output) * Decimal("3.20")
        )
        * factor
        / Decimal(1000000)
    )


def usage_report(database, now=None):
    """Aggregate completed runs without modifying AstrBot's database."""
    now = now or datetime.now(TZ)
    start = (
        now.astimezone(TZ)
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .timestamp()
    )
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=3) as con:
        rows = con.execute(
            "SELECT provider_model, token_input_other, token_input_cached, token_output, end_time FROM provider_stats WHERE status = ? AND end_time <= ?",
            ("completed", now.timestamp()),
        ).fetchall()
    totals = []
    for period in ([r for r in rows if r[4] >= start], rows):
        costs = [estimated_cost(*r) for r in period]
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
        "调用按任务计；Token含缓存和推理。\n"
        "费用为人民币估算，不含搜索、语音等及优惠抵扣；非账户余额。"
    )

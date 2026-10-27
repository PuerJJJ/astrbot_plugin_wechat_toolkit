import importlib.util
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

path = Path(__file__).resolve().parent / "accounting.py"
pkg = types.ModuleType("toolkit_under_test")
pkg.__path__ = [str(path.parent)]
sys.modules["toolkit_under_test"] = pkg
spec = importlib.util.spec_from_file_location("toolkit_under_test.accounting", path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


SNAPSHOT = {
    "models": {
        "deepseek-v4-1-flash-economy": {
            "offpeak": ["0.80", "0.016", "3.20"],
            "peak": ["1.60", "0.032", "6.40"],
        }
    },
    "peak_weekdays": [0, 1, 2, 3, 4],
    "peak_windows": [[540, 720], [840, 1080]],
}


class AccountingTests(unittest.TestCase):
    def test_peak_boundaries(self):
        for day, hour, expected in [
            (27, 8, "4.016"),
            (27, 9, "8.032"),
            (27, 12, "4.016"),
            (27, 14, "8.032"),
            (27, 18, "4.016"),
            (31, 10, "4.016"),
        ]:
            t = datetime(2026, 10, day, hour, tzinfo=a.TZ).timestamp()
            self.assertEqual(
                a.estimated_cost(a.MODEL, 1000000, 1000000, 1000000, t, SNAPSHOT),
                Decimal(expected),
            )

    def test_unknown_model_is_not_free(self):
        self.assertIsNone(a.estimated_cost("other", 1, 2, 3, 0))
        self.assertIsNone(a.estimated_cost(a.MODEL, -1, 2, 3, 0))

    def test_timezone_failed_runs_and_cached_tokens(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.db"
            with sqlite3.connect(p) as con:
                con.execute(
                    "CREATE TABLE provider_stats (provider_model TEXT,token_input_other INT,token_input_cached INT,token_output INT,end_time REAL,status TEXT,provider_id TEXT,umo TEXT)"
                )
                for day, hour, status in [
                    (26, 23, "completed"),
                    (27, 0, "completed"),
                    (27, 9, "failed"),
                    (28, 0, "completed"),
                ]:
                    con.execute(
                        "INSERT INTO provider_stats VALUES (?,?,?,?,?,?,?,?)",
                        (
                            a.MODEL,
                            10,
                            20,
                            30,
                            datetime(2026, 10, day, hour, tzinfo=a.TZ).timestamp(),
                            status,
                            "ark",
                            "bot_a:FriendMessage:user",
                        ),
                    )
            report = a.usage_report(
                p,
                datetime(2026, 10, 27, 11, tzinfo=a.TZ),
                snapshot=SNAPSHOT,
                eligible_provider_ids=("ark",),
                platform_id="bot_a",
            )
            self.assertIn("调用　1 次", report)
            self.assertIn("调用　2 次", report)
            self.assertIn("Token　60", report)
            self.assertIn("Token　120", report)
            self.assertIn("估算", report)
            self.assertNotIn("非账户余额", report)
            self.assertNotIn("本机已完成对话", report)

    def test_bot_scope_is_exact_and_no_unscoped_query(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "scope.db"
            with sqlite3.connect(p) as con:
                con.execute(
                    "CREATE TABLE provider_stats (provider_model TEXT,token_input_other INT,token_input_cached INT,token_output INT,end_time REAL,status TEXT,provider_id TEXT,umo TEXT)"
                )
                for bot, tokens in [("bot_a", 100), ("bot_ab", 200), ("bot%_", 300)]:
                    con.execute(
                        "INSERT INTO provider_stats VALUES (?,?,?,?,?,?,?,?)",
                        (
                            a.MODEL,
                            tokens,
                            0,
                            0,
                            1,
                            "completed",
                            "ark",
                            bot + ":FriendMessage:user",
                        ),
                    )
            for bot, tokens in [("bot_a", 100), ("bot_ab", 200), ("bot%_", 300)]:
                report = a.usage_report(
                    p,
                    platform_id=bot,
                    snapshot=SNAPSHOT,
                    eligible_provider_ids=("ark",),
                )
                self.assertIn(f"Token　{tokens}", report)
                self.assertIn("调用　1 次", report)
            with self.assertRaises(ValueError):
                a.usage_report(p)

    def test_turn_receipt_aggregates_all_calls_and_unknowns(self):
        when = datetime(2026, 10, 27, 10, tzinfo=a.TZ).timestamp()
        records = [
            {
                "provider_id": "ark",
                "model": a.MODEL,
                "input_other": 100,
                "input_cached": 200,
                "output": 30,
                "ended": when,
            },
            {
                "provider_id": "ark",
                "model": a.MODEL,
                "input_other": 50,
                "input_cached": 300,
                "output": 20,
                "ended": when,
            },
        ]
        result = a.turn_usage_text(records, SNAPSHOT, {"ark"})
        self.assertIn("700 Token", result)
        self.assertIn("约¥", result)
        self.assertIn("费用未知", a.turn_usage_text(records, None, {"ark"}))
        self.assertIn(
            "Token未知", a.turn_usage_text([], SNAPSHOT, {"ark"}, attempted=True)
        )
        self.assertIn("0 Token", a.turn_usage_text([], SNAPSHOT, {"ark"}))

    def test_missing_database_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "absent.db"
            with self.assertRaises(sqlite3.OperationalError):
                a.usage_report(p, platform_id="bot_a")
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()

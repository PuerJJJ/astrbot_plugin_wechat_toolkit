import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

path = Path(__file__).resolve().parent / "accounting.py"
spec = importlib.util.spec_from_file_location("accounting", path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


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
                a.estimated_cost(a.MODEL, 1000000, 1000000, 1000000, t),
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
                    "CREATE TABLE provider_stats (provider_model TEXT,token_input_other INT,token_input_cached INT,token_output INT,end_time REAL,status TEXT)"
                )
                for day, hour, status in [
                    (26, 23, "completed"),
                    (27, 0, "completed"),
                    (27, 9, "failed"),
                    (28, 0, "completed"),
                ]:
                    con.execute(
                        "INSERT INTO provider_stats VALUES (?,?,?,?,?,?)",
                        (
                            a.MODEL,
                            10,
                            20,
                            30,
                            datetime(2026, 10, day, hour, tzinfo=a.TZ).timestamp(),
                            status,
                        ),
                    )
            report = a.usage_report(p, datetime(2026, 10, 27, 11, tzinfo=a.TZ))
            self.assertIn("调用　1 次", report)
            self.assertIn("调用　2 次", report)
            self.assertIn("Token　60", report)
            self.assertIn("Token　120", report)
            self.assertIn("估算", report)
            self.assertIn("非账户余额", report)

    def test_missing_database_not_created(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "absent.db"
            with self.assertRaises(sqlite3.OperationalError):
                a.usage_report(p)
            self.assertFalse(p.exists())


if __name__ == "__main__":
    unittest.main()

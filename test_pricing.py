import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import aiohttp

path = Path(__file__).resolve().parent / "pricing.py"
spec = importlib.util.spec_from_file_location("toolkit_pricing_test", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
DOCUMENT = """## 在线推理（常规）
北京时间周一至周五 09:00–12:00、14:00–18:00
单位：元/百万token
|模型名称|条件|输入(非音频)|输入(音频)|缓存存储|缓存命中(非音频)|缓存命中(音频)|输出|
|deepseek-v4-1-flash|空闲时段|1|-|0.017|0.02|-|4|
|^^| | | | | | | |
| |高峰时段|2|-|0.017|0.04|-|8|
|deepseek-v4-1-flash经济版|空闲时段|0.8|-|0.017|0.016|-|3.2|
| |高峰时段|1.6|-|0.017|0.032|-|6.4|
## 在线推理（低延迟）
"""


class PricingTests(unittest.IsolatedAsyncioTestCase):
    def test_current_rows_and_continuation(self):
        s = m.parse_prices(DOCUMENT)
        self.assertEqual(
            s["models"]["deepseek-v4-1-flash-economy"]["peak"], ["1.6", "0.032", "6.4"]
        )

    def test_changed_tariff_is_used_without_code_change(self):
        s = m.parse_prices(DOCUMENT.replace("|1.6|", "|1.9|"))
        self.assertEqual(s["models"]["deepseek-v4-1-flash-economy"]["peak"][0], "1.9")

    def test_incomplete_or_ambiguous_document_is_rejected(self):
        for doc in [
            DOCUMENT.replace("| |高峰时段|1.6|-|0.017|0.032|-|6.4|", ""),
            DOCUMENT.replace("元/百万token", "美元/千token"),
            DOCUMENT.replace("北京时间周一至周五", "每天"),
        ]:
            with self.subTest(doc=doc), self.assertRaises(ValueError):
                m.parse_prices(doc)

    async def test_failed_refresh_preserves_cache(self):
        with tempfile.TemporaryDirectory() as d:
            store = m.PriceStore(Path(d))
            store.snapshot = m.parse_prices(DOCUMENT)
            old = store.snapshot
            with patch.object(
                aiohttp, "ClientSession", side_effect=aiohttp.ClientError("network")
            ):
                await store.refresh(force=True)
            self.assertIs(old, store.snapshot)
            self.assertIn("旧缓存", store.label())

    async def test_missing_price_is_unknown_and_requests_are_throttled(self):
        with tempfile.TemporaryDirectory() as d:
            store = m.PriceStore(Path(d))
            with patch.object(
                aiohttp, "ClientSession", side_effect=aiohttp.ClientError("network")
            ) as create:
                self.assertIsNone(await store.get())
                self.assertIsNone(await store.get())
                self.assertEqual(create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

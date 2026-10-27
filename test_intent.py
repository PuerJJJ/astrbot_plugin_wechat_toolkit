import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parent / "intent.py"
spec = importlib.util.spec_from_file_location("intent", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class IntentTests(unittest.TestCase):
    def test_direct_and_common_natural_queries(self):
        for text in [
            "费用",
            "余额",
            "/用量",
            "用量统计",
            "帮我查一下余额",
            "今天花了多少钱？",
            "看看今日用量",
            "请帮我查一下今天花了多少钱",
            "这个机器人今天用了多少Token",
            "还剩多少额度",
            "余额是多少",
            "今天调用了几次",
            "一共消耗了多少token",
            "查看一下我今天的账单",
        ]:
            with self.subTest(text=text):
                self.assertTrue(m.is_usage_query(text))

    def test_unrelated_or_instructional_messages_are_not_intercepted(self):
        for text in [
            "银行卡余额",
            "查一下话费余额",
            "帮我查支付宝余额",
            "余额是什么意思",
            "今天花了多少钱这个问题怎么写代码实现",
            "不要查余额",
            "什么是Token",
            "写一个查询余额的函数",
            "消费税怎么计算",
            "你好",
            "帮我设计一个用量统计页面",
            "帮我翻译：今天花了多少钱",
            "a" * 100 + "余额",
        ]:
            with self.subTest(text=text):
                self.assertFalse(m.is_usage_query(text))


if __name__ == "__main__":
    unittest.main()

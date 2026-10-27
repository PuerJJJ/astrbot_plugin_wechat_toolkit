"""Conservative shortcuts; open-ended intent recognition belongs to the LLM tool."""

import re


def is_usage_query(text: str) -> bool:
    """Match standalone local-usage requests without capturing unrelated finances."""
    text = re.sub(r"\s+", "", text).lower().strip("/。？！?!，,～~")
    if len(text) > 80 or any(
        word in text
        for word in (
            "银行卡",
            "支付宝",
            "微信钱包",
            "话费",
            "电费",
            "水费",
            "信用卡",
            "流量套餐",
            "是什么意思",
            "什么是",
            "怎么计算",
            "如何计算",
            "不要",
            "不用",
            "别查",
            "写代码",
            "写一个",
            "实现",
            "设计",
            "翻译",
        )
    ):
        return False
    text = re.sub(
        r"^(?:请)?(?:(?:帮我|给我|替我))?(?:查询一下|查询|查看一下|查看|查一下|看一下|查查|看看|统计一下|统计|查|看)?",
        "",
        text,
    )
    text = re.sub(
        r"^(?:astrbot|这个机器人|机器人|这个bot|bot|模型|api|我们|我)(?:的)?", "", text
    )
    text = text.rstrip("呀啊呢吧吗")
    return bool(
        re.fullmatch(
            r"(?:余额|额度|用量|费用|消费|用量统计|消费统计|账单)"
            r"|(?:今天|今日|本日|累计|总共|一共|目前|现在)?(?:的)?(?:用量|费用|消费|账单|token(?:用量)?)"
            r"|(?:今天|今日|本日|累计|总共|一共|目前|现在)?(?:花了|用了|消耗了|消费了|调用了|花|用|消耗|消费|调用)(?:多少(?:钱|次|个token|token|额度)?|几次)"
            r"|(?:还剩|还剩下|剩余|剩下|还有)(?:多少)(?:钱|额度|余额|token)?"
            r"|(?:余额|额度|用量|费用|消费|token(?:用量)?)(?:还)?(?:有多少|是多少|剩多少|还剩多少)"
            r"|(?:今天|今日|累计|总共|一共)(?:调用|请求)(?:了)?(?:多少次|几次)",
            text,
        )
    )

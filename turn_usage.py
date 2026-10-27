"""Meter completed agent calls per event and append one concise usage receipt."""

import functools
import time
from copy import copy

from astrbot.api import logger
from astrbot.api.event import MessageChain
from astrbot.api.message_components import Plain
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.platform.sources.weixin_oc.weixin_oc_event import WeixinOCMessageEvent

from .accounting import eligible_ark_providers, turn_usage_text


def new_state():
    return {
        "records": [],
        "previous": {},
        "seen": set(),
        "root_context": None,
        "attempted": False,
        "done": False,
        "direct": False,
        "added": False,
    }


class TurnUsage:
    def __init__(self, owner):
        self.owner = owner
        self.patches = []

    def text(self, event):
        state = event.get_extra("wechat_turn_usage") or new_state()
        try:
            return turn_usage_text(
                state["records"],
                self.owner.prices.snapshot,
                eligible_ark_providers(self.owner.context.get_config()),
                state["attempted"],
            )
        except Exception as exc:  # noqa: BLE001 - receipt formatting is non-critical
            logger.warning("Usage receipt formatting failed: %s", type(exc).__name__)
            return "本次对话模型：用量暂不可用"

    def install(self):
        meter = self
        original_execute = PipelineScheduler.execute

        @functools.wraps(original_execute)
        async def execute(scheduler, event):
            if event.get_platform_name() != "weixin_oc":
                return await original_execute(scheduler, event)
            event.set_extra("wechat_turn_usage", new_state())
            try:
                return await original_execute(scheduler, event)
            finally:
                state = event.get_extra("wechat_turn_usage")
                if (
                    getattr(event, "_has_send_oper", False)
                    and state
                    and not state["added"]
                ):
                    state["added"] = True
                    try:
                        await event.send(MessageChain([Plain(meter.text(event))]))
                    except Exception as exc:  # noqa: BLE001 - receipt failure must not mask the turn outcome
                        logger.warning(
                            "Usage receipt delivery failed: %s", type(exc).__name__
                        )

        original_step = ToolLoopAgentRunner.step

        @functools.wraps(original_step)
        async def step(runner, *args, **kwargs):
            context = getattr(runner, "run_context", None)
            event = getattr(getattr(context, "context", None), "event", None)
            state = None
            if event is not None and event.get_platform_name() == "weixin_oc":
                state = event.get_extra("wechat_turn_usage")
                if not state or (
                    state["added"] and state["root_context"] != id(context)
                ):
                    state = new_state()
                    event.set_extra("wechat_turn_usage", state)
                if state["root_context"] is None:
                    state["root_context"] = id(context)
                state["attempted"] = True
            iterator = original_step(runner, *args, **kwargs)
            try:
                async for item in iterator:
                    if state is not None and item.type == "agent_stats":
                        stats = runner.stats.to_dict()
                        usage = stats.get("token_usage", {})
                        current = tuple(
                            int(usage.get(k, 0) or 0)
                            for k in ["input_other", "input_cached", "output"]
                        )
                        key = (id(runner), stats.get("start_time", 0))
                        stamp = (key, stats.get("end_time", 0), current)
                        if stamp not in state["seen"]:
                            previous = state["previous"].get(key, (0, 0, 0))
                            delta = tuple(
                                max(0, n - o)
                                for n, o in zip(current, previous, strict=True)
                            )
                            state["previous"][key] = current
                            state["seen"].add(stamp)
                            provider = runner.provider
                            state["records"].append(
                                {
                                    "provider_id": (
                                        getattr(provider, "provider_config", {}) or {}
                                    ).get("id")
                                    or provider.meta().id,
                                    "model": provider.get_model(),
                                    "input_other": delta[0],
                                    "input_cached": delta[1],
                                    "output": delta[2],
                                    "ended": stats.get("end_time") or time.time(),
                                    "missing": not any(delta)
                                    or any(
                                        n < o
                                        for n, o in zip(current, previous, strict=True)
                                    ),
                                }
                            )
                    if (
                        state is not None
                        and item.type == "llm_result"
                        and state["root_context"] == id(context)
                        and runner.done()
                    ):
                        state["done"] = True
                    yield item
            finally:
                await iterator.aclose()

        original_send = WeixinOCMessageEvent.send

        @functools.wraps(original_send)
        async def send(event, message):
            state = event.get_extra("wechat_turn_usage")
            if (
                state
                and not state["added"]
                and (state["done"] or state["direct"])
                and message
                and message.chain
            ):
                message = copy(message)
                message.chain = list(message.chain) + [
                    Plain("\n\n" + meter.text(event))
                ]
                state["added"] = True
            return await original_send(event, message)

        original_stream = WeixinOCMessageEvent.send_streaming

        @functools.wraps(original_stream)
        async def send_streaming(event, generator, use_fallback=False):
            async def decorated():
                async for chain in generator:
                    yield chain
                state = event.get_extra("wechat_turn_usage")
                if state and not state["added"]:
                    state["attempted"] = True
                    state["added"] = True
                    yield MessageChain([Plain("\n\n" + meter.text(event))])

            return await original_stream(event, decorated(), use_fallback)

        for cls, name, replacement in [
            (PipelineScheduler, "execute", execute),
            (ToolLoopAgentRunner, "step", step),
            (WeixinOCMessageEvent, "send", send),
            (WeixinOCMessageEvent, "send_streaming", send_streaming),
        ]:
            self.patches.append((cls, name, getattr(cls, name), replacement))
            setattr(cls, name, replacement)

    def close(self):
        for cls, name, original, replacement in reversed(self.patches):
            if getattr(cls, name) is replacement:
                setattr(cls, name, original)
        self.patches.clear()

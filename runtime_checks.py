"""Run in the AstrBot container; only temporary storage and mocked providers are used."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from data.plugins.astrbot_plugin_wechat_toolkit import main as m


class Event:
    def __init__(self, admin=True):
        self.admin = admin
        self.extra = {
            "wechat_toolkit_timing": {"_start": time.perf_counter(), "model_calls": []}
        }
        self.stopped = False
        self.unified_msg_origin = "test-session"
        self.sent = []

    def get_message_str(self):
        return "用量"

    def get_platform_id(self):
        return "bot_a"

    def get_platform_name(self):
        return "weixin_oc"

    def is_admin(self):
        return self.admin

    def plain_result(self, text):
        return text

    def stop_event(self):
        self.stopped = True

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_messages(self):
        return []

    async def send(self, chain):
        self.sent.append(chain)

    def get_extra(self, key):
        return self.extra.get(key)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pathpatch = patch.object(
            m, "get_astrbot_data_path", return_value=self.tmp.name
        )
        self.pathpatch.start()
        self.plugin = m.WechatToolkit(
            SimpleNamespace(get_provider_by_id=lambda _: None, get_config=dict),
            {
                "tts_provider_id": "mimo_tts",
                "enable_timing": True,
                "large_video_links": True,
            },
        )

        self.plugin.prices.get = AsyncMock(return_value=None)
        self.plugin.prices.run = AsyncMock(return_value=None)

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.pathpatch.stop()
        self.tmp.cleanup()

    async def test_nonadmin_usage_is_denied(self):
        event = Event(False)
        result = [x async for x in self.plugin.usage(event)]
        self.assertIn("仅管理员", result[0])
        self.assertTrue(event.stopped)

    async def test_missing_database_does_not_create_one(self):
        event = Event()
        result = [x async for x in self.plugin.usage(event)]
        self.assertIn("暂时无法", result[0])
        self.assertFalse((Path(self.tmp.name) / "data_v4.db").exists())

    async def test_voice_limits_and_missing_provider(self):
        self.assertIn("仅限", await self.plugin.send_voice_file(Event(False), "hello"))
        self.assertIn("1～600", await self.plugin.send_voice_file(Event(), "x" * 601))
        self.assertIn("未配置", await self.plugin.send_voice_file(Event(), "你好"))

    async def test_usage_tool_checks_permissions_before_reading(self):
        with patch.object(
            m, "usage_report", side_effect=AssertionError("must not read")
        ):
            self.assertIn("仅管理员", await self.plugin.query_usage(Event(False)))

    async def test_disabled_timing_leaves_methods_unchanged(self):
        self.plugin.config["enable_timing"] = False
        original = m.PipelineScheduler.execute
        await self.plugin.initialize()
        self.assertIs(original, m.PipelineScheduler.execute)

    async def test_image_permission_and_disabled_config(self):
        self.assertIn("仅限", await self.plugin.make_image(Event(False), "a cat"))
        self.assertIn("尚未启用", await self.plugin.make_image(Event(), "a cat"))

    async def test_image_failure_does_not_repeat_a_paid_request(self):
        self.plugin.config["image_generation"] = {
            "enabled": True,
            "api_base": "https://example.com/v1",
            "api_key": "test-key",
        }
        event = Event()
        with patch.object(
            m,
            "request_with_retries",
            new=AsyncMock(side_effect=m.ImageRequestError("timeout")),
        ) as send:
            await self.plugin.make_image(event, "a cat")
            second = await self.plugin.make_image(event, "a cat")
            self.assertEqual(send.await_count, 1)
            self.assertIn("已提交", second)

    async def test_image_tool_background_flag_restored(self):
        tool = SimpleNamespace(is_background_task=False)
        self.plugin.context.get_llm_tool_manager = lambda: SimpleNamespace(
            get_func=lambda name: tool
        )
        await self.plugin.initialize()
        self.assertTrue(tool.is_background_task)
        await self.plugin.terminate()
        self.assertFalse(tool.is_background_task)

    async def test_background_flag_keeps_permission_proxy(self):
        from astrbot.core.agent.tool import FunctionTool
        from astrbot.core.provider.func_tool_manager import FunctionToolManager

        manager = FunctionToolManager()
        manager.func_list = [
            FunctionTool(
                name="wechat_generate_image",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
            FunctionTool(
                name="other",
                description="test",
                parameters={"type": "object", "properties": {}},
            ),
        ]
        self.plugin.context.get_llm_tool_manager = lambda: manager
        await self.plugin.initialize()
        guarded = manager.get_full_tool_set().get_tool("wechat_generate_image")
        self.assertTrue(guarded.is_background_task)
        self.assertEqual(type(guarded).__name__, "_PermissionGuardedTool")
        self.assertIsNone(guarded.handler)
        self.assertFalse(
            manager.get_full_tool_set().get_tool("other").is_background_task
        )
        await self.plugin.terminate()
        self.assertFalse(manager.get_func("wechat_generate_image").is_background_task)

    async def test_model_selection_and_permission(self):
        self.plugin._model_choices = AsyncMock(
            return_value=(list(m.PRESET_MODELS), True)
        )
        self.assertIn("仅个人微信管理员", await self.plugin.image_models(Event(False)))
        self.plugin._model_choices.assert_not_awaited()
        await self.plugin.image_models(Event(), "select", "flare")
        self.assertEqual(
            self.plugin.config["image_generation"]["model"], "gpt-image-2.5-flare"
        )
        self.plugin.config["image_generation"]["extra_models"] = ["custom-image"]
        await self.plugin.image_models(Event(), "select", "custom-image")
        self.assertEqual(
            self.plugin.config["image_generation"]["model"], "custom-image"
        )

    async def test_image_send_failure_never_regenerates(self):
        self.plugin.config["image_generation"] = {
            "enabled": True,
            "api_base": "https://example.com/v1",
            "api_key": "test-key",
        }
        output = Path(self.tmp.name) / "image.png"
        output.write_bytes(b"test")
        event = Event()
        event.send = AsyncMock(side_effect=RuntimeError("send failed"))
        with patch.object(
            m, "request_with_retries", new=AsyncMock(return_value=output)
        ) as post:
            result = await self.plugin.make_image(event, "cat")
            self.assertIn("发送失败不会重新生成", result)
            self.assertEqual(post.await_count, 1)

    async def test_forward_flatten_preserves_order_and_original(self):
        from astrbot.api.message_components import File, Image, Node, Nodes, Plain

        image = Image.fromFileSystem("/tmp/example.png")
        file = File(name="sound.wav", file="/tmp/sound.wav")
        original = m.MessageChain(
            [
                Nodes(
                    [
                        Node(content=[Plain("header"), image]),
                        Node(content=[Nodes([Node(content=[file])])]),
                    ]
                )
            ]
        )
        flat = m.flatten_forward_chain(original)
        self.assertEqual(
            [type(x).__name__ for x in flat.chain], ["Plain", "Image", "File"]
        )
        self.assertIs(flat.chain[1], original.chain[0].nodes[0].content[1])
        self.assertEqual(type(original.chain[0]).__name__, "Nodes")

    async def test_forward_adapter_preserves_bot_and_restores(self):
        from astrbot.api.message_components import Node, Nodes, Plain

        seen = []

        async def original(adapter, session, message):
            seen.append((adapter, session, message))

        with patch.object(m.WeixinOCAdapter, "send_by_session", original):
            await self.plugin.initialize()
            for bot in ["first", "second"]:
                await m.WeixinOCAdapter.send_by_session(
                    bot,
                    bot + "-session",
                    m.MessageChain([Nodes([Node(content=[Plain("ok")])])]),
                )
            self.assertEqual(
                [(x[0], x[1]) for x in seen],
                [("first", "first-session"), ("second", "second-session")],
            )
            self.assertTrue(all(type(x[2].chain[0]).__name__ == "Plain" for x in seen))
            await self.plugin.terminate()
            self.assertIs(m.WeixinOCAdapter.send_by_session, original)

    async def test_large_video_link_keeps_original(self):
        from astrbot.api.message_components import Video

        source = Path(self.tmp.name) / "original.mp4"
        source.write_bytes(b"original-bytes")
        links = SimpleNamespace(
            create=AsyncMock(
                return_value={
                    "url": "https://example.com/api/file/token",
                    "size_bytes": source.stat().st_size,
                }
            )
        )
        chain = m.MessageChain([Video.fromFileSystem(str(source))])
        output = await m.link_large_videos(
            chain, links, "https://example.com", threshold_mb=0, ttl=3600
        )
        self.assertEqual(type(output.chain[0]).__name__, "Plain")
        self.assertIn("原视频", output.chain[0].text)
        self.assertEqual(source.read_bytes(), b"original-bytes")
        small = await m.link_large_videos(
            chain, links, "https://example.com", threshold_mb=20
        )
        self.assertEqual(type(small.chain[0]).__name__, "Video")
        self.assertEqual(links.create.await_count, 1)

    async def test_video_link_api_uses_native_json_and_rejects_paths(self):
        from astrbot.api.web import PluginRequest

        request = unittest.mock.create_autospec(PluginRequest, instance=True)
        request.json.return_value = {"filename": "../private.mp4"}
        with patch.object(m, "request", request):
            result = await self.plugin.create_original_video_link()
        self.assertEqual(result["message"], "Invalid video filename")
        request.json.assert_awaited_once()

        root = Path(self.tmp.name) / "plugin_data/astrbot_plugin_parser/cache"
        root.mkdir(parents=True)
        source = root / "video.mp4"
        source.write_bytes(b"original")
        request.json.return_value = {"filename": source.name}
        self.plugin.downloads.create = AsyncMock(return_value={"url": "test"})
        with patch.object(m, "request", request):
            result = await self.plugin.create_original_video_link()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.plugin.downloads.create.await_args.args[0], source)

    async def test_turn_meter_uses_deltas_and_deduplicates(self):
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
        from data.plugins.astrbot_plugin_wechat_toolkit.turn_usage import (
            TurnUsage,
            new_state,
        )

        event = Event()
        event.set_extra("wechat_turn_usage", new_state())
        context = SimpleNamespace(context=SimpleNamespace(event=event))
        stats = {
            "start_time": 1,
            "end_time": 2,
            "token_usage": {"input_other": 10, "input_cached": 20, "output": 5},
        }
        runner = SimpleNamespace(
            done=lambda: True,
            run_context=context,
            stats=SimpleNamespace(to_dict=lambda: stats.copy()),
            provider=SimpleNamespace(
                provider_config={"id": "ark"},
                meta=lambda: SimpleNamespace(id="ark"),
                get_model=lambda: "deepseek-v4-1-flash-economy-261017",
            ),
        )

        async def step(runner):
            yield SimpleNamespace(type="agent_stats")
            yield SimpleNamespace(type="agent_stats")
            stats.update(
                end_time=3,
                token_usage={"input_other": 15, "input_cached": 40, "output": 8},
            )
            yield SimpleNamespace(type="agent_stats")
            yield SimpleNamespace(type="llm_result")

        with patch.object(ToolLoopAgentRunner, "step", step):
            meter = TurnUsage(self.plugin)
            meter.install()
            try:
                async for _ in ToolLoopAgentRunner.step(runner):
                    pass
                state = event.get_extra("wechat_turn_usage")
                self.assertEqual(len(state["records"]), 2)
                self.assertEqual(
                    sum(
                        r["input_other"] + r["input_cached"] + r["output"]
                        for r in state["records"]
                    ),
                    63,
                )
                self.assertTrue(state["done"])
            finally:
                meter.close()

    async def test_streaming_receipt_once(self):
        from astrbot.core.platform.sources.weixin_oc.weixin_oc_event import (
            WeixinOCMessageEvent,
        )
        from data.plugins.astrbot_plugin_wechat_toolkit.turn_usage import (
            TurnUsage,
            new_state,
        )

        event = Event()
        event.set_extra("wechat_turn_usage", new_state())
        event.get_extra("wechat_turn_usage")["attempted"] = True
        emitted = []

        async def original(event, generator, use_fallback=False):
            async for chain in generator:
                emitted.append(chain.get_plain_text())

        async def stream():
            yield m.MessageChain([m.Plain("answer")])

        with patch.object(WeixinOCMessageEvent, "send_streaming", original):
            meter = TurnUsage(self.plugin)
            meter.install()
            try:
                await WeixinOCMessageEvent.send_streaming(event, stream())
            finally:
                meter.close()
        self.assertEqual(sum("本次对话模型" in x for x in emitted), 1)
        self.assertIn("Token未知", emitted[-1])

    async def test_receipt_waits_for_final_and_isolated_per_event(self):
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
        from astrbot.core.platform.sources.weixin_oc.weixin_oc_event import (
            WeixinOCMessageEvent,
        )
        from data.plugins.astrbot_plugin_wechat_toolkit.turn_usage import (
            TurnUsage,
            new_state,
        )

        event, other = Event(), Event()
        for item in (event, other):
            item.set_extra("wechat_turn_usage", new_state())
        sent = []
        finished = False
        runner = SimpleNamespace(
            run_context=SimpleNamespace(context=SimpleNamespace(event=event)),
            done=lambda: finished,
        )

        async def step(runner):
            yield SimpleNamespace(type="llm_result")

        async def send(event, message):
            sent.append(message.get_plain_text())

        with (
            patch.object(ToolLoopAgentRunner, "step", step),
            patch.object(WeixinOCMessageEvent, "send", send),
        ):
            meter = TurnUsage(self.plugin)
            meter.install()
            try:
                async for _ in ToolLoopAgentRunner.step(runner):
                    pass
                await WeixinOCMessageEvent.send(
                    event, m.MessageChain([m.Plain("working")])
                )
                self.assertNotIn("本次对话模型", sent[-1])
                finished = True
                async for _ in ToolLoopAgentRunner.step(runner):
                    pass
                await WeixinOCMessageEvent.send(
                    event, m.MessageChain([m.Plain("answer")])
                )
                await WeixinOCMessageEvent.send(
                    event, m.MessageChain([m.Plain("followup")])
                )
                self.assertEqual(sum("本次对话模型" in x for x in sent), 1)
                self.assertFalse(other.get_extra("wechat_turn_usage")["added"])
                self.assertFalse(other.get_extra("wechat_turn_usage")["attempted"])
            finally:
                meter.close()
            self.assertIs(WeixinOCMessageEvent.send, send)
            self.assertIs(ToolLoopAgentRunner.step, step)

    async def test_generator_close_is_not_an_error_and_restores_patch(self):
        closed = []

        async def fake(provider, *args, **kwargs):
            try:
                yield SimpleNamespace(is_chunk=False, usage=SimpleNamespace(total=7))
            finally:
                closed.append(True)

        with patch.object(m.ProviderOpenAIResponses, "_query_stream", fake):
            await self.plugin.initialize()
            event = Event()
            token = m.current_message_event.set(event)
            try:
                stream = m.ProviderOpenAIResponses._query_stream(object())
                result = await anext(stream)
                self.assertEqual(result.usage.total, 7)
                await stream.aclose()
            finally:
                m.current_message_event.reset(token)
            self.assertTrue(closed)
            self.assertNotIn(
                "error", event.extra["wechat_toolkit_timing"]["model_calls"][0]
            )
            await self.plugin.terminate()
            self.assertIs(m.ProviderOpenAIResponses._query_stream, fake)


if __name__ == "__main__":
    unittest.main()

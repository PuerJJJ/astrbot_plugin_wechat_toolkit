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
            {"tts_provider_id": "mimo_tts", "enable_timing": True},
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

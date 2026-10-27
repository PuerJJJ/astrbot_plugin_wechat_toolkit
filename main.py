"""Persistent local operations plugin for AstrBot 4.29.3."""

import asyncio
import functools
import hashlib
import json
import sqlite3
import time
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image, Plain
from astrbot.api.star import Context, Star
from astrbot.core.pipeline.scheduler import PipelineScheduler
from astrbot.core.platform.astr_message_event import current_message_event
from astrbot.core.platform.sources.weixin_oc.weixin_oc_client import WeixinOCClient
from astrbot.core.provider.sources.openai_responses_source import (
    ProviderOpenAIResponses,
)
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

from .accounting import usage_report
from .image_client import ImageRequestError, request_image, validate_options
from .intent import is_usage_query
from .pricing import PriceStore


class UsageIntentFilter(filter.CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        return event.get_platform_name() == "weixin_oc" and is_usage_query(
            event.get_message_str()
        )


class WechatToolkit(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self.root = (
            Path(get_astrbot_data_path())
            / "plugin_data"
            / "astrbot_plugin_wechat_toolkit"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "latency.db"
        self.patches = []
        self.voice_lock = asyncio.Lock()
        self.image_lock = asyncio.Lock()
        self.prices = PriceStore(self.root)
        self.price_task = None
        self.last_images_path = self.root / "last-images.json"
        try:
            self.last_images = (
                json.loads(self.last_images_path.read_text())
                if self.last_images_path.exists()
                else {}
            )
        except (OSError, ValueError):
            self.last_images = {}
        if not isinstance(self.last_images, dict):
            self.last_images = {}
        with sqlite3.connect(self.database) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS latency (id INTEGER PRIMARY KEY, ended REAL, session_hash TEXT, data TEXT)"
            )

    async def initialize(self):
        self.price_task = asyncio.create_task(self.prices.run())
        if not self.config.get("enable_timing", False):
            return
        owner = self
        original_execute = PipelineScheduler.execute

        @functools.wraps(original_execute)
        async def execute(scheduler, event):
            if event.get_platform_name() != "weixin_oc":
                return await original_execute(scheduler, event)
            start = time.perf_counter()
            raw = event.message_obj.raw_message
            created = raw.get("create_time_ms") if isinstance(raw, dict) else None
            inbound = (
                (event.created_at - created / 1000)
                if isinstance(created, (int, float)) and created > 1e12
                else None
            )
            data = {
                "queue_s": max(0, time.time() - event.created_at),
                "inbound_timestamp_lag_s": inbound,
                "model_calls": [],
                "wechat_send_s": [],
                "tools": [],
                "_start": start,
            }
            event.set_extra("wechat_toolkit_timing", data)
            try:
                return await original_execute(scheduler, event)
            except BaseException as exc:
                data["pipeline_error"] = type(exc).__name__
                raise
            finally:
                data["pipeline_s"] = time.perf_counter() - start
                safe = {k: v for k, v in data.items() if not k.startswith("_")}
                digest = hashlib.sha256(event.unified_msg_origin.encode()).hexdigest()[
                    :24
                ]
                try:
                    with sqlite3.connect(owner.database, timeout=2) as con:
                        if event.get_message_str().strip().lstrip("/") not in {
                            "耗时",
                            "测速",
                        }:
                            con.execute(
                                "INSERT INTO latency (ended,session_hash,data) VALUES (?,?,?)",
                                (time.time(), digest, json.dumps(safe)),
                            )
                        con.execute(
                            "DELETE FROM latency WHERE id NOT IN (SELECT id FROM latency ORDER BY id DESC LIMIT 2000)"
                        )
                    logger.info("WeChat toolkit latency: " + json.dumps(safe))
                except sqlite3.Error as exc:
                    logger.warning(
                        "WeChat toolkit latency persistence failed: %s",
                        type(exc).__name__,
                    )

        original_stream = ProviderOpenAIResponses._query_stream

        @functools.wraps(original_stream)
        async def query_stream(provider, *args, **kwargs):
            event = current_message_event.get()
            data = event.get_extra("wechat_toolkit_timing") if event else None
            call = {"stream": True}
            start = time.perf_counter()
            if data is not None:
                data.setdefault("before_model_s", start - data["_start"])
                data["model_calls"].append(call)
            stream = original_stream(provider, *args, **kwargs)
            try:
                async for response in stream:
                    if response.is_chunk:
                        call.setdefault("first_delta_s", time.perf_counter() - start)
                        if response.result_chain or response.completion_text:
                            call.setdefault("first_text_s", time.perf_counter() - start)
                    elif response.usage:
                        call["tokens"] = response.usage.total
                    yield response
            except GeneratorExit:
                # The runner closes the stream after its terminal response.
                raise
            except BaseException as exc:
                call["error"] = type(exc).__name__
                raise
            finally:
                await stream.aclose()
                call["duration_s"] = time.perf_counter() - start

        original_query = ProviderOpenAIResponses._query

        @functools.wraps(original_query)
        async def query(provider, *args, **kwargs):
            event = current_message_event.get()
            data = event.get_extra("wechat_toolkit_timing") if event else None
            start = time.perf_counter()
            call = {"stream": False}
            if data is not None:
                data.setdefault("before_model_s", start - data["_start"])
                data["model_calls"].append(call)
            try:
                return await original_query(provider, *args, **kwargs)
            except BaseException as exc:
                call["error"] = type(exc).__name__
                raise
            finally:
                call["duration_s"] = time.perf_counter() - start

        original_http = WeixinOCClient.request_json

        @functools.wraps(original_http)
        async def request_json(client, method, endpoint, **kwargs):
            start = time.perf_counter()
            event = current_message_event.get()
            data = event.get_extra("wechat_toolkit_timing") if event else None
            try:
                return await original_http(client, method, endpoint, **kwargs)
            finally:
                if data is not None and endpoint.endswith("/sendmessage"):
                    data["wechat_send_s"].append(time.perf_counter() - start)

        for cls, name, replacement in [
            (PipelineScheduler, "execute", execute),
            (ProviderOpenAIResponses, "_query_stream", query_stream),
            (ProviderOpenAIResponses, "_query", query),
            (WeixinOCClient, "request_json", request_json),
        ]:
            self.patches.append((cls, name, getattr(cls, name), replacement))
            setattr(cls, name, replacement)
        logger.info(
            "WeChat toolkit initialized; timing contains no message text or credentials"
        )

    async def terminate(self):
        if self.price_task is not None:
            self.price_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.price_task
            self.price_task = None
        for cls, name, original, replacement in reversed(self.patches):
            if getattr(cls, name) is replacement:
                setattr(cls, name, original)
        self.patches.clear()

    @filter.custom_filter(UsageIntentFilter, priority=100)
    async def usage(self, event: AstrMessageEvent):
        if event.get_platform_name() != "weixin_oc":
            return
        if not self.config.get(
            "natural_shortcuts", True
        ) and event.get_message_str().strip().lstrip("/") not in {
            "余额",
            "额度",
            "用量",
            "费用",
            "消费",
            "用量统计",
        }:
            return
        yield event.plain_result(await self.query_usage(event))
        event.stop_event()

    @filter.llm_tool(name="wechat_query_usage")
    async def query_usage(self, event: AstrMessageEvent) -> str:
        """查询本机 AstrBot 的真实用量统计。用户询问本机器人/模型/API今天或累计调用次数、Token、花了多少钱、费用、用量、余额或剩余额度时使用；包括“看看我用了多少”“算算这个助手到现在花了多少”“还剩多少额度”。直接使用此工具，不要执行Shell或自行读取配置/数据库。返回今日和累计统计报告，请保留格式与估算口径。此工具没有账户余额/剩余额度数据，不能把累计消费当余额或编造充值金额。银行卡、话费、购物账单、概念解释、代码开发等无关话题不要调用。"""
        if event.get_platform_name() != "weixin_oc":
            return "本工具仅支持个人微信渠道。"
        if not event.is_admin():
            return "仅管理员可以查看本机用量。"
        try:
            snapshot = await self.prices.get()
            app_config = self.context.get_config()
            sources = {
                item["id"]: item for item in app_config.get("provider_sources", [])
            }
            eligible = []
            for item in app_config.get("provider", []):
                source = sources.get(item.get("provider_source_id"), item)
                tier = (item.get("custom_extra_body") or {}).get(
                    "service_tier",
                    (source.get("custom_extra_body") or {}).get(
                        "service_tier", "default"
                    ),
                )
                if urlsplit(
                    source.get("api_base", "")
                ).hostname == "ark.cn-beijing.volces.com" and tier in (None, "default"):
                    eligible.append(item["id"])
            return await asyncio.to_thread(
                usage_report,
                Path(get_astrbot_data_path()) / "data_v4.db",
                snapshot=snapshot,
                price_label=self.prices.label(),
                eligible_provider_ids=eligible,
            )
        except sqlite3.Error as exc:
            logger.warning("Usage query failed: %s", type(exc).__name__)
            return "暂时无法读取用量统计，请稍后再试。"

    @filter.regex(r"^/?(?:资费|刷新资费)$", priority=100)
    async def tariff(self, event: AstrMessageEvent):
        if event.get_platform_name() != "weixin_oc":
            return
        if not event.is_admin():
            yield event.plain_result("仅管理员可以查看或刷新资费。")
            event.stop_event()
            return
        snapshot = (
            await self.prices.refresh(force=True)
            if "刷新" in event.get_message_str()
            else await self.prices.get()
        )
        lines = ["📋 火山公开模型资费", self.prices.label()]
        if snapshot:
            for model, periods in snapshot["models"].items():
                lines.extend(
                    [
                        model,
                        "空闲 输入/缓存/输出：" + " / ".join(periods["offpeak"]),
                        "高峰 输入/缓存/输出：" + " / ".join(periods["peak"]),
                    ]
                )
            lines.append("单位：人民币/百万Token；账户折扣与实扣账单未接入。")
            lines.append(snapshot["source"])
        yield event.plain_result("\n".join(lines))
        event.stop_event()

    @filter.regex(r"^/?(?:生图|画图|改图)(?:\s|[：:])", priority=100)
    async def image_command(self, event: AstrMessageEvent):
        if event.get_platform_name() != "weixin_oc":
            return
        text = event.get_message_str().strip().lstrip("/")
        result = await self.make_image(
            event,
            text[2:].lstrip(" ：:"),
            mode="edit" if text.startswith("改图") else "generate",
        )
        yield event.plain_result(result)
        event.stop_event()

    @filter.llm_tool(name="wechat_generate_image")
    async def make_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        mode: str = "generate",
        quality: str = "",
        size: str = "",
    ) -> str:
        """用户明确要求生成图片、画图或修改图片时使用配置的生图服务。编辑使用本条消息图片，无图片则使用本会话上次生成图。只执行一次付费请求，失败或超时不得自动重试或换模型。用户没有要求生图时不要调用。不能读取任意本地文件。

        Args:
            prompt(string): 具体描述目标、风格和构图；改图写明修改内容及必须保留的内容。
            mode(string): generate为文生图，edit为参考图或修改上一张图。
            quality(string): 仅当用户指定时传low、medium、high、xhigh、max或auto，否则留空。
            size(string): 仅当用户指定时传auto或宽x高，否则留空。
        """
        if event.get_platform_name() != "weixin_oc" or not event.is_admin():
            return "生图功能仅限个人微信管理员使用。"
        config = dict(self.config.get("image_generation", {}))
        if not config.get("enabled", False):
            return "生图尚未启用，请在插件配置填写生图地址和API Key并开启。"
        if quality:
            config["quality"] = quality
        if size:
            config["edit_size" if mode == "edit" else "size"] = size
        if event.get_extra("wechat_toolkit_image_attempted"):
            return "本条消息已尝试过生图；为避免重复计费，请等待结果或由用户发新消息明确要求重试。"
        if self.image_lock.locked():
            return "已有生图任务正在处理，请稍后再试。"
        digest = hashlib.sha256(event.unified_msg_origin.encode()).hexdigest()
        refs = []
        try:
            validate_options(config, mode)
            if mode == "edit":
                for component in event.get_messages():
                    if isinstance(component, Image):
                        refs.append(Path(await component.convert_to_file_path()))
                if not refs and digest in self.last_images:
                    previous = (
                        self.root / "images" / Path(self.last_images[digest]).name
                    )
                    if previous.is_file():
                        refs = [previous]
                if not 1 <= len(refs) <= 4:
                    return "改图需要本条消息包含1～4张图片，或本会话已有生成图片。"
            async with self.image_lock:
                event.set_extra("wechat_toolkit_image_attempted", True)
                output = await request_image(
                    config, prompt, mode, refs, self.root / "images"
                )
                self.last_images.pop(digest, None)
                self.last_images[digest] = output.name
                self.last_images = dict(list(self.last_images.items())[-200:])
                tmp = self.last_images_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self.last_images))
                tmp.replace(self.last_images_path)
                await event.send(MessageChain([Image.fromFileSystem(str(output))]))
                return "图片已生成并提交微信发送接口。"
        except ImageRequestError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - sanitized image delivery boundary
            logger.warning("Image delivery failed: %s", type(exc).__name__)
            return (
                "生图或图片发送失败；若图片已生成，本次不会自动重试，以避免重复计费。"
            )

    @filter.regex(r"^/?(?:耗时|测速)$", priority=100)
    async def latency(self, event: AstrMessageEvent):
        if event.get_platform_name() != "weixin_oc":
            return
        if not event.is_admin():
            yield event.plain_result("仅管理员可以查看诊断。")
            event.stop_event()
            return
        if not self.config.get("enable_timing", False):
            yield event.plain_result("耗时诊断尚未开启，请在插件配置中开启后重载插件。")
            event.stop_event()
            return
        digest = hashlib.sha256(event.unified_msg_origin.encode()).hexdigest()[:24]
        with sqlite3.connect(self.database) as con:
            row = con.execute(
                "SELECT data FROM latency WHERE session_hash=? ORDER BY id DESC LIMIT 1",
                (digest,),
            ).fetchone()
        if not row:
            report = "还没有耗时记录，请先发一条普通消息，再发送“耗时”。"
        else:
            d = json.loads(row[0])
            calls = d["model_calls"]
            report = (
                f"最近完成消息耗时\n服务器处理合计　{d['pipeline_s']:.3f} 秒\n"
                f"排队　{d['queue_s']:.3f} 秒\n"
                f"模型请求　{len(calls)} 次 / {sum(c.get('duration_s', 0) for c in calls):.3f} 秒\n"
                f"微信发送接口　{sum(d['wechat_send_s']):.3f} 秒\n"
                f"工具调用　{sum(t.get('duration_s', 0) for t in d['tools']):.3f} 秒\n"
            )
            if calls and "first_text_s" in calls[0]:
                report += f"首次模型正文　{calls[0]['first_text_s']:.3f} 秒\n"
            if d.get("inbound_timestamp_lag_s") is not None:
                report += f"入站时间戳差　{d['inbound_timestamp_lag_s']:.3f} 秒（受时钟影响）\n"
            report += "模型时间包含网络与服务端推理；微信接口成功不等于手机已显示。"
        yield event.plain_result(report)
        event.stop_event()

    @filter.on_using_llm_tool()
    async def tool_start(self, event, tool, tool_args):
        data = event.get_extra("wechat_toolkit_timing")
        if data is not None:
            data.setdefault("_tool_starts", {})[(tool.name, id(tool_args))] = (
                time.perf_counter()
            )

    @filter.on_llm_tool_respond()
    async def tool_end(self, event, tool, tool_args, tool_result):
        data = event.get_extra("wechat_toolkit_timing")
        if data is not None:
            start = data.get("_tool_starts", {}).pop((tool.name, id(tool_args)), None)
            if start:
                data["tools"].append(
                    {"name": tool.name, "duration_s": time.perf_counter() - start}
                )

    @filter.regex(r"^/?语音(?:\s|[：:])", priority=100)
    async def voice_command(self, event: AstrMessageEvent):
        if event.get_platform_name() != "weixin_oc":
            return
        text = event.get_message_str().strip().lstrip("/")[2:].lstrip(" ：:")
        result = await self.send_voice_file(event, text)
        if result:
            yield event.plain_result(result)
        event.stop_event()

    @filter.llm_tool(name="wechat_send_voice")
    async def send_voice_file(self, event: AstrMessageEvent, text: str) -> str:
        """Use the configured TTS to send an audio file when the user explicitly asks for voice. Personal WeChat does not support native voice bubbles. Do not install TTS packages or run shell commands for speech.

        Args:
            text(string): Chinese or other speech text, at most 600 characters.
        """
        if event.get_platform_name() != "weixin_oc" or not event.is_admin():
            return "此语音工具仅限个人微信管理员使用。"
        text = text.strip()
        if not text or len(text) > 600:
            return "请提供 1～600 字的语音内容，例如：语音 你好，很高兴认识你。"
        provider_id = self.config.get("tts_provider_id", "").strip()
        provider = (
            self.context.get_provider_by_id(provider_id)
            if provider_id
            else await self.context.get_using_tts_provider_async(
                event.unified_msg_origin
            )
        )
        if provider is None:
            return "TTS 未配置或未启用，请在插件配置中选择语音提供商。"
        if self.voice_lock.locked():
            return "上一条语音仍在合成，请稍后再试。"
        async with self.voice_lock:
            try:
                start = time.perf_counter()
                path = Path(
                    await asyncio.wait_for(provider.get_audio(text), timeout=45)
                )
                data = event.get_extra("wechat_toolkit_timing")
                if data is not None:
                    data["tts_s"] = time.perf_counter() - start
                if not path.is_file() or path.stat().st_size == 0:
                    return "语音合成没有生成有效文件。"
                await event.send(
                    MessageChain(
                        [
                            Plain("语音已合成。当前微信渠道以音频文件发送："),
                            File(name="语音回复" + path.suffix, file=str(path)),
                        ]
                    )
                )
                return "音频文件已成功提交微信接口。"
            except Exception as exc:  # noqa: BLE001 - provider and delivery boundary
                logger.warning("TTS voice delivery failed: %s", type(exc).__name__)
                return (
                    "语音合成或发送失败，请查看 AstrBot 日志；本次不能声称已发送成功。"
                )

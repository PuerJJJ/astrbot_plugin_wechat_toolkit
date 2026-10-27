"""Flatten forward-node components for adapters without native forward support."""

from copy import copy

from astrbot.api.message_components import Node, Nodes, Plain


def flatten_forward_chain(message):
    """Preserve order and metadata without mutating the original chain."""
    if not any(isinstance(x, (Node, Nodes)) for x in message.chain):
        return message
    flat = []

    def visit(items, depth):
        if depth > 8:
            raise ValueError("Forward nesting exceeds the supported depth")
        for item in items:
            if isinstance(item, Nodes):
                visit(item.nodes, depth + 1)
            elif isinstance(item, Node):
                if (
                    flat
                    and isinstance(flat[-1], Plain)
                    and item.content
                    and isinstance(item.content[0], Plain)
                ):
                    flat.append(Plain("\n"))
                visit(item.content, depth + 1)
            else:
                flat.append(item)
                if len(flat) > 100:
                    raise ValueError("Forward message exceeds 100 segments")

    visit(message.chain, 0)
    result = copy(message)
    result.chain = flat
    return result


async def link_large_videos(message, links, base_url, threshold_mb=20, ttl=3600):
    """Preserve source bytes; replace only large videos with scoped download URLs."""
    from pathlib import Path

    from astrbot.api.message_components import Video

    ttl = max(300, min(86400, int(ttl)))
    result = copy(message)
    result.chain = list(message.chain)
    if not base_url:
        return result
    for index, segment in enumerate(message.chain):
        if not isinstance(segment, Video):
            continue
        source = Path(await segment.convert_to_file_path())
        if source.is_file() and source.stat().st_size > threshold_mb * 1024 * 1024:
            info = await links.create(source, base_url, ttl)
            result.chain[index] = Plain(
                f"🎬 原视频 · {info['size_bytes'] / 1024 / 1024:.1f} MB\n下载链接（{ttl // 60}分钟内有效）：\n{info['url']}"
            )
    return result

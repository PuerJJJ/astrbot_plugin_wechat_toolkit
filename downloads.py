"""Short-lived single-file links using AstrBot's built-in token service."""

import asyncio
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


class DownloadLinks:
    def __init__(self, root, token_service):
        self.root = Path(root) / "downloads"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.index = self.root / "leases.json"
        self.service = token_service
        self.lock = asyncio.Lock()
        self.leases = {}
        try:
            data = json.loads(self.index.read_text()) if self.index.exists() else {}
            if isinstance(data, dict):
                self.leases = {
                    k: float(v)
                    for k, v in data.items()
                    if re.fullmatch(r"[a-f0-9]{32}\.[A-Za-z0-9]{1,8}", k)
                    and isinstance(v, (int, float))
                }
        except (OSError, ValueError):
            pass

    def _save(self):
        temp = self.index.with_suffix(".tmp")
        temp.write_text(json.dumps(self.leases))
        temp.chmod(0o600)
        temp.replace(self.index)

    async def cleanup(self):
        async with self.lock:
            for name, expiry in list(self.leases.items()):
                if expiry < time.time():
                    (self.root / name).unlink(missing_ok=True)
                    self.leases.pop(name, None)
            self._save()

    async def create(self, source, base_url, ttl=3600):
        base = base_url.strip().rstrip("/")
        parsed = urlsplit(base)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("请配置有效的HTTPS下载入口地址。")
        source = Path(source).resolve()
        if not source.is_file():
            raise ValueError("原视频文件不存在。")
        ttl = max(300, min(86400, int(ttl)))
        await self.cleanup()
        async with self.lock:
            pin = self.root / (
                uuid.uuid4().hex
                + (
                    source.suffix
                    if re.fullmatch(r"\.[A-Za-z0-9]{1,8}", source.suffix)
                    else ".bin"
                )
            )
            try:
                await asyncio.to_thread(shutil.copyfile, source, pin)
                token = await self.service.register_file(
                    str(pin), timeout=ttl, single_use=False
                )
                expiry = time.time() + ttl
                self.leases[pin.name] = expiry
                self._save()
            except BaseException:
                pin.unlink(missing_ok=True)
                self.leases.pop(pin.name, None)
                raise
            return {
                "url": base + "/api/file/" + token,
                "expires_at": expiry,
                "size_bytes": pin.stat().st_size,
                "filename": source.name,
            }

    async def run(self):
        while True:
            await self.cleanup()
            await asyncio.sleep(300)

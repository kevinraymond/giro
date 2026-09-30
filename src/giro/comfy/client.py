"""Async client for a headless ComfyUI instance bound to localhost.

Only giro's worker layer talks to ComfyUI. The client queues API-format
prompts, streams progress and latent previews from the websocket, and fetches
the outputs a prompt produced.
"""

from __future__ import annotations

import asyncio
import json
import struct
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

# protocol.BinaryEventTypes in ComfyUI
_PREVIEW_IMAGE = 1
_PREVIEW_IMAGE_WITH_METADATA = 4


class ComfyError(RuntimeError):
    """ComfyUI rejected a prompt or failed while executing it."""


@dataclass
class Progress:
    node: str | None
    value: int
    max: int


@dataclass
class Preview:
    image: bytes
    mime: str
    node: str | None = None


@dataclass
class NodeStarted:
    node: str
    class_type: str | None


@dataclass
class Done:
    prompt_id: str
    outputs: dict[str, Any] = field(default_factory=dict)


Event = Progress | Preview | NodeStarted | Done


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8190"):
        self.base_url = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> ComfyClient:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10))
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("use ComfyClient as an async context manager")
        return self._session

    async def system_stats(self) -> dict[str, Any]:
        async with self.session.get(f"{self.base_url}/system_stats") as r:
            r.raise_for_status()
            return await r.json()

    async def upload_image(self, path: Path, subfolder: str = "giro") -> str:
        """Upload an input image; returns the name LoadImage expects."""
        form = aiohttp.FormData()
        form.add_field("image", path.read_bytes(), filename=path.name, content_type="image/png")
        form.add_field("subfolder", subfolder)
        form.add_field("overwrite", "true")
        async with self.session.post(f"{self.base_url}/upload/image", data=form) as r:
            r.raise_for_status()
            info = await r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    async def run(self, prompt: dict[str, Any]) -> AsyncIterator[Event]:
        """Queue a prompt and yield its events until it finishes.

        The websocket is opened before queueing so no early event is missed.
        """
        ws_url = self.base_url.replace("http", "ws", 1) + f"/ws?clientId={self.client_id}"
        async with self.session.ws_connect(ws_url, max_msg_size=0, heartbeat=30) as ws:
            # Ask for previews tagged with node/prompt ids.
            await ws.send_str(json.dumps({"type": "feature_flags", "data": {"supports_preview_metadata": True}}))
            prompt_id = await self._queue(prompt)
            current_node: str | None = None
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    # Previews are sent only to this client id, so they are ours.
                    if preview := _decode_preview(msg.data):
                        yield preview
                    continue
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        raise ComfyError(f"websocket closed while running {prompt_id}")
                    continue
                event = json.loads(msg.data)
                kind, data = event.get("type"), event.get("data", {})
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if kind == "executing":
                    current_node = data.get("node")
                    if current_node is not None:
                        yield NodeStarted(current_node, prompt.get(current_node, {}).get("class_type"))
                elif kind == "progress":
                    yield Progress(data.get("node"), int(data["value"]), int(data["max"]))
                elif kind == "execution_error":
                    raise ComfyError(
                        f"{data.get('node_type')} (node {data.get('node_id')}): "
                        f"{data.get('exception_type')}: {data.get('exception_message', '').strip()}"
                    )
                elif kind == "execution_interrupted":
                    raise ComfyError(f"prompt {prompt_id} was interrupted")
                elif kind == "execution_success":
                    break
            yield Done(prompt_id, await self._outputs(prompt_id))

    async def interrupt(self) -> None:
        async with self.session.post(f"{self.base_url}/interrupt") as r:
            r.raise_for_status()

    async def free(self, unload_models: bool = True) -> None:
        async with self.session.post(
            f"{self.base_url}/free", json={"unload_models": unload_models, "free_memory": True}
        ) as r:
            r.raise_for_status()

    async def download(self, image: dict[str, str], dest: Path) -> Path:
        """Fetch one output file described by a SaveImage output entry."""
        params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": image.get("type", "output")}
        async with self.session.get(f"{self.base_url}/view", params=params) as r:
            r.raise_for_status()
            dest.write_bytes(await r.read())
        return dest

    async def _queue(self, prompt: dict[str, Any]) -> str:
        async with self.session.post(
            f"{self.base_url}/prompt", json={"prompt": prompt, "client_id": self.client_id}
        ) as r:
            body = await r.json(content_type=None)
            if r.status != 200:
                raise ComfyError(_describe_validation_error(body))
        return body["prompt_id"]

    async def _outputs(self, prompt_id: str) -> dict[str, Any]:
        # History is written just after execution_success; retry briefly.
        for _ in range(20):
            async with self.session.get(f"{self.base_url}/history/{prompt_id}") as r:
                r.raise_for_status()
                history = await r.json()
            if prompt_id in history:
                return history[prompt_id].get("outputs", {})
            await asyncio.sleep(0.25)
        raise ComfyError(f"no history for prompt {prompt_id}")


def _decode_preview(data: bytes) -> Preview | None:
    if len(data) < 8:
        return None
    (event,) = struct.unpack(">I", data[:4])
    if event == _PREVIEW_IMAGE:
        (fmt,) = struct.unpack(">I", data[4:8])
        return Preview(data[8:], "image/png" if fmt == 2 else "image/jpeg")
    if event == _PREVIEW_IMAGE_WITH_METADATA:
        (meta_len,) = struct.unpack(">I", data[4:8])
        meta = json.loads(data[8 : 8 + meta_len])
        return Preview(data[8 + meta_len :], meta.get("image_type", "image/jpeg"), meta.get("node_id"))
    return None


def _describe_validation_error(body: Any) -> str:
    if not isinstance(body, dict):
        return str(body)
    parts = [body.get("error", {}).get("message", "prompt rejected")]
    for node_id, info in (body.get("node_errors") or {}).items():
        for err in info.get("errors", []):
            parts.append(f"node {node_id} ({info.get('class_type')}): {err.get('message')} {err.get('details', '')}".strip())
    return "; ".join(parts)

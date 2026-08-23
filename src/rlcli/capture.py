"""Streaming trace capture: a transparent OpenAI-compatible proxy.

Point your agent's base_url at `rlcli capture` and every completed chat call
is appended to a JSONL trace file — request messages plus the assistant
reply, tool calls included — in exactly the shape `rlcli import` emits, so
captured traces pipe straight into training. Requests are forwarded to the
upstream unchanged (SSE streams are relayed chunk-by-chunk and reassembled
for the trace), so capture is invisible to the agent.

Each trace line carries a trace_id (the upstream completion id, or the
caller's X-Trace-Id header when present) for later telemetry joins.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

_HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding",
                "transfer-encoding"}


def _assemble_stream(chunks: list[dict]) -> dict:
    """Reassemble an SSE delta stream into one assistant message."""
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    completion_id = ""
    for chunk in chunks:
        completion_id = chunk.get("id") or completion_id
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    idx, {"id": "", "type": "function",
                          "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    message: dict = {"role": "assistant", "content": "".join(content_parts)}
    calls = [tool_calls[i] for i in sorted(tool_calls)]
    if calls:
        message["tool_calls"] = calls
    return {"message": message, "id": completion_id}


class _TraceWriter:
    def __init__(self, path: str):
        self._path = path
        self._lock = threading.Lock()

    def write(self, request_body: dict, reply: dict, trace_id: str) -> None:
        messages = list(request_body.get("messages", []))
        messages.append(reply)
        record = {"messages": messages}
        if trace_id:
            record["trace_id"] = trace_id
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_handler(upstream: str, writer: _TraceWriter, api_key: str | None,
                 client: httpx.Client):
    class CaptureHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in _HOP_HEADERS}
            if api_key and "authorization" not in {k.lower() for k in headers}:
                headers["Authorization"] = f"Bearer {api_key}"
            url = upstream.rstrip("/") + self.path
            caller_trace_id = self.headers.get("X-Trace-Id", "")
            try:
                with client.stream("POST", url, content=raw, headers=headers,
                                   timeout=600) as resp:
                    self.send_response(resp.status_code)
                    for k, v in resp.headers.items():
                        if k.lower() not in _HOP_HEADERS:
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if body.get("stream"):
                        chunks: list[dict] = []
                        for line_bytes in resp.iter_raw():
                            self.wfile.write(line_bytes)
                            self.wfile.flush()
                            for line in line_bytes.decode("utf-8", "replace").splitlines():
                                line = line.strip()
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        chunks.append(json.loads(line[6:]))
                                    except json.JSONDecodeError:
                                        pass
                        if resp.status_code == 200 and body.get("messages"):
                            assembled = _assemble_stream(chunks)
                            writer.write(body, assembled["message"],
                                         caller_trace_id or assembled["id"])
                    else:
                        data = resp.read()
                        self.wfile.write(data)
                        if resp.status_code == 200 and body.get("messages"):
                            try:
                                payload = json.loads(data)
                                reply = payload["choices"][0]["message"]
                                writer.write(body, reply,
                                             caller_trace_id or payload.get("id", ""))
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
            except httpx.HTTPError as e:
                self.send_response(502)
                msg = json.dumps({"error": f"rlcli capture: upstream error: {e}"}).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

    return CaptureHandler


def serve_capture(listen: str, upstream: str, out_path: str,
                  api_key: str | None = None) -> ThreadingHTTPServer:
    """Start the capture proxy (non-blocking); caller drives serve_forever."""
    host, _, port = listen.rpartition(":")
    writer = _TraceWriter(out_path)
    client = httpx.Client()
    handler = make_handler(upstream, writer, api_key, client)
    return ThreadingHTTPServer((host or "127.0.0.1", int(port)), handler)

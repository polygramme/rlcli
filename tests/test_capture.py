import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from rlcli.capture import _assemble_stream, serve_capture


class _StubUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            chunks = [
                {"id": "cmpl-s1", "choices": [{"delta": {"content": "Hel"}}]},
                {"id": "cmpl-s1", "choices": [{"delta": {"content": "lo"}}]},
            ]
            payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            data = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            reply = {"id": "cmpl-1", "choices": [{"message": {
                "role": "assistant", "content": "4",
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "calc", "arguments": "{}"}}],
            }}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


@pytest.fixture()
def proxy(tmp_path):
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _StubUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    out = tmp_path / "traces.jsonl"
    server = serve_capture(
        "127.0.0.1:0", f"http://127.0.0.1:{upstream.server_port}", str(out))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", out
    server.shutdown()
    upstream.shutdown()


def test_capture_non_streaming_with_tool_calls(proxy):
    base, out = proxy
    resp = httpx.post(base + "/chat/completions", json={
        "model": "m", "messages": [{"role": "user", "content": "2+2?"}]})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "4"
    row = json.loads(out.read_text().strip())
    assert row["trace_id"] == "cmpl-1"
    assert row["messages"][0] == {"role": "user", "content": "2+2?"}
    assert row["messages"][1]["tool_calls"][0]["function"]["name"] == "calc"


def test_capture_streaming_reassembles(proxy):
    base, out = proxy
    with httpx.stream("POST", base + "/chat/completions", json={
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}) as resp:
        text = "".join(resp.iter_text())
    assert "data: [DONE]" in text
    row = json.loads(out.read_text().strip())
    assert row["messages"][-1] == {"role": "assistant", "content": "Hello"}
    assert row["trace_id"] == "cmpl-s1"


def test_assemble_stream_tool_call_deltas():
    chunks = [
        {"id": "x", "choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c9", "function": {"name": "se"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "arch", "arguments": '{"q":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"x"}'}}]}}]},
    ]
    result = _assemble_stream(chunks)
    (call,) = result["message"]["tool_calls"]
    assert call["id"] == "c9"
    assert call["function"]["name"] == "search"
    assert call["function"]["arguments"] == '{"q":"x"}'


def test_import_project_requires_langsmith_sdk_or_mocks(tmp_path, monkeypatch):
    """--project uses the langsmith SDK when present; here we inject a fake."""
    import sys
    import types
    import uuid

    from click.testing import CliRunner

    from rlcli.cli import main_cli

    run = types.SimpleNamespace(
        id=uuid.UUID(int=1), trace_id=uuid.UUID(int=1),
        inputs={"messages": [{"type": "human", "content": "hi"}]},
        outputs={"output": {"role": "assistant", "content": "hello"}},
        feedback_stats={"quality": {"n": 1, "avg": 0.9}},
    )

    class FakeClient:
        def list_runs(self, project_name, is_root, limit):
            assert project_name == "prod-agent"
            yield run

    monkeypatch.setitem(sys.modules, "langsmith",
                        types.SimpleNamespace(Client=FakeClient))
    result = CliRunner().invoke(main_cli, ["import", "--project", "prod-agent"])
    assert result.exit_code == 0, result.output
    row = json.loads(next(l for l in result.output.splitlines() if l.startswith("{")))
    assert row["reward"] == 0.9
    assert row["messages"][-1]["content"] == "hello"
    assert row["trace_id"] == str(uuid.UUID(int=1))

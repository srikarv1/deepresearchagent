"""Local OpenAI-compatible /v1/chat/completions endpoint backed by Bedrock Converse.

The account's Organizations policy blocks Bedrock bearer (API) keys, so the
OpenAI-compatible "mantle" endpoint is unusable. This shim signs requests with
the IAM access keys (SigV4, via boto3) instead, so the agent, the replay writer
and the judges keep talking plain OpenAI.

    set -a; source .env; set +a
    python scripts/bedrock_openai_shim.py          # serves http://127.0.0.1:4000/v1

Supports: system/user/assistant messages (string or text-part content),
max_tokens / max_completion_tokens, temperature, top_p, stop, response_format
(json_schema via a forced tool call; json_object via an instruction) and a
single-chunk stream=true. Usage tokens are passed through.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import boto3
from botocore.config import Config

MODELS = {
    "openai.gpt-oss-120b": "openai.gpt-oss-120b-1:0",
    "openai.gpt-oss-20b": "openai.gpt-oss-20b-1:0",
    "qwen.qwen3-32b": "qwen.qwen3-32b-v1:0",
}
PORT = int(os.environ.get("SHIM_PORT", "4000"))
_client = boto3.client(
    "bedrock-runtime",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    config=Config(read_timeout=600, connect_timeout=30, retries={"max_attempts": 6, "mode": "adaptive"}),
)
_lock = threading.Lock()
_usage = {"calls": 0, "in": 0, "out": 0}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def _converse_args(body: dict) -> tuple[dict, str | None]:
    model = body.get("model", "")
    model_id = MODELS.get(model, model)
    system, msgs = [], []
    for m in body.get("messages", []):
        role, text = m.get("role"), _text(m.get("content"))
        if role in ("system", "developer"):
            system.append({"text": text})
            continue
        role = "assistant" if role == "assistant" else "user"
        if msgs and msgs[-1]["role"] == role:  # Converse needs alternating turns
            msgs[-1]["content"][0]["text"] += "\n\n" + text
        else:
            msgs.append({"role": role, "content": [{"text": text or " "}]})
    if not msgs or msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": [{"text": " "}]})

    cfg = {}
    mt = body.get("max_completion_tokens") or body.get("max_tokens")
    cfg["maxTokens"] = int(mt) if mt else 8192
    if body.get("temperature") is not None:
        cfg["temperature"] = float(body["temperature"])
    if body.get("top_p") is not None:
        cfg["topP"] = float(body["top_p"])
    if body.get("stop"):
        cfg["stopSequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]

    args = {"modelId": model_id, "messages": msgs, "inferenceConfig": cfg}
    tool_name = None
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        tool_name = (js.get("name") or "respond")[:64]
        args["toolConfig"] = {
            "tools": [{"toolSpec": {"name": tool_name, "description": "Return the answer in this structure.",
                                    "inputSchema": {"json": js.get("schema") or {"type": "object"}}}}],
            "toolChoice": {"tool": {"name": tool_name}},
        }
    elif rf.get("type") == "json_object":
        system.append({"text": "Respond with a single valid JSON object and nothing else."})
    if system:
        args["system"] = system
    return args, tool_name


def complete(body: dict) -> dict:
    args, tool_name = _converse_args(body)
    r = _client.converse(**args)
    content, finish = "", "stop"
    for block in r["output"]["message"]["content"]:
        if "text" in block:
            content += block["text"]
        elif "toolUse" in block and tool_name:
            content = json.dumps(block["toolUse"]["input"])
    if r.get("stopReason") == "max_tokens":
        finish = "length"
    u = r.get("usage", {})
    with _lock:
        _usage["calls"] += 1
        _usage["in"] += u.get("inputTokens", 0)
        _usage["out"] += u.get("outputTokens", 0)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": u.get("inputTokens", 0), "completion_tokens": u.get("outputTokens", 0),
                  "total_tokens": u.get("totalTokens", 0)},
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, payload: dict | None = None, raw: bytes | None = None, ctype="application/json"):
        data = raw if raw is not None else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._send(200, {"ok": True, "usage": _usage})
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": k, "object": "model"} for k in MODELS]})
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._send(404, {"error": {"message": f"unsupported path {self.path}"}})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            out = complete(body)
        except Exception as e:  # surface as an OpenAI-style error so client retries work
            code = 429 if "Throttl" in type(e).__name__ or "Throttl" in str(e) else 500
            return self._send(code, {"error": {"message": f"{type(e).__name__}: {str(e)[:400]}", "type": "bedrock_error"}})
        if body.get("stream"):
            chunk = {**out, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": out["choices"][0]["message"]["content"]},
                                  "finish_reason": out["choices"][0]["finish_reason"]}]}
            sse = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            return self._send(200, raw=sse, ctype="text/event-stream")
        self._send(200, out)


if __name__ == "__main__":
    print(f"bedrock shim on http://127.0.0.1:{PORT}/v1", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

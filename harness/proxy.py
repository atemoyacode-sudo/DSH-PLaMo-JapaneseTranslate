"""An OpenAI-compatible server in front of the model, so that existing chat apps get Japanese in and out.

  python -m harness.proxy --port 8092      # then point a chat app at http://127.0.0.1:8092/v1

POST /v1/chat/completions (stream or not) and GET /v1/models. Japanese user, system and assistant text is
translated to English for the model; the answer comes back in Japanese when the last user message is Japanese
(otherwise as the model wrote it). Tool definitions, tool calls and tool results pass through unchanged. The model's
thinking is dropped unless --pass-thinking (then it is sent untranslated as `reasoning_content`).
Binds 127.0.0.1 by default and checks no API key: do not expose it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chat import add_common, build
from .pipeline import Harness, UpstreamError

KEEPALIVE_S = 10.0


def make_handler(h: Harness, served_name: str, pass_thinking: bool):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # one short line per request on stderr
            sys.stderr.write(f"[proxy] {self.command} {self.path} {fmt % a}\n")

        def _json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                entry = {"id": served_name, "object": "model", "owned_by": "harness"}
                if h.translator.credit:
                    entry["description"] = h.translator.credit
                self._json(200, {"object": "list", "data": [entry]})
            elif self.path.rstrip("/") in ("", "/health"):
                self._json(200, {"status": "ok", **({"credit": h.translator.credit} if h.translator.credit else {})})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                self._json(404, {"error": {"message": "only /v1/chat/completions is served"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                messages = body["messages"]
            except (ValueError, KeyError) as e:
                self._json(400, {"error": {"message": f"bad request: {e}"}})
                return
            try:
                msgs, target = h.prepare(messages)
                events = h.run(msgs, target, body)
                if body.get("stream"):
                    self._stream(events, body)
                else:
                    self._complete(events)
            except UpstreamError as e:
                self._json(502, {"error": {"message": str(e)}})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _complete(self, events) -> None:
            reasoning, text, calls, fin = [], [], None, {}
            for ev in events:
                if ev.kind == "reasoning":
                    reasoning.append(ev.text)
                elif ev.kind == "text":
                    text.append(ev.text)
                elif ev.kind == "tool_calls":
                    calls = [{k: v for k, v in c.items() if k != "index"} for c in ev.data]
                elif ev.kind == "finish":
                    fin = ev.data
            msg = {"role": "assistant", "content": "".join(text) or (None if calls else "")}
            if calls:
                msg["tool_calls"] = calls
            if pass_thinking and reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                             "model": served_name, "choices": [{"index": 0, "message": msg,
                                                                "finish_reason": "tool_calls" if calls else fin.get("finish_reason")}],
                             "usage": fin.get("usage")})

        def _stream(self, events, body) -> None:
            cid, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def send(delta=None, finish=None, usage=None, choices=True):
                obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": served_name,
                       "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}] if choices else []}
                if usage is not None:
                    obj["usage"] = usage
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()

            send({"role": "assistant", "content": ""})
            last_beat, had_calls = time.monotonic(), False
            try:
                for ev in events:
                    if ev.kind == "text" and ev.text:
                        send({"content": ev.text})
                    elif ev.kind == "reasoning" and pass_thinking:
                        send({"reasoning_content": ev.text})
                    elif ev.kind == "tool_calls":
                        had_calls = True
                        send({"tool_calls": ev.data})
                    elif ev.kind == "finish":
                        send(finish="tool_calls" if had_calls else ev.data["finish_reason"])
                        if ev.data.get("usage") and (body.get("stream_options") or {}).get("include_usage"):
                            send(usage=ev.data["usage"], choices=False)
                    elif time.monotonic() - last_beat > KEEPALIVE_S:  # long thinking: keep the client's connection
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        last_beat = time.monotonic()
            except UpstreamError as e:
                self.wfile.write(b"data: " + json.dumps({"error": {"message": str(e)}}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.proxy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--served-name", help="model id shown to chat apps (default astrantia-ja-plamo with PLaMo, whose "
                    "license asks for PLaMo in the name; astrantia-ja otherwise)")
    ap.add_argument("--pass-thinking", action="store_true", help="send the thinking (untranslated) as reasoning_content")
    args = ap.parse_args(argv)
    h = build(args)
    name = args.served_name or ("astrantia-ja-plamo" if h.translator.credit == "Built with PLaMo" else "astrantia-ja")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(h, name, args.pass_thinking))
    print(f"harness proxy on http://{args.host}:{srv.server_port}/v1 (model id {name}) -> model {args.model_url}, "
          f"translator {args.translator}{' — ' + h.translator.credit if h.translator.credit else ''}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke tests for the thin Python client (stdlib only)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from langchainrust import Client, LangChainRustError


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # keep test output clean
        pass

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": "mock"}]})
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for text in ("he", "llo"):
                chunk = {"choices": [{"delta": {"content": text}}]}
                self.wfile.write(
                    ("data: " + json.dumps(chunk) + "\n\n").encode()
                )
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        if "bad" in json.dumps(payload):
            self._send_json(400, {"error": {"message": "bad request"}})
            return

        self._send_json(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


class ClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = Client(
            base_url=f"http://127.0.0.1:{cls.server.server_port}",
            model="mock",
        )

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_chat_returns_content(self):
        result = self.client.chat("hi")
        self.assertEqual(result.content, "hello")

    def test_chat_parses_tool_calls(self):
        tools = [
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {"type": "object"}},
            }
        ]
        result = self.client.chat_with_tools("weather?", tools)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "get_weather")
        self.assertEqual(result.tool_calls[0].id, "call_1")

    def test_stream_yields_text(self):
        chunks = list(self.client.stream("hi"))
        self.assertEqual(chunks, ["he", "llo"])

    def test_models_lists_models(self):
        models = self.client.models()
        self.assertEqual(models[0]["id"], "mock")

    def test_http_error_raises(self):
        with self.assertRaises(LangChainRustError) as ctx:
            self.client.chat("bad")
        self.assertEqual(ctx.exception.status, 400)


if __name__ == "__main__":
    unittest.main()

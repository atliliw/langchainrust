import assert from "node:assert/strict";
import http from "node:http";
import { after, before, describe, it } from "node:test";

import { Client, LangChainRustError } from "../src/index.js";

function readBody(req) {
  return new Promise((resolve) => {
    let data = "";
    req.on("data", (chunk) => {
      data += chunk;
    });
    req.on("end", () => resolve(data ? JSON.parse(data) : {}));
  });
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

describe("langchainrust TypeScript client", () => {
  let server;
  let client;

  before(async () => {
    server = http.createServer(async (req, res) => {
      if (req.method === "GET" && req.url === "/v1/models") {
        sendJson(res, 200, { object: "list", data: [{ id: "mock" }] });
        return;
      }

      if (req.method === "POST" && req.url === "/v1/chat/completions") {
        const payload = await readBody(req);
        const serialized = JSON.stringify(payload);

        if (payload.stream) {
          res.writeHead(200, { "content-type": "text/event-stream" });
          for (const text of ["he", "llo"]) {
            res.write(
              `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\n`,
            );
          }
          res.write("data: [DONE]\n\n");
          res.end();
          return;
        }

        if (serialized.includes("bad")) {
          sendJson(res, 400, { error: { message: "bad request" } });
          return;
        }

        sendJson(res, 200, {
          choices: [
            {
              message: {
                role: "assistant",
                content: "hello",
                tool_calls: [
                  {
                    id: "call_1",
                    type: "function",
                    function: { name: "get_weather", arguments: "{}" },
                  },
                ],
              },
              finish_reason: "tool_calls",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        });
        return;
      }

      sendJson(res, 404, { error: { message: "not found" } });
    });

    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address();
    client = new Client(`http://127.0.0.1:${port}`, "mock");
  });

  after(async () => {
    await new Promise((resolve) => server.close(resolve));
  });

  it("returns content", async () => {
    const result = await client.chat("hi");
    assert.equal(result.content, "hello");
  });

  it("parses tool calls", async () => {
    const tools = [
      {
        type: "function",
        function: { name: "get_weather", parameters: { type: "object" } },
      },
    ];
    const result = await client.chatWithTools("weather?", tools);
    assert.equal(result.finishReason, "tool_calls");
    assert.equal(result.toolCalls.length, 1);
    assert.equal(result.toolCalls[0].name, "get_weather");
    assert.equal(result.toolCalls[0].id, "call_1");
  });

  it("streams text deltas", async () => {
    const chunks = [];
    for await (const chunk of client.stream("hi")) {
      chunks.push(chunk);
    }
    assert.deepEqual(chunks, ["he", "llo"]);
  });

  it("lists models", async () => {
    const models = await client.models();
    assert.equal(models[0].id, "mock");
  });

  it("raises on HTTP errors", async () => {
    await assert.rejects(
      () => client.chat("bad"),
      (error) => error instanceof LangChainRustError && error.status === 400,
    );
  });
});

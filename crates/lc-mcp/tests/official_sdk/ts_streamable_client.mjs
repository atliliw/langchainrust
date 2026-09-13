#!/usr/bin/env node
// B1 interop gate — official TypeScript MCP SDK **client** speaking to the
// langchainrust Streamable HTTP server (examples/streamable_echo_server.rs).
//
//   node ts_streamable_client.mjs --url http://127.0.0.1:PORT/mcp [--bearer TOKEN]
//
// Exits 0 only when the whole official-client flow succeeds: handshake
// (initialize + notifications/initialized, Mcp-Session-Id handled inside the
// SDK), tools/list, tools/call roundtrip, and ping. Prints TS_STREAMABLE_OK
// as its final line on success.

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

function arg(name, fallback = undefined) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const url = arg("url");
const bearer = arg("bearer");
if (!url) {
  console.error("usage: ts_streamable_client.mjs --url <http://.../mcp> [--bearer TOKEN]");
  process.exit(2);
}

const marker = `ts-sdk-marker-${Date.now()}`;

const transport = new StreamableHTTPClientTransport(
  new URL(url),
  bearer
    ? { requestInit: { headers: { Authorization: `Bearer ${bearer}` } } }
    : undefined,
);
const client = new Client(
  { name: "lc-ts-interop", version: "0.22.4" },
  { capabilities: {} },
);

try {
  // connect() performs initialize and notifications/initialized; the SDK
  // sends Accept: application/json, text/event-stream and tracks
  // Mcp-Session-Id on every subsequent request.
  await client.connect(transport);
  console.log("[ts] connected; initialize handshake complete");

  const listed = await client.listTools();
  const names = listed.tools.map((t) => t.name);
  console.log(`[ts] tools/list -> ${JSON.stringify(names)}`);
  if (!names.includes("echo")) {
    throw new Error(`expected 'echo' tool, got ${JSON.stringify(names)}`);
  }

  const called = await client.callTool({
    name: "echo",
    arguments: { msg: marker },
  });
  const text = (called.content ?? [])
    .filter((c) => c.type === "text")
    .map((c) => c.text)
    .join("");
  console.log(`[ts] tools/call -> ${JSON.stringify(text)}`);
  if (called.isError) {
    throw new Error("echo tool call flagged isError");
  }
  if (!text.includes(marker)) {
    throw new Error(`echo payload missing marker ${marker}: ${text}`);
  }

  // Official SDK keep-alive against our real MCPServer ping handler.
  await client.ping();
  console.log("[ts] ping ok");

  await client.close();
  console.log("TS_STREAMABLE_OK");
} catch (err) {
  console.error("[ts] INTEROP FAILED:", err);
  process.exit(1);
}

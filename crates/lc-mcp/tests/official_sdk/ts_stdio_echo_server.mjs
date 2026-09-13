// B1 interop gate — a REAL MCP server built on the official TypeScript SDK,
// served over stdio. The Rust ignored-by-default integration test
// (tests/official_sdk_stdio_interop.rs) spawns this file with
// StdioMcpClient and drives the full official handshake / tool matrix.
//
//   node ts_stdio_echo_server.mjs
//
// Protocol frames go to stdout; all diagnostics to stderr (SDK default).

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "ts-echo-stdio", version: "0.22.4" },
  { capabilities: { tools: { listChanged: false } } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "echo",
      description: "Echoes its msg argument back as text.",
      inputSchema: {
        type: "object",
        properties: { msg: { type: "string" } },
        required: ["msg"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  if (name !== "echo") {
    // JSON-RPC method/tool error, exactly the shape a compliant MCP server
    // returns for an unknown tool (our client must surface code -32601).
    const err = new Error(`unknown tool: ${String(name)}`);
    err.code = -32601;
    throw err;
  }
  const msg = args?.msg ?? "";
  return {
    content: [{ type: "text", text: `echo: ${msg}` }],
    isError: false,
  };
});

// `ping` is answered by the SDK protocol layer itself.
const transport = new StdioServerTransport();
await server.connect(transport);
console.error("[ts-echo-stdio] official TS SDK MCP server ready on stdio");

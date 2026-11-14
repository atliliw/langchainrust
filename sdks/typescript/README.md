# langchainrust — TypeScript thin client

Zero-dependency client for the **OpenAI-compatible** endpoint exposed by
`lc serve` (or any OpenAI-compatible server). Node 18+ (global `fetch`).

```bash
npm install langchainrust
```

```ts
import { Client } from "langchainrust";

const client = new Client("http://localhost:8080", "my-agent");

console.log((await client.chat("hello")).content);

for await (const chunk of client.stream("hello")) {
  process.stdout.write(chunk);
}

const tools = [
  {
    type: "function" as const,
    function: {
      name: "get_weather",
      description: "Get the weather for a city",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
        required: ["city"],
      },
    },
  },
];

const result = await client.chatWithTools("weather in beijing?", tools);
for (const call of result.toolCalls) {
  console.log(call.name, call.arguments);
}
```

## Tests

```bash
cd sdks/typescript
npm test
```

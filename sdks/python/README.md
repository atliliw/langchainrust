# langchainrust — Python thin client

Zero-dependency client for the **OpenAI-compatible** endpoint exposed by
`lc serve` (or any OpenAI-compatible server). It only speaks HTTP; agent logic
stays in the Rust runtime.

```bash
pip install -e sdks/python
```

```python
from langchainrust import Client

c = Client("http://localhost:8080", model="my-agent")

# Plain chat
print(c.chat("hello").content)

# Streaming
for chunk in c.stream("hello"):
    print(chunk, end="")

# Tool calling
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
result = c.chat_with_tools("weather in beijing?", tools)
for call in result.tool_calls:
    print(call.name, call.arguments)
```

## Tests

```bash
cd sdks/python
python -m unittest discover -s tests
```

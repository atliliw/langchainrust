/**
 * Thin TypeScript/JavaScript client for the langchainrust OpenAI-compatible
 * server. It only speaks HTTP; agent logic stays in the Rust runtime.
 */

export class LangChainRustError extends Error {
  constructor(message, status = undefined) {
    super(message);
    this.name = "LangChainRustError";
    this.status = status;
  }
}

function asMessages(input) {
  return typeof input === "string" ? [{ role: "user", content: input }] : input;
}

function parseResult(payload) {
  const choice = payload.choices?.[0];
  if (!choice) {
    throw new LangChainRustError("response contained no choices");
  }
  const message = choice.message ?? {};
  return {
    content: message.content ?? "",
    finishReason: choice.finish_reason ?? null,
    toolCalls: (message.tool_calls ?? []).map((call) => ({
      id: call.id ?? "",
      name: call.function?.name ?? "",
      arguments: call.function?.arguments ?? "",
    })),
    usage: payload.usage ?? null,
    raw: payload,
  };
}

export class Client {
  /**
   * @param {string} [baseUrl]
   * @param {string} [model]
   * @param {{ apiKey?: string, timeoutMs?: number }} [options]
   */
  constructor(baseUrl = "http://localhost:8080", model = "default", options = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.model = model;
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 60_000;
  }

  headers() {
    const headers = { "content-type": "application/json" };
    if (this.apiKey) {
      headers.authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async request(path, { method = "GET", body } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers(),
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new LangChainRustError(`HTTP ${response.status}: ${text}`, response.status);
      }
      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  async models() {
    const response = await this.request("/v1/models");
    const payload = await response.json();
    return payload.data ?? [];
  }

  async chat(input, { tools, ...options } = {}) {
    const body = {
      model: this.model,
      messages: asMessages(input),
      ...options,
    };
    if (tools) {
      body.tools = tools;
    }
    const response = await this.request("/v1/chat/completions", {
      method: "POST",
      body,
    });
    return parseResult(await response.json());
  }

  async chatWithTools(input, tools, options = {}) {
    return this.chat(input, { ...options, tools });
  }

  async *stream(input, options = {}) {
    const body = {
      model: this.model,
      messages: asMessages(input),
      stream: true,
      ...options,
    };
    const response = await this.request("/v1/chat/completions", {
      method: "POST",
      body,
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let newline;
        while ((newline = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, newline).trim();
          buffer = buffer.slice(newline + 1);
          if (!line.startsWith("data:")) {
            continue;
          }
          const data = line.slice("data:".length).trim();
          if (data === "[DONE]") {
            return;
          }
          try {
            const event = JSON.parse(data);
            const delta = event.choices?.[0]?.delta;
            if (delta?.content) {
              yield delta.content;
            }
          } catch {
            // Ignore keep-alives / malformed frames.
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

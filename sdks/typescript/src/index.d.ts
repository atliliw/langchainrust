export interface WireMessage {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  tool_call_id?: string;
  tool_calls?: WireToolCall[];
}

export interface WireToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface ChatResult {
  content: string;
  finishReason: string | null;
  toolCalls: ToolCall[];
  usage: Record<string, number> | null;
  raw: unknown;
}

export interface ClientOptions {
  apiKey?: string;
  timeoutMs?: number;
}

export declare class LangChainRustError extends Error {
  readonly status?: number;
}

export declare class Client {
  constructor(baseUrl?: string, model?: string, options?: ClientOptions);
  models(): Promise<Array<{ id: string }>>;
  chat(
    input: string | WireMessage[],
    options?: { tools?: ToolDefinition[] } & Record<string, unknown>,
  ): Promise<ChatResult>;
  chatWithTools(
    input: string | WireMessage[],
    tools: ToolDefinition[],
    options?: Record<string, unknown>,
  ): Promise<ChatResult>;
  stream(
    input: string | WireMessage[],
    options?: Record<string, unknown>,
  ): AsyncGenerator<string, void, unknown>;
}

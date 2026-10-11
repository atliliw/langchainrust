# lc-providers` 审阅报告

本报告为只读审计(未改动任何代码),覆盖 `crates/lc-providers/src/` 全部 `.rs` 文件(含子目录)。每条发现均标注 `文件:行号` + 触发条件 + 后果。

## 1. 概览

`lc-providers` 是本框架的模型提供方接入层,统一封装 OpenAI / Azure / Anthropic / Gemini / Cohere / Ollama / Mistral / DeepSeek / Qwen / Moonshot / Zhipu / OpenAI-compatible 等聊天与响应模型。核心组成:

- **两套并行 HTTP 重试机制**:本地 `retry::send_with_retry`(Anthropic / Gemini / Ollama 用)vs 统一的 `lc_core` HTTP 层(`provider_api_client` / `provider_sse_client`,OpenAI / Azure / Cohere / Responses 用)。
- **A14 非幂等 POST 重试边界**:`SAFE_RETRY`(`PreDispatchOnly`,只重试 connect 类)用于计费 POST,杜绝一次生成被重复扣费;`DEFAULT_RETRY`(`AllTransportErrors`)会额外重试超时。
- **o-series 推理模型路由**:`is_openai_reasoning_model()` 按 `o<数字>` 前缀判断,决定发 `max_completion_tokens` 还是 `max_tokens`。
- **SSE**:有界 `SseByteFramer`(16MiB,O(n) 线性扫描、溢出即失败)、`SSEParser`、流式 tool-call 累加器;多路 provider 都有 A12/M-8 终态检测(连接提前关闭 → `StreamInterrupted`,不把截断输出当完整答案)。

总体判断:**核心接入逻辑扎实、安全基线高**(key 全脱敏、SSE 有界、终态检测齐全、计费 POST 采用 at-most-once)。真实 bug 集中在少数几处:Azure 的 o-series `max_tokens`、两个 provider 的流式无 idle 超时、Ollama 注释/代码重试策略不一致。

## 2. 遍历的文件清单

```
crates/lc-providers/src/retry.rs
crates/lc-providers/src/provider_http.rs
crates/lc-providers/src/error.rs
crates/lc-providers/src/client.rs
crates/lc-providers/src/wrapper.rs
crates/lc-providers/src/media.rs
crates/lc-providers/src/sampling.rs
crates/lc-providers/src/lib.rs
crates/lc-providers/src/providers/mod.rs
crates/lc-providers/src/providers/anthropic/{mod,config,types,error,impls,chat,tests}.rs
crates/lc-providers/src/providers/gemini/{mod,types,error,tests}.rs
crates/lc-providers/src/providers/azure/{mod,config,types,error,tests}.rs
crates/lc-providers/src/providers/cohere/{mod,config,types,error,tests}.rs
crates/lc-providers/src/providers/mistral.rs
crates/lc-providers/src/providers/deepseek.rs
crates/lc-providers/src/providers/qwen.rs
crates/lc-providers/src/providers/moonshot.rs
crates/lc-providers/src/providers/zhipu.rs
crates/lc-providers/src/ollama/{mod,config,chat}.rs
crates/lc-providers/src/openai/{mod,config,sse}.rs
crates/lc-providers/src/openai/chat/{mod,error,tests,structured}.rs
crates/lc-providers/src/openai/responses/{mod,model,types,tests}.rs
crates/lc-providers/src/openai/{multimodal,response_format,assistants}.rs
crates/lc-providers/src/openai_compatible/{mod,config}.rs
```

## 3. 发现的 bug / 高风险点

### 3.1. [HIGH] Azure 的 o 系模型仍发 `max_tokens`,无 `max_completion_tokens` 路由

- 位置:`crates/lc-providers/src/providers/azure/mod.rs:188-190`
- 触发:`AzureOpenAIConfig::with_max_tokens(n)` 配合 Azure 上的 o1/o3/o4 推理模型部署。
- 后果:请求体始终写 `body["max_tokens"] = json!(max)`。OpenAI 的 o 系列模型拒绝 `max_tokens`(只接受 `max_completion_tokens`),此请求会以 HTTP 400 失败;用户也无法在此路径上控制推理输出长度。对比:`openai/chat/mod.rs:147-154`(和 Responses `max_output_tokens`,`responses/model.rs:152-154`)都已按模型族正确路由,Azure 未同步更新。触发条件是配置显式命中 o 系模型时才成立——虽需模型部署本身为 o 系,但这是 Azure 路径的确定性缺陷。

### 3.2. [MEDIUM-HIGH] Gemini 流式请求绕过 SSE idle 超时,对方挂起会永久悬挂

- 位置:`crates/lc-providers/src/providers/gemini/mod.rs:641-678`
- 触发:`GeminiChat::stream` / `stream_chat` 走流式路径;对端(网关/代理/模型)发送后中途停住不回包也不关连接。
- 后果:此处用 `self.client.post(url)...send()` + `response.bytes_stream()`(而不走统一层的 `open_sse` + `with_sse_idle_timeout`),client 来自 `default_client()`(只设 connect 超时,无总超时/读超时)。数据静止后任务永久 pending,调用方无任何超时兜底。A7/A12 的防护只在建流请求本身,不覆盖传输中。

### 3.3. [MEDIUM-HIGH] Anthropic 流式请求同款无 idle 超时悬挂

- 位置:`crates/lc-providers/src/providers/anthropic/chat.rs:553-562`(`stream_chat_internal`)
- 触发:`stream_chat` 流式路径;对端发送后停住。
- 后果:与 3.2 同类——裸 `self.client.post(...).send()` + `bytes_stream()`,client 为 `default_client()`(connect-only 超时)。流中途静止会一直挂住,`ReceiverStream(rx)` 永不结束。**注意**:非流式的 `chat_internal`(440-449)已正确使用 `CHAT_RETRY = SAFE_RETRY`(at-most-once 计费,0.26.0 已修复),问题只在流式路径的空闲超时缺失。

### 3.4. [MEDIUM] Ollama 重试策略:注释写 SAFE_RETRY,代码用 DEFAULT_RETRY

- 位置:`crates/lc-providers/src/ollama/chat.rs:214-223`(注释 214-215)
- 触发:`chat_internal` 发出非幂等、计费 POST(可能已被 HTTP 层派发执行)后返回 5xx/超时。
- 后果:注释声称"use retry::SAFE_RETRY to forbid replaying a possibly-dispatched request",实际代码(223 行)传的是 `DEFAULT_RETRY`。`DEFAULT_RETRY`(`AllTransportErrors`)在超时/5xx 后会**原样重发同一 POST body**,而模型端可能已生成并产生了输出,导致一次对话被计费两次、第一次结果被丢弃。注释与实现不符,担保被夸大,是 incident 级隐患。对比:Anthropic 对同一场景已收敛到 `CHAT_RETRY=SAFE_RETRY`。

### 3.5. [LOW] Ollama 流式路径同样裸 send 无 idle 超时

- 位置:`crates/lc-providers/src/ollama/chat.rs:270-285`(`stream_chat_internal`)
- 触发:流式对话,对端停住。
- 后果:与 3.2/3.3 同类,`default_client()` + `bytes_stream()`,无 SSE idle 超时,静止悬挂。(A12 终态检测 387-394 有,但不覆盖"既不结束也不发数据"的冻结。)

### 3.6. [LOW] Responses `Runnable::stream` 完全忽略采样覆盖

- 位置:`crates/lc-providers/src/openai/responses/model.rs:578-603`
- 触发:经 `stream` 路径传入 `RunnableConfig` 里的 temperature / max_output_tokens 覆盖。
- 后果:该路径未调用 `apply_sampling_overrides`(而 `stream_chat`,741 行,正确调用),传入的采样覆盖被直接丢弃。同一模型 `invoke` 与 `stream` 行为不一致。

### 3.7. [LOW] Assistants 终态消息拉取不受轮询预算约束

- 位置:`crates/lc-providers/src/openai/assistants.rs:310-316`(终态 `get`),以及 `parse`(448-453)
- 触发:轮询循环已消耗大量 `poll_config.timeout` 后,第 5 步取最终消息用了完整的 `REQUEST_TIMEOUT`(180s),不是剩余预算。
- 后果:取消息请求可把整体延迟拉过 `poll_config.timeout` 预算。另 `parse` 先 `.json()` 后判 status,若错误响应不是 JSON,会报 `Parse` 而非带状态码的 `Api`,定位成本略升。

### 3.8. [LOW] Anthropic 错误体读取失败被吞(小)

- 位置:`crates/lc-providers/src/providers/anthropic/chat.rs:456,566`
- 触发:API 返回 4xx/5xx 且读错误体失败。
- 后果:`.unwrap_or_default()` 退化成空串,仅保留状态码(仍以 Api 错误上抛,不静默吞掉整个失败)。属 P1-4 语义的轻微回退,不阻断。

## 4. 不合理 / 可改进

- **错误体读取失败大多吞掉**:anthropic 的 `response.text().await.unwrap_or_default()` 若换显式检查,可保留"读错误体失败"这个独立错误态。
- **Ollama 缺 `max_completion_tokens` 判断**:它构造 OpenAI 系模型但无 o 期路由(毕竟本地模型前缀不是 `o\`),Ollama 上的 o 兼容模型(如通过 OpenWebUI 入口)无法利用路由,低风险。
- **API key 校验时机不一**:部分 provider 构造时不校验空 key(等待请求 401),部分 fail-fast。建议统一为构造期 fail-fast。
- **`responses/model.rs` 的 `apply_sampling_overrides`(178-188)采用 clone-self 模式**:每次请求 clone 一份 self 改 temperature/max_tokens,内存略冗余,但可读性好,可接受。

## 5. 可加的新功能(2026 主流 LLM SDK 角度)

1. **OpenTelemetry / 结构化追踪集成**:把 `lc_callbacks` 的 run 生命周期接入 OTel span,统一 LLM 调用、token、cache_read/cache_created 指标(source: Anthropic/OpenAI 官方均已支持 OTEL 语义)。
2. **结构化的 `LangSmith` / `Langfuse` / `Langwatch` 远端追踪导出器**:当前只有回调处理器,缺少开箱即用的远端 trace 导出。
3. **o 系模型的 reasoning_effort / reasoning 摘要暴露**:OpenAI/OpenRouter 已把 `reasoning_effort` 提升为一等参数,可加入路由与 `gemini` 的 `thought` 一致的 thinking_content 面板。
4. **多 provider 自动故障转移 / 择优路由**:内置"主 → 备"失败切换与延迟打分,匹配业界 provider router 趋势。
5. **开源模型拉取(如 Ollama)的模型是否存在预检**:调用前 `GET /api/tags` 校验,构造期失败而不是 404 流式请求。

## 6. 对标主流框架

- **创始人友好**:`SAFE_RETRY` 的 at-most-once 边界、SSE 有界帧、A12/M-8 终态检测,质量上不输 langchain-js / llama_index(Section 3 的同类问题它们也普遍存在)。尤其 OpenAI 的 `max_completion_tokens` 路由与 Responses `max_output_tokens` 字段选择是**超出多数同行的细节**。
- **差距**:Azure o 系路由缺口(langchain 的 AzureChatOpenAI 已按 model 判断),以及流式空闲超时缺失(多数 SDK 靠基础 client timeout 兜底)。
- **优势**:统一的 `provider_api_client`/`provider_sse_client` 时间策略、跨 provider 一致的错误模型(`ProviderError`)、media 层 SSRF 防护是加分项。

## 7. 一句总评

工程基线高且细节用心(至少一次计费、SSE 有界、key 全脱敏、o 系路由),真正需要修的只有 Azure 的 o 系 max_tokens、Gemini/Anthropic 流式 idle 超时、Ollama 的重试策略三处;修掉即接近可发布基准。
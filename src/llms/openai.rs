use std::collections::HashMap;
use std::error::Error;
use serde::Deserialize;
use crate::prompts::ChatPromptTemplate;
use std::pin::Pin;
use futures_util::{
    stream::{Stream, StreamExt, TryStreamExt},
};

// 辅助类型别名，避免重复写长签名
type TokenStream = Pin<Box<dyn Stream<Item = Result<String, Box<dyn std::error::Error>>> + Send>>;

#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub streaming: bool,
}

#[derive(Debug, Clone)]
pub struct LLM {
    client: reqwest::Client,
    config: OpenAIConfig,
}

impl LLM {
    pub fn new(config: OpenAIConfig) -> Self {
        let client = reqwest::Client::new();
        Self {
            client, config
        }
    }

    // Convenience method for chat with system message
    pub async fn chat(
        &self,
        system_message: Option<&str>,
        human_message: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let mut messages = Vec::new();

        if let Some(system) = system_message {
            messages.push(crate::messages::Message::system(system));
        }
        messages.push(crate::messages::Message::human(human_message));
        self.generate_with_messages(messages).await
    }


    pub async fn invoke_chat_template(
        &self,
        template: &ChatPromptTemplate,
        values: &HashMap<&str, &str>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let messages = template.format(values)
            .map_err(|e| format!("模板格式化失败: {}", e))?;

        self.generate_with_messages(messages).await
    }
    pub async fn invoke_chat_template_stream(
        &self,
        template: &ChatPromptTemplate,
        values: &HashMap<&str, &str>,
    ) -> Result<TokenStream, Box<dyn Error>> {
        let messages = template.format(values)
            .map_err(|e| format!("模板格式化失败: {}", e))?;

        self.stream_with_messages(messages).await
    }

    pub async fn invoke(
        &self,
        content: &str,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.generate(content).await
    }


    pub async fn generate(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let messages = vec![crate::messages::Message::human(prompt)];
        self.generate_with_messages(messages).await
    }

    pub async fn stream_generate(
        &self,
        prompt: &str,
    ) -> Result<impl Stream<Item = Result<String, Box<dyn std::error::Error>>>, Box<dyn std::error::Error>>
    {
        let messages = vec![crate::messages::Message::human(prompt)];
        self.stream_with_messages(messages).await
    }

    pub async fn stream_with_messages(
        &self,
        messages: Vec<crate::messages::Message>,
    ) -> Result<TokenStream, Box<dyn std::error::Error>> {
        if !self.config.streaming {
            return Err("Streaming must be enabled in config".into());
        }

        let url = format!("{}/chat/completions", self.config.base_url);
        let openai_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|m| serde_json::json!({ "role": m.role(), "content": m.content() }))
            .collect();

        let body = serde_json::json!({
        "model": self.config.model,
        "messages": openai_messages,
        "stream": true,
    });

        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        let byte_stream = response.bytes_stream();

        // 使用 unfold 手动管理 buffer 状态
        let token_stream = futures_util::stream::unfold(
            (byte_stream.map_err(|e| Box::new(e) as Box<dyn std::error::Error>).boxed(), String::new()),
            |(mut stream, mut buffer)| async move {
                loop {
                    // 读取下一个字节块
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            // 追加到缓冲区
                            buffer.push_str(&String::from_utf8_lossy(&chunk));

                            // 查找所有完整的行（以 \n 结尾）
                            let mut pos = 0;
                            let mut last_line_start = 0;
                            let mut found_complete_line = false;

                            // 遍历每个字符，找 \n
                            for (i, ch) in buffer.char_indices() {
                                if ch == '\n' {
                                    let line = &buffer[last_line_start..i];
                                    pos = i + 1; // 跳过 \n
                                    last_line_start = pos;
                                    found_complete_line = true;

                                    // 跳过空行
                                    if line.trim().is_empty() {
                                        continue;
                                    }

                                    // 处理 data: 行
                                    if line.starts_with("data: ") {
                                        let data = &line[6..];
                                        if data.trim() == "[DONE]" {
                                            return None; // 流结束
                                        }

                                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                            if let Some(content) = json
                                                .get("choices")
                                                .and_then(|c| c.get(0))
                                                .and_then(|c| c.get("delta"))
                                                .and_then(|d| d.get("content"))
                                                .and_then(|v| v.as_str())
                                            {
                                                if !content.is_empty() {
                                                    // 返回 token，并保留剩余 buffer
                                                    let remaining = buffer[pos..].to_string();
                                                    return Some((
                                                        Ok(content.to_string()),
                                                        (stream, remaining),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // 如果没有完整行，保留整个 buffer
                            if !found_complete_line {
                                // 继续等待更多数据
                                // 注意：这里不能 return，必须继续 loop
                                // 但为了避免无限循环，break 出去让下一次 unfold 调用继续
                                // 所以我们只保留未处理部分
                                let remaining = buffer[last_line_start..].to_string();
                                // 没有可返回的 token，继续下一轮
                                // 但由于 unfold 必须返回 Some 或 None，我们不能在这里阻塞
                                // 所以：如果没有完整行，就 break 并等待下次调用
                                // 但这样会漏掉？不，下一次会带着 remaining 进来
                                // 所以我们只能返回 None？不行！
                                // 正确做法：如果没有 token，就继续读下一个 chunk
                                // 所以不能 break，必须继续 loop
                                // 但这样可能死循环？不会，因为 stream 会结束
                                // 所以继续 loop
                            } else {
                                // 有完整行但没找到有效 token，清理已处理部分
                                buffer = buffer[pos..].to_string();
                            }
                        }
                        Some(Err(e)) => {
                            return Some((Err(e), (stream, String::new())));
                        }
                        None => {
                            // 流结束
                            return None;
                        }
                    }
                }
            },
        );

        Ok(Box::pin(token_stream))
    }


    pub async fn generate_with_messages(
        &self,
        messages: Vec<crate::messages::Message>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let openai_messages: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|m| {
                serde_json::json!({
                "role": m.role(),
                "content": m.content()
            })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": openai_messages,
        });

        // Add streaming parameter if enabled
        if self.config.streaming {
            body["stream"] = serde_json::json!(true);
        }

        let request = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body);

        if self.config.streaming {
            // Handle streaming response
            let mut stream = request.send().await?.bytes_stream();
            let mut full_content = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                let chunk_str = String::from_utf8_lossy(&chunk);

                // Parse Server-Sent Events format
                for line in chunk_str.lines() {
                    if line.starts_with("data: ") {
                        let data = &line[6..]; // Remove "data: " prefix

                        // Check for [DONE] marker
                        if data.trim() == "[DONE]" {
                            break;
                        }

                        // Parse the JSON chunk
                        if let Ok(chunk_json) = serde_json::from_str::<serde_json::Value>(data) {
                            // Extract content from the chunk
                            if let Some(choices) = chunk_json.get("choices") {
                                if let Some(first_choice) = choices.get(0) {
                                    if let Some(delta) = first_choice.get("delta") {
                                        if let Some(content) = delta.get("content") {
                                            if let Some(text) = content.as_str() {
                                                full_content.push_str(text);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Ok(full_content)
        } else {
            // Non-streaming response (original behavior)
            let response: ChatResponse = request
                .send()
                .await?
                .json()
                .await?;
            let first_choice = response.choices.first().ok_or("No choices returned")?;
            Ok(first_choice.message.content.clone())
        }
    }
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}


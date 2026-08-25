//! 示例 / 可部署的 A2A(Agent-to-Agent)HTTP 服务器。
//!
//! 把 LLMChain(真实模型)通过 `A2AServer::serve_on` 暴露为网络 A2A agent,
//! 任何 A2A 客户端(本仓库 `A2AClient` / 其他语言的 A2A SDK)都能按协议调用:
//!
//! | 路由 | 说明 |
//! |---|---|
//! | `GET /.well-known/agent-card.json` | 发现 agent:名字/描述/技能/地址 |
//! | `POST /` | `tasks/send` / `tasks/get` / `tasks/cancel`(JSON-RPC) |
//! | `GET /events` | SSE 任务进度推送(需 `with_streaming` 开启) |
//!
//! # 运行(本地联调)
//!
//! ```powershell
//! cargo run -p langchainrust --example a2a_http_server
//! ```
//!
//! # 构建独立可执行文件(部署用)
//!
//! ```powershell
//! cargo build --release -p langchainrust --example a2a_http_server
//! # 产物:target/release/examples/a2a_http_server.exe,拷到远程服务器即可运行
//! ```
//!
//! # 运行配置(环境变量,不写死在代码里)
//!
//! | 变量 | 默认 | 说明 |
//! |---|---|---|
//! | `A2A_HOST` | `127.0.0.1` | 绑定地址(默认仅本机;远程访问需显式设为 `0.0.0.0`,且必须自行配置鉴权/网络白名单) |
//! | `A2A_PORT` | `8080` | 监听端口 |
//! | `A2A_API_KEY` | 内置 key | 阿里云 MaaS 的 API key |
//! | `A2A_BASE_URL` | 内置地址 | 阿里云 MaaS OpenAI 兼容端点 |
//! | `A2A_MODEL` | `qwen3.7-max-2026-06-08` | 模型名(账号无该模型权限时,换成有权限的,如 `qwen3.5-ocr`) |
//! | `A2A_QUESTION_TEMPLATE` | `用一句话回答:{question}` | 发给 LLM 的提示词模板 |
//! | `A2A_AUTH_TOKEN` | 空 | 设置后所有请求需带 `Authorization: Bearer <token>` |
//!
//! 远程部署时把 `A2A_HOST` 设为 `0.0.0.0` 并配置网络白名单/反向代理鉴权。
//! 启动后 agent 卡片地址是 `http://<host>:<port>/.well-known/agent-card.json`。
//!
//! ```powershell
//! $env:A2A_PORT = "8080"
//! .\target\release\examples\a2a_http_server.exe
//! ```
//!
//! # 用本仓库的客户端测(另开一个进程)
//!
//! ```powershell
//! cargo test -p langchainrust --test a2a f04_remote_http_roundtrip
//! ```

use std::sync::Arc;

use langchainrust::{A2AServer, LLMChain, OpenAIChat, OpenAIConfig};
use tokio::net::TcpListener;

/// 连真实端点的模型:key/地址/模型名都能用环境变量覆盖,默认值一眼能看懂。
fn real_llm() -> OpenAIChat {
    OpenAIChat::new(OpenAIConfig {
        api_key: std::env::var("A2A_API_KEY")
            .unwrap_or_else(|_| "sk-6eb65fcf5d17491ca10b984efe1f43e7".to_string()),
        base_url: std::env::var("A2A_BASE_URL").unwrap_or_else(|_| {
            "https://llm-8xo1b7o30z27y2xc.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
                .to_string()
        }), // 阿里云 MaaS 端点
        model: std::env::var("A2A_MODEL").unwrap_or_else(|_| "qwen3.7-max-2026-06-08".to_string()),
        streaming: false,
        temperature: Some(0.3),
        max_tokens: Some(500),
        ..Default::default()
    })
}

#[tokio::main]
async fn main() {
    // 1. 读取配置(环境变量,带默认值)。默认仅绑定本机回环;该 server 无内置鉴权,
    //    绑 0.0.0.0 会把服务暴露给任何能到达该端口的人,须显式选择并自行加鉴权。
    let host = std::env::var("A2A_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("A2A_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8080);

    // 2. 造 LLMChain(真实模型 + 提示词模板),作为 A2A 服务端的技能
    let template = std::env::var("A2A_QUESTION_TEMPLATE")
        .unwrap_or_else(|_| "用一句话回答:{question}".to_string());
    let chain = LLMChain::new(real_llm(), template);

    // 3. 建 A2A 服务端:包链;可选 bearer 鉴权 token
    let server = A2AServer::new(Arc::new(chain) as Arc<dyn langchainrust::BaseChain>);
    let server = match std::env::var("A2A_AUTH_TOKEN").ok() {
        Some(token) if !token.is_empty() => server.with_auth_token(&token),
        _ => server,
    };

    // 4. 绑定监听地址(默认 127.0.0.1 = 仅本机;0.0.0.0 = 所有网卡,须显式设置)
    let listener = TcpListener::bind((host.as_str(), port))
        .await
        .unwrap_or_else(|e| {
            eprintln!("绑定 {host}:{port} 失败: {e}");
            std::process::exit(1);
        });
    let bound = listener.local_addr().unwrap();

    // 5. 打印入口并开服(axum HTTP 层,进程存活由 serve_on 的请求循环维持)
    println!("A2A agent 已启动 ✅");
    println!("  agent 卡片: http://{bound}/.well-known/agent-card.json");
    println!("  任务入口 : POST http://{bound}/  (tasks/send / tasks/get / tasks/cancel)");
    println!("  技能     : 一句话问答(qwen3.7-max)");
    println!("按 Ctrl+C 停止。");

    server.serve_on(listener).await.unwrap_or_else(|e| {
        eprintln!("A2A 服务异常退出: {e}");
        std::process::exit(1);
    });
}

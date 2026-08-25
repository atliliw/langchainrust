# A2A Agent 服务器部署指南

> 把一个开箱即用的 A2A(Agent-to-Agent)Agent 服务器部署到远程 Linux 服务器,
> 供任意 A2A 客户端(本仓库 `A2AClient` / 其他语言的 A2A SDK)通过
> `http://<服务器IP>:18080/` 发现并调用。
>
> 服务器本体是 `crates/lc/examples/a2a_http_server.rs`,基于框架的
> `A2AServer::serve_on` 实现——客户端看到的协议(agent 卡片 / `tasks/send` /
> `tasks/get` / `tasks/cancel` / SSE)、任务生命周期都来自框架自身,不是另一套
> 实现。以下步骤与 [MCP SSE 服务器部署指南](./mcp-sse-server-deployment.md)
> 完全对应,只是把 MCP 工具换成了 A2A agent(LLMChain)。

---

## 1. 服务器是什么

示例二进制 `a2a_http_server` 把一个 `LLMChain`(真实模型 + 提示词模板)暴露为
网络 A2A agent:

| 路由 | 说明 |
|---|---|
| `GET /.well-known/agent-card.json` | 发现 agent:名字/描述/技能/地址 |
| `POST /` | `tasks/send` / `tasks/get` / `tasks/cancel`(JSON-RPC 2.0) |
| `GET /events` | SSE 任务进度推送(需代码里开 `with_streaming`) |

任务模型是**异步**的:`tasks/send` 立即返回 `submitted`,服务端后台执行 LLM 链
(`submitted -> working -> completed`),客户端轮询 `tasks/get` 拿结果。

> 该 agent 的"技能"就是一句话问答:`LLMChain` 模板
> `用一句话回答:{question}` + 真实模型。模型权限在**服务端**检查——agent 服务
> 器需要能出网访问模型端点(阿里云 MaaS),且账号对所选模型有权限。

## 2. 环境与前置

- 一台 Linux 服务器(本指南:CentOS 7, x86_64),能 SSH 登录(`root`)
- 服务器端口 `18080` 对外可达(客户端能连到;默认绑定 `0.0.0.0` 所有网卡)
- 服务器上装有 Rust 工具链(`rustc`/`cargo`,版本 ≥ 项目 MSRV 1.82)
- **出网**:agent 服务器要能 HTTPS 访问模型端点(阿里云 MaaS
  `*.maas.aliyuncs.com`)。客户端不需要访问模型端点,只访问 agent 的 `18080`。
- **网络注意(中国大陆)**:`static.rust-lang.org` / `crates.io` 直连常被墙,
  需要配置镜像源,见 [第 3.3 节](#33-配置-cargo-镜像源中国大陆)

## 3. 构建

### 3.1 方案 A:服务器上构建(推荐)

把源码传到服务器,在服务器上直接编译,免去交叉编译的麻烦:

```bash
# 1. 把整个仓库拷到服务器(或用 git clone / rsync)
scp -r langchainrust root@192.168.10.100:/opt/a2a-server-src

# 2. 服务器上配置 cargo 镜像 + 构建(见 3.3 / 3.4)
cargo build --release -p langchainrust --example a2a_http_server

# 3. 产物路径
#    target/release/examples/a2a_http_server
```

> **工具链注意**:项目根有 `rust-toolchain.toml` 锁 1.82。若服务器 `rustup
> toolchain install 1.82` 因 GFW 下载失败,可把 `rust-toolchain.toml` 改名
> 为 `rust-toolchain.toml.bak`,改用服务器已装的 stable(≥ 1.82 即可)。

### 3.2 方案 B:本地构建后拷贝

本地(Windows)构建 Linux 二进制需要交叉编译目标 `x86_64-unknown-linux-gnu`
+ glibc 链接器,依赖较复杂。**更稳的做法是在服务器上构建**(方案 A)。
若本地仅做语法/测试验证,直接 `cargo build -p langchainrust --example a2a_http_server`
构建 Windows 版即可(不可拷到 Linux 跑)。

> **Windows 本地构建的坑**:仓库若在网盘同步目录(如 `D:\BaiduNetdiskDownload\`),
> 云盘同步驱动会锁住新生成的 `.dll/.exe`,cargo 链接报 `Permission denied`。
> 构建前设 `CARGO_TARGET_DIR` 到同步目录外:
> ```powershell
> $env:CARGO_TARGET_DIR = "C:\Users\Administrator\rust-target"
> cargo test -p langchainrust --test a2a f05 -- --nocapture
> ```

### 3.3 配置 cargo 镜像源(中国大陆)

`~/.cargo/config.toml`:

```toml
[source.crates-io]
replace-with = "ustc"

[source.ustc]
registry = "sparse+https://mirrors.ustc.edu.cn/crates.io-index/"
```

### 3.4 离线缓存同步(服务器无法联网拉依赖时)

若镜像也不通,可把本地已下载的依赖缓存同步过去再离线构建:

```bash
# 本地:同步 mirror 缓存目录(注意是镜像 hash 目录,不是 index.crates.io-*)
scp -r ~/.cargo/registry/cache/mirrors.ustc.edu.cn-38d0e5eb5da2abae/* \
    root@192.168.10.100:/root/.cargo/registry/cache/mirrors.ustc.edu.cn-38d0e5eb5da2abae/

# 服务器:离线构建
cargo build --offline --release -p langchainrust --example a2a_http_server
```

## 4. 部署

### 4.1 放置可执行文件

```bash
mkdir -p /opt/a2a-server
cp target/release/examples/a2a_http_server /opt/a2a-server/
chmod +x /opt/a2a-server/a2a_http_server
```

### 4.2 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `A2A_HOST` | `127.0.0.1` | 绑定地址;**远程访问必须设 `0.0.0.0`**(所有网卡) |
| `A2A_PORT` | `8080` | 监听端口(本指南用 `18080`) |
| `A2A_API_KEY` | 内置 key | 阿里云 MaaS 的 API key |
| `A2A_BASE_URL` | 内置地址 | 阿里云 MaaS OpenAI 兼容端点 |
| `A2A_MODEL` | `qwen3.7-max-2026-06-08` | **模型名**。该账号当前只有 `qwen3.5-ocr` 有权限(实测能回答);用 `qwen3.7-max` 会后台 `failed`(403 AccessDenied.Unpurchased)。按账号实际情况设,如 `qwen3.5-ocr` |
| `A2A_QUESTION_TEMPLATE` | `用一句话回答:{question}` | 发给 LLM 的提示词模板 |
| `A2A_AUTH_TOKEN` | 空 | 设置后所有请求需带 `Authorization: Bearer <token>`(强烈建议远程部署开启) |

### 4.3 systemd 服务(开机自启 + 崩溃自动拉起)

`/etc/systemd/system/a2a-agent.service`:

```ini
[Unit]
Description=langchainrust A2A Agent Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/a2a-server
ExecStart=/opt/a2a-server/a2a_http_server
Restart=always
RestartSec=3
Environment=A2A_HOST=0.0.0.0
Environment=A2A_PORT=18080
Environment=A2A_MODEL=qwen3.5-ocr
Environment=A2A_AUTH_TOKEN=换成你的随机token
# 若 key/端点不同再追加:
# Environment=A2A_API_KEY=sk-xxxx
# Environment=A2A_BASE_URL=https://.../compatible-mode/v1

[Install]
WantedBy=multi-user.target
```

启动与自启:

```bash
systemctl daemon-reload
systemctl enable --now a2a-agent
systemctl status a2a-agent   # 应显示 active (running)
```

启动日志里会打印入口:
`agent 卡片: http://0.0.0.0:18080/.well-known/agent-card.json`

## 5. 验证

### 5.1 端口与 agent 卡片

```bash
curl -s -m 5 http://192.168.10.100:18080/.well-known/agent-card.json
# {"name":"llm_chain","description":"Agent backed by llm_chain",...}
```

### 5.2 提交任务并轮询(tasks/send -> tasks/get)

```bash
# 提交任务(注意:中文 payload 要存成 UTF-8 文件再 -d @file,否则 Windows
# 终端/部分环境会按 GBK 发出去,服务端按 UTF-8 解析报 invalid utf-8)
cat > /tmp/send.json <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"tasks/send","params":{"message":{"role":"user","content":"什么是 Rust?请一句话回答"}}}
EOF
curl -s -m 15 http://192.168.10.100:18080/ -H "Content-Type: application/json" \
  --data-binary @/tmp/send.json
# -> task.status 为 submitted,记下 task.id

# 轮询直到 completed/failed(把 <taskId> 换成上面的 id)
cat > /tmp/get.json <<'EOF'
{"jsonrpc":"2.0","id":2,"method":"tasks/get","params":{"taskId":"<taskId>"}}
EOF
curl -s -m 15 http://192.168.10.100:18080/ -H "Content-Type: application/json" \
  --data-binary @/tmp/get.json
# -> completed,result.output 是 agent 的回答
```

### 5.3 完整链路:用框架客户端实测(推荐)

仓库里有现成用例,直连已部署服务器完成 发现 agent → 提交任务 → 拿到回答:

```powershell
# 本地(同步目录外构建)
$env:CARGO_TARGET_DIR = "C:\Users\Administrator\rust-target"
$env:A2A_SERVER_URL = "http://192.168.10.100:18080"
cargo test -p langchainrust --test a2a f05 -- --nocapture
```

期望输出:

```
[F05] 目标已部署服务器: http://192.168.10.100:18080
[F05] agent 卡片: name=llm_chain
[F05] ✅ 远程 agent 回答: "Rust 是一种编程语言,以其安全、高效和内存管理能力强"
test f05_deployed_remote_server_roundtrip ... ok
```

> 若服务器开启了 `A2A_AUTH_TOKEN`,客户端需带 token:
> `langchainrust::A2AClient::builder(url).bearer_token("xxx").build()?`
> (见 `crates/lc-a2a/src/server_impl.rs` 测试里的用法)。

## 6. 客户端接入

- **agent 卡片**:`http://<服务器IP>:18080/.well-known/agent-card.json`
- **任务入口**:`http://<服务器IP>:18080/`(`tasks/send` / `tasks/get` / `tasks/cancel`)
- 框架客户端:
  ```rust
  let client = A2AClient::new("http://192.168.10.100:18080".to_string())?;
  let card = client.get_agent_card().await?;                     // 发现 agent
  let result = client.send_task_and_wait(                        // 提交并等到完成
      A2AMessage::user("什么是 Rust?"),
      Duration::from_secs(90),
  ).await?;
  ```
- 其他语言:A2A 协议是 JSON-RPC 2.0,任何语言的 HTTP 客户端都能按 5.2 的
  请求/响应格式接入。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 任务一直 `submitted`/`working` 后变 `failed`,错误含 `403 AccessDenied.Unpurchased` | 服务端账号对该模型无权限;按 4.2 把 `A2A_MODEL` 设为有权限的模型(如 `qwen3.5-ocr`) |
| `POST /` 返回 `invalid utf-8 sequence` | 客户端把中文按非 UTF-8 编码发出;用 `-d @file` 传 UTF-8 文件(5.2) |
| 客户端连不上 `18080` | 服务器没设 `A2A_HOST=0.0.0.0`(只绑了回环),或防火墙/安全组没放行端口 |
| 远程请求报 401 / `Authentication required` | 服务器开了 `A2A_AUTH_TOKEN`,客户端没带 `Authorization: Bearer <token>`(5.3 末) |
| `rustup toolchain install 1.82` 卡在 TLS | GFW 挡 `static.rust-lang.org`;改名 `rust-toolchain.toml` 用服务器已装 stable |
| `cargo build` 拉依赖超时 | 配 USTC 镜像(3.3)或离线缓存(3.4) |
| Windows 本地 cargo 链接 `Permission denied` | 网盘同步目录锁文件;设 `CARGO_TARGET_DIR` 到同步目录外(3.2) |
| 想换技能/加 SSE | 编辑 `crates/lc/examples/a2a_http_server.rs`(换模板 / `with_streaming`),重新构建部署 |

---

相关代码:`crates/lc/examples/a2a_http_server.rs`、`crates/lc/tests/a2a.rs`(F05)。
框架实现:`lc-a2a` 的 `A2AServer::serve_on` / `handle_a2a_request` 与 `A2AClient`。

# MCP SSE 服务器部署指南

> 把一个开箱即用的 MCP 服务器部署到远程 Linux 服务器,供任意 MCP 客户端
> (`MCPClient` / Cursor / Claude Desktop 等)通过 `http://<服务器IP>:8788/sse`
> 连接并调用工具。
>
> 服务器本体是 `crates/lc/examples/mcp_sse_server.rs`,基于框架的
> `MCPServer::serve_sse` 实现——客户端看到的协议、握手、工具都来自框架自身,
> 不是另一套实现。以下步骤是**一次真实部署**(目标:CentOS 7 / glibc 2.17 /
> x86_64,`192.168.10.100`)的完整记录,命令可直接照做。

---

## 1. 服务器是什么

示例二进制 `mcp_sse_server` 把 langchainrust 的 6 个内置工具暴露为网络 MCP Server:

| 工具 | 说明 |
|---|---|
| `calculator` | 数学表达式求值(支持函数/常量) |
| `math` | 简单数学 |
| `datetime` | 日期时间查询/计算 |
| `url_fetch` | 抓取网页内容 |
| `wikipedia` | Wikipedia 查询 |
| `web_search` | DuckDuckGo 网页搜索 |

> 刻意**不注册** `PythonREPLTool`(远程可执行任意代码,有安全风险)。需要更多
> 工具时在 `examples/mcp_sse_server.rs` 里加 `.with_tool(...)` 重新构建即可。

## 2. 环境与前置

- 一台 Linux 服务器(本指南:CentOS 7, x86_64),能 SSH 登录(`root`)
- 服务器端口 `8788` 对外可达(客户端能连到)
- 服务器上装有 Rust 工具链(`rustc`/`cargo`,版本 ≥ 项目 MSRV 1.85)
- **网络注意(中国大陆)**:`static.rust-lang.org` / `crates.io` 直连常被墙,
  需要配置镜像源,见 [第 3.3 节](#33-配置-cargo-镜像源中国大陆)

## 3. 构建

### 3.1 方案 A:服务器上构建(推荐,本次实际采用)

把源码传到服务器,在服务器上直接编译,免去交叉编译的麻烦:

```bash
# 1. 把整个仓库拷到服务器(或用 git clone / rsync)
scp -r langchainrust root@192.168.10.100:/opt/mcp-server-src

# 2. 服务器上配置 cargo 镜像 + 构建(见 3.3 / 3.4)
cargo build --release -p langchainrust --example mcp_sse_server

# 3. 产物路径
#    target/release/examples/mcp_sse_server
```

> **工具链注意**:项目根有 `rust-toolchain.toml` 锁 1.85。若服务器 `rustup
> toolchain install 1.85` 因 GFW 下载失败,可把 `rust-toolchain.toml` 改名
> 为 `rust-toolchain.toml.bak`,改用服务器已装的 stable(≥ 1.85 即可,项目
> MSRV 是 1.85)。

### 3.2 方案 B:本地构建后拷贝

本地(Windows)构建 Linux 二进制需要交叉编译目标 `x86_64-unknown-linux-gnu`
+ glibc 链接器,依赖较复杂。**更稳的做法是在服务器上构建**(方案 A)。
若本地仅做语法/测试验证,直接 `cargo build -p langchainrust --example mcp_sse_server` 构建 Windows 版即可(不可拷到 Linux 跑)。

> **Windows 本地构建的坑**:仓库若在网盘同步目录(如 `D:\BaiduNetdiskDownload\`),
> 云盘同步驱动会锁住新生成的 `.dll/.exe`,cargo 链接报 `Permission denied`。
> 构建前设 `CARGO_TARGET_DIR` 到同步目录外:
> ```powershell
> $env:CARGO_TARGET_DIR = "C:\Users\Administrator\rust-target"
> cargo test -p langchainrust --test mcp f10 -- --nocapture
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

若服务器能访问 USTC 镜像则直接构建即可。若镜像也不通,可把本地已下载的
依赖缓存同步过去再离线构建:

```bash
# 本地:同步 mirror 缓存目录(注意是镜像 hash 目录,不是 index.crates.io-*)
scp -r ~/.cargo/registry/cache/mirrors.ustc.edu.cn-38d0e5eb5da2abae/* \
    root@192.168.10.100:/root/.cargo/registry/cache/mirrors.ustc.edu.cn-38d0e5eb5da2abae/

# 服务器:离线构建
cargo build --offline --release -p langchainrust --example mcp_sse_server
```

## 4. 部署

### 4.1 放置可执行文件

```bash
mkdir -p /opt/mcp-server
cp target/release/examples/mcp_sse_server /opt/mcp-server/
chmod +x /opt/mcp-server/mcp_sse_server
```

### 4.2 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MCP_SERVER_HOST` | `127.0.0.1` | 绑定地址(默认仅本机;**远程部署必须显式设为 `0.0.0.0`**) |
| `MCP_SERVER_PORT` | `8788` | 监听端口 |
| `MCP_SERVER_PUBLIC_URL` | 无 | **远程部署必须设置**;否则服务端发给客户端的回连 POST 地址写成 `0.0.0.0`,客户端连不上 |

### 4.3 systemd 服务(开机自启 + 崩溃自动拉起)

`/etc/systemd/system/mcp-sse-server.service`:

```ini
[Unit]
Description=langchainrust MCP SSE Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/mcp-server
ExecStart=/opt/mcp-server/mcp_sse_server
Restart=always
RestartSec=3
Environment=MCP_SERVER_HOST=0.0.0.0
Environment=MCP_SERVER_PORT=8788
Environment=MCP_SERVER_PUBLIC_URL=http://192.168.10.100:8788

[Install]
WantedBy=multi-user.target
```

启动与自启:

```bash
systemctl daemon-reload
systemctl enable --now mcp-sse-server
systemctl status mcp-sse-server   # 应显示 active (running)
```

> `MCP_SERVER_PUBLIC_URL` 换成你的实际地址:`http://<服务器IP或域名>:8788`。
> 启动日志里会打印客户端连接入口:`客户端连接入口: http://<host>:<port>/sse`。

## 5. 验证

### 5.1 端口与首页

```bash
curl -s -m 5 -o /dev/null -w "HTTP %{http_code}\n" http://192.168.10.100:8788/
# HTTP 200
```

### 5.2 SSE 握手入口(长连接,看到 endpoint 即可 Ctrl+C)

```bash
curl -N http://192.168.10.100:8788/sse
# event: endpoint
# data: http://192.168.10.100:8788/message
#(随后是心跳;exit 28 超时/EOF 属正常,说明链路通)
```

### 5.3 完整链路:用框架客户端实测(推荐)

仓库里有现成用例,直连已部署服务器完成 握手 → 列工具 → 真调工具:

```powershell
# 本地(同步目录外构建)
$env:CARGO_TARGET_DIR = "C:\Users\Administrator\rust-target"
cargo test -p langchainrust --test mcp f10 -- --nocapture
```

期望输出(6 个工具 + 真实计算结果):

```
[F10] 握手成功,协议版本: Some("2024-11-05")
[F10] 远程列出 6 个工具: ["calculator", "math", "datetime", "url_fetch", "wikipedia", "web_search"]
[F10] 远程 calculator('2 + 3 * 4') => 2 + 3 * 4 = 14
test f10_deployed_remote_server_roundtrip ... ok
```

> 地址默认 `http://192.168.10.100:8788/sse`,可设环境变量 `MCP_SERVER_URL`
> 覆盖,指向你自己的服务器。

## 6. 客户端接入

- **SSE 入口**:`http://<服务器IP>:8788/sse`
- 框架客户端:
  ```rust
  let client = MCPClient::connect(MCPConfig::sse("http://192.168.10.100:8788/sse")).await?;
  let tools = client.list_tools().await?;       // tools/list
  let result = client.call_tool("calculator", json!({"expression": "2 + 3 * 4"})).await?;
  ```
- 其它宿主(Cursor / Claude Desktop 等)把上面对接进 MCP 配置即可。

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `rustup toolchain install 1.85` 卡在 TLS | GFW 挡 `static.rust-lang.org`;改名 `rust-toolchain.toml` 用服务器已装 stable |
| `cargo build` 拉依赖超时 | 配 USTC 镜像(3.3)或离线缓存(3.4) |
| 客户端报"连接 POST 地址失败" | `MCP_SERVER_PUBLIC_URL` 没设置,服务端发了 `0.0.0.0`;按 4.2 设置 |
| 客户端列出的工具不是 6 个 | 改了 `examples/mcp_sse_server.rs` 后忘了重新构建/部署 |
| Windows 本地 cargo 链接 `Permission denied` | 网盘同步目录锁文件;设 `CARGO_TARGET_DIR` 到同步目录外(3.2) |
| 想加/减工具 | 编辑 `crates/lc/examples/mcp_sse_server.rs` 的 `build_server()`,重新构建部署 |

---

相关代码:`crates/lc/examples/mcp_sse_server.rs`、`crates/lc/tests/mcp.rs`(F10)。
框架实现:`lc-mcp` 的 `MCPServer::serve_sse` 与 `MCPClient`。

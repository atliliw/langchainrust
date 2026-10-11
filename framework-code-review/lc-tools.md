# lc-tools 代码审查报告

## 1. 概览

`crates/lc-tools` 是 langchainrust 的内置工具库(版本 0.26.0,默认 feature 为空)。src/ 下 26 个 `.rs`(含 3 个子目录模块)全部通读。工具族涵盖:`expr_eval`(去 meval 化的表达式求值)、`calculator`/`math`(计算)、`datetime`、`search` + `hosted_search/{tavily,serper,exa}`(联网搜索)、`url_fetch` + `extended/http`(带 SSRF,实现在 lc-core)、`wikipedia`、`extended/sql`(只读 SQLite)、`extended/file`(沙箱文件)、`sandbox/{local,mod}`(代码执行)、`browser`(CDP,feature-gated)、`extended/computer/{actions,screen,native}`(computer use,feature-gated)、`skills`(SKILL.md 加载)、`python_repl`(内核 python REPL)。

整体质量高:注释诚实标注了"noise filter 不是安全边界",程序化宏/feature 分级清晰,绝大多数代码有 `#[cfg(test)]` 且有针对性测试。发现的问题集中在:**顽固的 SSRF 端态复查(非真 per-hop)**、**SQL 数字列静默变空串**、**LocalSandbox 与 PythonREPL 的过滤器强度不一致**。逐条见下。

## 2. 遍历文件清单(26)

```
crates/lc-tools/src/lib.rs
crates/lc-tools/src/expr_eval.rs
crates/lc-tools/src/calculator.rs
crates/lc-tools/src/math.rs
crates/lc-tools/src/datetime.rs
crates/lc-tools/src/search.rs
crates/lc-tools/src/wikipedia.rs
crates/lc-tools/src/url_fetch.rs
crates/lc-tools/src/python_repl.rs
crates/lc-tools/src/skills.rs
crates/lc-tools/src/ssrf.rs
crates/lc-tools/src/sandbox/mod.rs
crates/lc-tools/src/sandbox/local.rs
crates/lc-tools/src/browser/mod.rs
crates/lc-tools/src/hosted_search/mod.rs
crates/lc-tools/src/hosted_search/tavily.rs
crates/lc-tools/src/hosted_search/serper.rs
crates/lc-tools/src/hosted_search/exa.rs
crates/lc-tools/src/extended/mod.rs
crates/lc-tools/src/extended/file.rs
crates/lc-tools/src/extended/http.rs
crates/lc-tools/src/extended/sql.rs
crates/lc-tools/src/extended/computer/mod.rs
crates/lc-tools/src/extended/computer/actions.rs
crates/lc-tools/src/extended/computer/screen.rs
crates/lc-tools/src/extended/computer/native.rs
```

## 3. bug / 高风险点(按严重度)

### HIGH — B1:SQL 工具数字列静默输出空串
`crates/lc-tools/src/extended/sql.rs:163-166`
```rust
let val: String = row
    .get::<_, Option<String>>(i)
    .unwrap_or(None)          // 吞掉所有 FromSql 错误
    .unwrap_or_default();     // → ""
```
触发:`SELECT id FROM users` 这类含 `INTEGER/REAL/BLOB` 列的查询(rusqlite 的 `String: FromSql` 只接受 `Text`,整数列返回 `InvalidType` 错误)。
后果:该 `get` 返回 `Err`,被 `.unwrap_or(None)` 静默吞掉 → 数字/long/bool 列一律变成 `""`,**无任何报错**、数据静默丢失。现有测试只断言 TEXT 列(name),因此未暴露。修法:`row.*get::<Option<rusqlite::types::Value>>(i)` 再按 `Value` 序列化,或对 `Integer/Real` 作显式类型转换。

### MEDIUM — B2:LocalSandbox 的 Python 过滤器漏了 `open` 等文件逃逸
`crates/lc-tools/src/sandbox/local.rs:73-76`
```rust
BLOCKED_PYTHON_CALL_REGEX = \b(?:__import__\s*\(|importlib\b|exec\s*\(|eval\s*\(|compile\s*\(|__builtins__)
```
· 已修复(与任务核验点一致):`__import__`/`importlib`/`exec`/`eval`/`compile`/`__builtins__` 这些绕过项现已被 call 正则捕获(LocalSandbox 与 python_repl 两者一致);opt-in 门默认关闭(`allow_dangerous: false`,local.rs:173 默认值),危险操作由 `with_dangerously_allow(true)` 才放行 —— **该核验点成立**。
· 遗留漏洞(不一致):**`BLOCKED_PYTHON_CALL_REGEX` 没有 `open`/`input`/`breakpoint`/`execfile`**,而同仓库的 `python_repl.rs:80-90` `DANGEROUS_BUILTIN_CALLS` 却包含这些。因此 `LocalSandbox.run("open('/etc/passwd').read()", Python, _)` 在默认配置下**实际执行并读到任意文件**,而 PythonREPLTool 会拦下。子串扫描也被绕:`import   os`(多空格)、`import\tos` 因不含精确子串 `"import os"` 而漏过(local.rs:55)。
触发:非受信代码含 `open(...)`/多空格 import。
后果:sandbox 的"默认禁危险"承诺弱于同一能力域里的兄弟工具,文档虽诚实标注 noise filter,但强度在本 crate 内不自洽(防御应同层,error 语义却不同)。

### MEDIUM — B3:computer native 的 `wait` 用 `std::thread::sleep` 阻塞 async 执行器
`crates/lc-tools/src/extended/computer/native.rs:135`
```rust
let ms = input.duration_ms.unwrap_or(1000).min(60_000);
std::thread::sleep(std::time::Duration::from_millis(ms));
```
触发:Native 模式任一 `{"action":"wait","duration_ms":N}`。
后果:`execute` 是 async 但此调用阻塞所在 Tokio worker 线程;current-thread 运行时直接卡死整条事件循环,多线程运行时占用一个 worker 多达 60 秒,并发的其它工具/任务被拖慢。应改 `tokio::time::sleep(...).await`。

### MEDIUM — B4:"per-hop" SSRF 复查实为"端态复查",非真 per-hop
`crates/lc-tools/src/browser/mod.rs:523-586`
调用方 URL 只在 `invoke` 开头验一次(`validate_target(&input.url)`,line 524);导航后**只读一次 `location.href`** 并对**最终落地 URL** 验一次(line 582-584,标注为"H5 per-hop SSRF recheck")。但 `Page.navigate` 是一次性完成的重定向:Chrome 在 `loadEventFired` 前已经内部跟完所有 30x 和加载期 meta-refresh。
触发:公网 URL `A(公开)→ 中间跳 30x → B(内网)→ 又跳回 `C(公开)`,或落地后由 setTimeout/JS `location.href` 延迟跳到内网。
后果:
· 中间那条内网跳转**已被浏览器实际访问**(内网探测/SSRF 一次打进),端态复查拦不住;· 落地后才发生的 JS 延迟导航(晚于单次 `current_url()` 读取/wait_ms)在复查窗口之外,同样穿透。
这是对任务核验点"Chrome 自跟随 30x/meta-refresh 后是否 per-hop 防内网"的**否定答案**:实现的是 end-state 门,不是 `Network.requestWillBeSent`/`Page.frameRedirected` 式的逐跳门。若要真正 per-hop,需用 CDP（禁用 `followRedirects` 手动逐跳,或拦截 `requestWillBeSent` 逐请求验 IP）。

### LOW-MED — B5:CdpBrowser 的公开方法完全绕过 SSRF
`crates/lc-tools/src/browser/mod.rs:315-384`
`CdpBrowser::navigate/extract_text/extract_links/metadata` 都是 `pub async fn`,自身无 scheme/SSRF 校验;校验只在高一层 `CdpBrowserTool`(BaseTool 路径)里。模块文档示例正是 `CdpBrowser::connect(...).navigate(...)`(line 13-26 与 19-27 的 `no_run` demo)。触发:调用方直接使用 `CdpBrowser`(文档路径)。后果:这条 API 面无任何 SSRF 防护、无端态复查。建议把 SSRF 校验下沉到 `CdpBrowser` 公共方法或文档显式标注"仅 tool 层有防护"。

### LOW — B6:browser `connect` 的 PUT /json/new 由 reqwest 默认跟随重定向
`crates/lc-tools/src/browser/mod.rs:96-103` `reqwest::Client` 默认跟 30x。debugger 端点是运维方提供的本机地址,风险低;若被恶意外层代理劫持,`/json/new` 可被带到任意地址。不影响主链路,列为低。

## 4. 不合理 / 可改进

- **Native 截图只取第一块显示器**:`native.rs:233 capture_screenshot()` 用 `Monitor::all().first()`,与 `new_native(w,h)` 的宽高无关;坐标校验用**用户自报的 w/h**而非真实分辨率(`native.rs:176`)。若 `new_native(4,4)` 而真实桌面 1920×1080,`click(900,y)` 会被拒(本可点),或 `click(3,3)` 通过校验后 `move_mouse` 的实际效果与预期不符。测试注释已承认。设计上应把真实分辨率当坐标上界。
- **LocalSandbox JS 分支注释/空格绕过**:`contains_dangerous_javascript`(local.rs:130-151)只跳过行首 `//`,`require ('fs')`(module 名与左括号间有空格)绕过 module 黑名单,且 call 正则只有 `spawn(/exec(`、没有 `require` 令牌 → `require ('fs').readFileSync(...)` 穿透。与 B2 同类,强度低于应然。
- **datetime 加月跨界到月末**:`datetime.rs:139-145` `with_month` 在 2024-01-31 +1 月 = Feb 31 时返回 None → 报 "month calculation failed",而非钳制到 2 月末;月末日加月/减月这类常见语义会失败。低。M-27 的负 delta div_euclid 修得对。
- **wikipedia `lang` 未校验**、直接拼进 hostname(`wikipedia.rs:78-81,126-128`),`top_k` 未钳制。因落在 `.wikipedia.org` 后缀内,不构成内网 SSRF,但畸形 lang 会产生费解错误、超大 top_k 会放大输出。低。
- **serper 打分分母含被跳过的行**(`serper.rs:79` `n=items.len()` 而非有效行数),分数轻微压缩;测试注释已标注。仅美观。
- **url_fetch 的链接提取不解析相对 URL**(`url_fetch.rs:28-31` `extract_unique_links` 原样返回 href)。HTML 属性顺序/大小写/单引号等用正则也会漏;浏览器工具用 DOM 解绝对链接,但 URLFetchTool 返回相对链接原样。低。

## 5. 可加的新功能(与主流对齐)

- **真正 per-hop 的浏览器 SSRF**:用 CDP `Network.requestWillBeSent`/`DisablePageFollowRedirects` 逐跳验 IP,替换 B4 的端态复查——这是 internet-facing browser 工具与 LangChain `browser_use`/Playwright SSRF 工具的主要差距。
- **Sandbox 真实隔离后端回归**:`sandbox/mod.rs:20-23` 已移除空心 Wasm/E2B 后端,文档明确"回来时会实现"。可优先补一个**容器后端**(firecracker/podman)做真隔离,或接入 WASM(WASI)把 subprocess 换成纯内存运行——补齐 `LocalSandbox` 从 noise filter 到真安全边界的落差。
- **Python 执行补 `open`/`input` 防御后**把 LocalSandbox 与 PythonREPL 的黑名单合并为一处共享常量(消除 B2 的不自洽)。
- **OpenAI 工具格式/工具结果引用**:框架有 `lc-core::tools::Tool/ToolError`,可加一个 `parameters`(JSON Schema)工具接口对齐 OpenAI/Anthropic 工具调用协议,便于直接喂给两家 API(目前 computer 的 Anthropic 载荷是手写 JSON,天然是起点)。
- **computer native 加 `click` 前坐标截屏回显**:Anthropic computer-use beta 的标准反馈是截图确认坐标命中;本地模式可在点按后自动补一张截图提高闭环质量。

## 6. 对标主流

- **SSRF**:`url_fetch`/`extended/http` 经 lc-core `guarded_get` 做 DNS 解析→逐跳验 IP+IP 钉扎+手动跟重定向,已到主流服务端(如 Shopify/Python `requests` 加固)水准;真正拖后腿的是 browser 的端态复查(B4)与 CdpBrowser 公开面绕行(B5)。
- **Python 沙箱**:本地 subprocess+timeout 是 OpenAI `PythonREPLTool`/LangChain `PythonREPL` 的拷贝,但两者都只把它当"能跑代码"的便捷工具、文档反复强调非隔离。主流是 E2B/Pyodide/真容器;这里按诚实标注处理,方向正确。
- **computer use**:Anthropic API 载荷 + 自研 native 本地驱动(enigo/xcap)双轨,与 Anthropic `computer-use-2025-01-24` 规范对齐,结构优于多数盲转发实现;缺块监控/审计(每次动作记录日志)。
- **Search 后端抽象**(SearchBackend trait + 统一 SearchResult/rank 归一)设计与 LangChain 多 provider 接口同级,tavily/serper/exa 解析与测试干净。

## 7. 一句总评

lc-tools 是一个**工程纪律良好**(诚实标注重叠的安全边界、feature 分级、测试覆盖充分)的工具库,核心 bug 集中在三处现实的低层缺漏——SQL 数字列静默成空串、browser 的 SSRF 只是"端态复查"而非真 per-hop、以及 LocalSandbox 与 PythonREPL 过滤器强度不自洽——均属可控范围内,不构成系统性崩塌。
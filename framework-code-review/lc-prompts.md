# lc-prompts 代码审阅报告

> 审阅性质:只读,未改动任何源码。审阅对象:`crates/lc-prompts/src/` 全部功能性 `.rs`(7/7)。
> 本次审阅的核心验证目标四:模板/占位符解析是否自洽一致、有无注入面、有无死代码/半成品。诚实结论:解析器在 `PromptTemplate` 与 `ChatPromptTemplate` 之间已由单一 `template_parser.rs` 收敛(无漂移),转义({{/}})与 CJK/下划线变量命名一致,对"未闭合括号/非法变量名"按**字面量**处理(不报错、不吞 `MissingVariable` 语义);未发现实际 bug 与注入面;仅少数边界/设计取舍见第四节。

---

## 一、概览

lc-prompts 是提示模板层,四块:

1. **占位符解析内核** `template_parser.rs`:`{name}` 变量 + `{{`/`}}` 转义,一次性 parse 成 `TemplateSegment`,`format` 时走段而非逐 token 重扫;`PromptTemplate`/`ChatPromptTemplate` 共用它,杜绝两处实现漂移。
2. **两种模板**:`PromptTemplate`(字符串)/`ChatPromptTemplate`(多消息),都实现 `Runnable` 以接入 LCEL `pipe`。
3. **few-shot**:`FewShotPromptTemplate` + `ExampleSelector` + `LengthBasedExampleSelector`。
4. **版本化注册表** `registry.rs`:namespace 下单调 1 起版本号 + 内容寻址 SHA-256(SHA-256 前缀解析,最短 7 位)+ 不可变回滚(新版本重发历史内容,审计链不破坏)。

## 二、遍历文件清单(7/7,全部读完)

- `crates/lc-prompts/src/lib.rs` — 模块声明与 re-export
- `crates/lc-prompts/src/error.rs` — `PromptsError`(MissingVariable / InvalidNamespace / …/ AmbiguousHash)
- `crates/lc-prompts/src/template_parser.rs` — `parse_template` / `format_template` / `template_variables`
- `crates/lc-prompts/src/prompt_template.rs` — `PromptTemplate`
- `crates/lc-prompts/src/chat_prompt_template.rs` — `ChatPromptTemplate`
- `crates/lc-prompts/src/few_shot.rs` — `FewShotPromptTemplate` / `ExampleSelector` / `LengthBasedExampleSelector`
- `crates/lc-prompts/src/registry.rs` — `PromptRegistry` / `VersionSpec` / `resolve_hash_version`

## 三、bug/高风险点

**整体结论:未发现 bug(含高危 = 0)。** 三个验证面逐一交代:

### 3.1 解析一致性:两模板共享内核,无漂移
- `parse_template`(`template_parser.rs:31`)的变量名规则为"字母/`_` 开头,后接字母/数字/`_`",`is_alphabetic/is_alphanumeric` 覆盖中文(CJK);`{{`→`{`、`}}`→`}` 转义对两模板一视同仁。
- `PromptTemplate::new`(`prompt_template.rs:35`)与 `ChatPromptTemplate::new`(`chat_prompt_template.rs:40`)都调 `parse_template`,构造期各 parse 一次,`format` 只走 `TemplateSegment` 替换(`format_template`,`template_parser.rs:99`)。
- 一致性验证:转义用例、CJK 变量、`{123}` 非法名当字面量、未闭合 `{unclosed` 当字面量——均有两模板对称测试(`template_parser.rs:156-212`、`prompt_template.rs:142-159`、`chat_prompt_template.rs:210-220`)。**无漂移、无半成品。**

### 3.2 注入面:无
- 模板值是**原样代入**用户/上下文数据(`format_template`,`template_parser.rs:108`),不做 HTML/Shell/SQL 转义——对 LLM 提示模板这是**正确行为**(提示就是要原始文本进模型),不存在把用户值当可执行指令解析的面。唯一"注入"关切是"占位符变量名混入模板字节"：非法名被当字面量(`template_parser.rs:63`),不会产生新变量;拼写错的 `{nmae}` 会静默变成字面 `{nmae}`(见 4.1 讨论),而非被当作占位符执行——因此无解析注入。
- `registry.rs` 的 `resolve_ref` 用 `rsplit_once('@')`(`registry.rs:395`)切 `namespace@spec`,namespace 明文校验 `[A-Za-z0-9._-]` 且禁止空段(`validate_namespace`,`registry.rs:345`);非法引用如 `"ns@v2"`、`"bad namespace@1"`、`""` 均报 `InvalidReference`(`registry.rs:394-417` 测试 602-607)。无路径/命令注入面。

### 3.3 死代码/半成品:无
- `ChatPromptTemplate::variables`(77)用 HashSet 去重但**丢弃顺序**(`seen.into_iter()`):顺序不定,只用于"是否有该变量"这类成员判断,文档已写"deduplicated"。无死代码、无未连通的 `pub`。
- `registry.rs` 的 `insert_test_record`(306-327)是 `#[cfg(test)]` 专用 seam,生产不含;`resolve` 里 `VersionSpec::Latest` 的分支 `expect("namespace always has at least one version")`(`registry.rs:209`)被文档+先前的 `NamespaceNotFound` 守卫保证安全,不是会炸的 `unwrap`。

---

## 四、不合理 / 可改进

### 4.1 [INFO] 非法/未闭合占位符被静默当字面量——拼写错的陷阱
- **触发**:模板里手写 `{nmae}`(名字拼错)或 `{unclosed`(忘闭合)。`parse_template`(63-67、79-81)把这类当**字面文本**输出,`format` 不报任何缺变量错。
- **后果**:配一个错误的变量名,产物是字面量花括号泄漏进提示,LLM 仍能正常响应——错误被*悄悄吞掉*。对 "fail-fast"理念(doc 注释多处强调 Q3/Q4 让遗漏`{answer}` 必须报错)有的地方非常严(见 4.2 few_shot suffix),但**模板主体**却默认宽松静默。**可改进**:给 `parse_template` 加一个 `strict` 变体或 `parse` 时对"形如 `{...}` 但名字非法"记 warning,让严模板作者能开 fail-fast。
- 说明:这不是 bug(文档明说按字面量处理、有测试锚定),是严谨度取舍。

### 4.2 [INFO] `FewShotPromptTemplate.format` 的变量校验口径略微不一
- 对 `input_variables` 显式逐个校验(`few_shot.rs:226-230`);对 **suffix** 走 `PromptTemplate::new(self.suffix).format(variables)`(`few_shot.rs:270`),这会让"suffix 里出现了 `input_variables` 之外的 `{answer}`"报 `MissingVariable`(对应 Q3 测试 360-376,是**刻意从严**,好事)。
- 但两条路径都在 `format` 运行时才校验,**没有**在构造期(`new`,`few_shot.rs:181`)就校验 `input_variables` 与 suffix 的占位符集合是否一致。构造期给一份"需要哪些变量"的提示可以更早暴露配置错。属可改进,非缺陷。

### 4.3 [INFO] `LengthBasedExampleSelector` 用字节长度而非字符数控量
- `select_examples`(69-79)与 `select_examples_by_length`(99-113)都用 `String::len()`(UTF-8 字节),而结构体 doc 说"最大文本长度(字符)"(`few_shot.rs:47`)。对含中文的提示,每个 CJK 字 3 字节,导致实际能塞进的示例**比按字符算的少**(低估预算、偏保守,不会超限放漏)。
- 方向安全(宁少勿多),但 `len()`(字节)语义与"字符"文档不符;**建议 `chars().count()`** 以匹配 doc。另外 `select_examples` 与 `select_examples_by_length` 两块"遍历选满"逻辑高度重复("占位符+真实格式化长 度")。注意 `FewShotPromptTemplate::format` 总是走 `select_examples_by_length`(`few_shot.rs:239`),所以基实现 `select_examples` 在 FewShot 路径下沉为"可直接调用的公共默认",非死代码但几乎未被内部用。

### 4.4 [INFO] `ChatPromptTemplate.format` 的错误类型被压成 `MissingVariable`
- `chat_prompt_template.rs:63-65`:`format_template(...).map_err(|e| PromptsError::MissingVariable(...))`。因 `format_template` 只可能返回 `MissingVariable`,`ChatPromptTemplate` 层重包后仍是 `MissingVariable`,语义未丢失——但重包丢了原参数名结构,错误信息变 `"{e} in message {msg:?}"`。目前安全,仅提示:若未来 `format_template` 新增错误变体,这里会静默改口。

### 4.5 [INFO] `Runnable` invoke 每调用一次分配一张引用 Map
- `prompt_template.rs:88-91`/`chat_prompt_template.rs:119-122`:把 `HashMap<String,String>` 转成 `HashMap<&str,&str>` 再交 `format`。只读一次调用,开销可忽略;若管道内高频调用,可考虑把 `format` 的私有签名改成返回值懒借,属微优化。

---

## 五、可加的新功能

1. **占位符未匹配预警(呼应 4.1)**:提供 `PromptTemplate::unused_or_malformed()` 校验,构造期/CI 时报"拼错的变量、未闭合花括号"。
2. **构造期变量集合预检(呼应 4.2)**:`FewShotPromptTemplate::new` 直接 diff 出"`input_variables` 需要的 vs suffix/示例实际出现的"变量,尽早暴露配置漂移。
3. **模板 diff / 变更影响面**:`registry.rs` 已有内容寻址哈希;可加"两个版本模板变量集合是否兼容"的 `compatible_with(prev)` 检查,在 `rollback`/`register` 时提示占位符变化对下游 `fetch` 的影响。
4. **模板缓存/渲染统计**:对大项目,可给 `PromptTemplate::format` 加一个可选的计次/耗时开关,支撑模板预热与性能观测。

## 六、对标主流

- **对齐**:`{var}` + `{{`/`}}` 转义、CJK 变量、LCEL `pipe` 即 LangChain PromptTemplate / ChatPromptTemplate 语义;`registry.rs` 的 namespace + 版本号 + 内容哈希即 PromptHub/OpenAI Systems 的 prompt 版本管理思路,且实现了**不可变回滚**+**SHA-256 前缀寻址**(多前几位段可 doc/CI 校验,非常少见,属加分项)。
- **略逊/差别**:主流(如 LangSmith)有云端/serde 持久化的 registry 与回滚 UI;本 crate 是内存 `RwLock<BTreeMap>` 单机态,无序列化落盘(`registry.rs` 结构不含 serde derive),跨进程/跨机器不可用——对纯库是合理取舍,但若要"模板即配置"需自接存储。
- **胜过**:单一解析内核(两模板零漂移)+ 内容寻址只读注册,设计与可测试性干净;`ambiguous hash` 明确报错而不猜测,属审慎工程。

## 七、一句总评

**无 bug、无注入面、无死代码**:解析在两模板间由单一内核收敛,转义/命名/CJK 与注册表寻址全部自洽且有测试锚定;唯一值得动手的是把"非法/未闭合占位符静默当字面量"补一个可选的 fail-fast(warning/strict)路径,并统一 `LengthBasedExampleSelector` 的字节/字符计数口径。
// lc-tools/src/wikipedia.rs
//! Wikipedia search tool
//!
//! Searches and fetches encyclopedia entry content via the Wikipedia API.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use lc_core::tools::{BaseTool, Tool, ToolError};

/// Wikipedia tool input
#[derive(Debug, Deserialize, JsonSchema)]
pub struct WikipediaInput {
    /// The search query
    pub query: String,
    /// Number of results to return (default: 3)
    pub top_k: Option<usize>,
    /// Language (default: zh, supports en/zh/ja, etc.)
    pub lang: Option<String>,
    /// Whether to fetch full content (default false, summary only)
    pub full_content: Option<bool>,
}

/// Wikipedia tool output
#[derive(Debug, Serialize)]
pub struct WikipediaOutput {
    /// The query
    pub query: String,
    /// The result list
    pub results: Vec<WikipediaResult>,
    /// Number of results
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct WikipediaResult {
    /// Title
    pub title: String,
    /// Snippet or content
    pub snippet: String,
    /// Full page URL
    pub url: String,
}

/// Wikipedia search tool
pub struct WikipediaTool {
    client: reqwest::Client,
}

impl WikipediaTool {
    /// Creates a Wikipedia search tool.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(15))
                .user_agent("LangChainRust/0.1 (Wikipedia Tool)")
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }
}

impl Default for WikipediaTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WikipediaTool {
    /// Searches Wikipedia entries
    async fn search(
        &self,
        query: &str,
        top_k: usize,
        lang: &str,
    ) -> Result<WikipediaOutput, ToolError> {
        let search_url = format!(
            "https://{}.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&format=json&srlimit={}",
            lang, urlencoding(query), top_k
        );

        let response =
            self.client.get(&search_url).send().await.map_err(|e| {
                ToolError::ExecutionFailed(format!("Wikipedia search failed: {}", e))
            })?;

        let body: serde_json::Value = response.json().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to parse search results: {}", e))
        })?;

        let search_results = body["query"]["search"]
            .as_array()
            .map(|arr| arr.to_vec())
            .unwrap_or_default();

        let mut results = Vec::new();

        for item in search_results.iter().take(top_k) {
            let title = item["title"].as_str().unwrap_or("").to_string();
            let snippet_html = item["snippet"].as_str().unwrap_or("").to_string();
            let snippet = strip_html(&snippet_html);
            let page_url = format!(
                "https://{}.wikipedia.org/wiki/{}",
                lang,
                urlencoding(&title)
            );

            results.push(WikipediaResult {
                title,
                snippet,
                url: page_url,
            });
        }

        Ok(WikipediaOutput {
            query: query.to_string(),
            total: results.len(),
            results,
        })
    }

    /// Fetches the full content of an entry
    async fn get_full_content(&self, title: &str, lang: &str) -> Result<String, ToolError> {
        let url = format!(
            "https://{}.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&titles={}&format=json",
            lang, urlencoding(title)
        );

        let response = self.client.get(&url).send().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to fetch page content: {}", e))
        })?;

        let body: serde_json::Value = response.json().await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to parse page content: {}", e))
        })?;

        let pages = body["query"]["pages"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        for (_, page) in pages {
            if let Some(extract) = page["extract"].as_str() {
                if extract.len() > 5000 {
                    return Ok(
                        extract.chars().take(5000).collect::<String>() + "\n... [内容已截断]"
                    );
                }
                return Ok(extract.to_string());
            }
        }

        Err(ToolError::ExecutionFailed(
            "page content not found".to_string(),
        ))
    }
}

/// Strips HTML tags
fn strip_html(html: &str) -> String {
    static TAG_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"<[^>]+>").expect("static regex literal must compile")
    });
    static WHITESPACE_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"\s+").expect("static regex literal must compile")
    });

    let result = TAG_RE.replace_all(html, "");
    WHITESPACE_RE.replace_all(&result, " ").trim().to_string()
}

/// URL encoding (using the urlencoding crate for complete encoding)
fn urlencoding(s: &str) -> String {
    urlencoding::encode(s).to_string()
}

/// Length-free, host-label-safe validation for a Wikipedia language subdomain.
///
/// `lang` is interpolated verbatim into the request host (`https://{lang}.wikipedia.org/…`),
/// so an unvalidated value lets a caller inject an arbitrary host/path (e.g. `evil.com/../`,
/// `zh?x=`. Anything other than a lowercase `[a-z]` + `[a-z0-9-]` label is rejected as
/// `InvalidInput` — blocking path/host/query injection while still accepting real
/// Wikipedia codes (`zh`, `en`, `simple`, `zh-min-nan`, …). The value is normalized to
/// lowercase since Wikipedia subdomains are case-insensitive but conventionally lowercase.
fn validate_lang(lang: &str) -> Result<&str, ToolError> {
    let ok = (2..=20).contains(&lang.len())
        && lang.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
        && !lang.starts_with('-')
        && !lang.ends_with('-');
    if ok {
        Ok(lang)
    } else {
        Err(ToolError::InvalidInput(format!(
            "invalid wikipedia lang: {lang:?} (expected a language code like zh/en/ja, \
             letters and hyphens only, no leading/trailing hyphen)"
        )))
    }
}

#[async_trait]
impl Tool for WikipediaTool {
    type Input = WikipediaInput;
    type Output = WikipediaOutput;

    async fn invoke(&self, input: Self::Input) -> Result<Self::Output, ToolError> {
        let top_k = input.top_k.unwrap_or(3);
        let lang = input.lang.as_deref().unwrap_or("zh");
        let lang = validate_lang(lang)?.to_string();
        let full = input.full_content.unwrap_or(false);

        if input.query.trim().is_empty() {
            return Err(ToolError::InvalidInput(
                "query must not be empty".to_string(),
            ));
        }

        let mut output = self.search(&input.query, top_k, &lang).await?;

        if full {
            for result in &mut output.results {
                if let Ok(content) = self.get_full_content(&result.title, &lang).await {
                    result.snippet = content;
                }
            }
        }

        Ok(output)
    }
}

#[async_trait]
impl BaseTool for WikipediaTool {
    fn name(&self) -> &str {
        "wikipedia"
    }

    fn description(&self) -> &str {
        "Wikipedia 百科搜索工具。搜索 Wikipedia 百科条目并返回摘要或完整内容。

参数:
- query: 搜索关键词
- top_k: 返回结果数量（默认 3）
- lang: 语言代码，如 zh/en/ja（默认 zh）
- full_content: 是否获取完整内容（默认 false）

示例:
- 搜索百科: {\"query\": \"Rust\", \"lang\": \"zh\"}
- 获取详细内容: {\"query\": \"Rust\", \"lang\": \"en\", \"full_content\": true}"
    }

    async fn run(&self, input: String) -> Result<String, ToolError> {
        let parsed: WikipediaInput = serde_json::from_str(&input)
            .map_err(|e| ToolError::InvalidInput(format!("JSON parse failed: {}", e)))?;

        let output = self.invoke(parsed).await?;

        let mut text = format!("Wikipedia 搜索结果 (查询: {})\n\n", output.query);
        for (i, result) in output.results.iter().enumerate() {
            text.push_str(&format!("{}. {}\n", i + 1, result.title));
            text.push_str(&format!("   {}\n", result.snippet));
            text.push_str(&format!("   URL: {}\n\n", result.url));
        }
        text.push_str(&format!("共 {} 条结果", output.total));

        Ok(text)
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        use schemars::schema_for;
        serde_json::to_value(schema_for!(WikipediaInput)).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wikipedia_tool_properties() {
        let tool = WikipediaTool::new();
        assert_eq!(tool.name(), "wikipedia");
        assert!(tool.description().contains("Wikipedia"));
        assert!(BaseTool::args_schema(&tool).is_some());
    }

    #[tokio::test]
    async fn test_wikipedia_empty_query() {
        let tool = WikipediaTool::new();
        let result = tool.run(r#"{"query": ""}"#.to_string()).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_strip_html() {
        let html = "<p>Hello <b>World</b></p>";
        assert_eq!(strip_html(html), "Hello World");
    }

    #[test]
    fn test_urlencoding() {
        let encoded = urlencoding("Rust programming");
        assert_eq!(encoded, "Rust%20programming");
        assert!(urlencoding("a&b=c#d?e").contains("%26"));
        assert!(urlencoding("a&b=c#d?e").contains("%3D"));
    }

    #[test]
    fn validate_lang_accepts_real_wiki_codes() {
        for lang in ["zh", "en", "ja", "simple", "zh-min-nan", "pt-br"] {
            assert_eq!(
                validate_lang(lang).unwrap(),
                lang,
                "code {lang} should be valid"
            );
        }
    }

    #[test]
    fn validate_lang_rejects_injection_candidates() {
        // Host/path/query injection and malformed labels must be rejected, not interpolated
        // into the request host.
        for lang in [
            "",
            "x",
            "a",
            "zh..en",
            "zh/en",
            "../../evil.com",
            "zh?x=1",
            "zh#frag",
            "zh:443",
            "zh%2e%2e",
            "-zh",
            "zh-",
            "z h",
        ] {
            assert!(
                validate_lang(lang).is_err(),
                "lang {lang:?} should be rejected"
            );
        }
    }

    #[tokio::test]
    async fn invoke_rejects_injected_lang() {
        let tool = WikipediaTool::new();
        let input = WikipediaInput {
            query: "Rust".to_string(),
            top_k: None,
            lang: Some("../../../evil.example.com".to_string()),
            full_content: None,
        };
        let err = tool.invoke(input).await.unwrap_err();
        assert!(
            err.to_string().contains("lang"),
            "expected an invalid-lang error, got: {err}"
        );
    }
}

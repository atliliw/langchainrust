// src/agents/deep_research/searcher.rs
//! Searcher - executes search queries in parallel across all configured
//! search tools, collects results, and deduplicates them.

use lc_core::tools::BaseTool;

use super::ResearchError;

/// A single search result from a search tool.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    /// The query that produced this result.
    pub query: String,
    /// Title of the result.
    pub title: String,
    /// Short snippet / abstract.
    pub snippet: String,
    /// URL of the result (may be empty).
    pub url: String,
}

/// Executes search queries in parallel across all configured search tools.
pub async fn search(
    searchers: &[Box<dyn BaseTool>],
    queries: &[String],
) -> Result<Vec<SearchResult>, ResearchError> {
    if searchers.is_empty() {
        return Err(ResearchError::Search(
            "no search tools available".to_string(),
        ));
    }

    let mut all_results = Vec::new();

    // Run all queries across all searchers in parallel
    let futures: Vec<_> = queries
        .iter()
        .flat_map(|query| {
            searchers.iter().map(|searcher| {
                let query = query.clone();
                let input = serde_json::json!({"query": query}).to_string();
                async move {
                    let output = searcher.run(input).await;
                    (query, output)
                }
            })
        })
        .collect();

    // Use join_all for parallel execution
    let results = futures_util::future::join_all(futures).await;

    for (query, output) in results {
        match output {
            Ok(response_text) => {
                let parsed = parse_search_response(&query, &response_text);
                all_results.extend(parsed);
            }
            Err(e) => {
                // Log but don't fail - other searchers may succeed
                log::warn!(
                    "[deep_research] search tool error for query '{}': {}",
                    query,
                    e
                );
            }
        }
    }

    if all_results.is_empty() {
        return Err(ResearchError::NoResults);
    }

    Ok(all_results)
}

/// Parses the JSON response from a search tool into `SearchResult` items.
///
/// Tolerates various response formats:
/// - `{"results": [{...}, ...]}` (DuckDuckGo-style)
/// - `[{...}, ...]` (plain array)
/// - Any other JSON object with a `results` field
fn parse_search_response(query: &str, response: &str) -> Vec<SearchResult> {
    // Try to parse as a JSON value first
    let value: serde_json::Value = match serde_json::from_str(response) {
        Ok(v) => v,
        Err(_) => {
            // If not valid JSON, treat the whole response as a single result
            return vec![SearchResult {
                query: query.to_string(),
                title: "Search Result".to_string(),
                snippet: response.to_string(),
                url: String::new(),
            }];
        }
    };

    // Try to extract results array
    let results_array = if let Some(arr) = value.get("results").and_then(|v| v.as_array()) {
        arr.clone()
    } else if let Some(arr) = value.as_array() {
        arr.clone()
    } else {
        return vec![SearchResult {
            query: query.to_string(),
            title: "Search Result".to_string(),
            snippet: response.to_string(),
            url: String::new(),
        }];
    };

    let mut results = Vec::new();
    for item in results_array {
        let title = item
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled")
            .to_string();
        let snippet = item
            .get("snippet")
            .or_else(|| item.get("abstract_text"))
            .or_else(|| item.get("description"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let url = item
            .get("url")
            .or_else(|| item.get("link"))
            .or_else(|| item.get("href"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        results.push(SearchResult {
            query: query.to_string(),
            title,
            snippet,
            url,
        });
    }

    results
}

/// Helper for collecting and deduplicating search results.
pub struct SearchCollector;

impl SearchCollector {
    /// Deduplicates search results by URL (preferring the first occurrence).
    ///
    /// Results with empty URLs are always kept.
    pub fn dedup(results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut seen_urls = std::collections::HashSet::new();
        let mut deduped = Vec::new();

        for result in results {
            if result.url.is_empty() || seen_urls.insert(result.url.clone()) {
                deduped.push(result);
            }
        }

        deduped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_search_response_results_field() {
        let response =
            r#"{"results": [{"title": "A", "snippet": "snip A", "url": "https://a.com"}]}"#;
        let results = parse_search_response("test", response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "A");
        assert_eq!(results[0].url, "https://a.com");
    }

    #[test]
    fn test_parse_search_response_plain_array() {
        let response = r#"[{"title": "B", "snippet": "snip B", "url": "https://b.com"}]"#;
        let results = parse_search_response("test", response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "B");
    }

    #[test]
    fn test_parse_search_response_non_json() {
        let response = "plain text result";
        let results = parse_search_response("test", response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "plain text result");
        assert!(results[0].url.is_empty());
    }

    #[test]
    fn test_parse_search_response_alternate_fields() {
        let response =
            r#"{"results": [{"title": "C", "description": "desc C", "link": "https://c.com"}]}"#;
        let results = parse_search_response("test", response);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].snippet, "desc C");
        assert_eq!(results[0].url, "https://c.com");
    }

    #[test]
    fn test_dedup_by_url() {
        let results = vec![
            SearchResult {
                query: "q1".to_string(),
                title: "A".to_string(),
                snippet: "first".to_string(),
                url: "https://example.com".to_string(),
            },
            SearchResult {
                query: "q2".to_string(),
                title: "A dup".to_string(),
                snippet: "duplicate".to_string(),
                url: "https://example.com".to_string(),
            },
            SearchResult {
                query: "q3".to_string(),
                title: "B".to_string(),
                snippet: "unique".to_string(),
                url: "https://other.com".to_string(),
            },
        ];

        let deduped = SearchCollector::dedup(results);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].title, "A");
        assert_eq!(deduped[1].title, "B");
    }

    #[test]
    fn test_dedup_keeps_empty_urls() {
        let results = vec![
            SearchResult {
                query: "q1".to_string(),
                title: "No URL 1".to_string(),
                snippet: "a".to_string(),
                url: String::new(),
            },
            SearchResult {
                query: "q2".to_string(),
                title: "No URL 2".to_string(),
                snippet: "b".to_string(),
                url: String::new(),
            },
        ];

        let deduped = SearchCollector::dedup(results);
        assert_eq!(deduped.len(), 2);
    }

    #[test]
    fn test_dedup_empty_input() {
        let deduped = SearchCollector::dedup(vec![]);
        assert!(deduped.is_empty());
    }
}

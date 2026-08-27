// src/retrieval/bm25/tokenizer.rs
//! Tokenizer implementation
//!
//! Provides simple tokenization for both English and Chinese

use std::collections::HashSet;

/// Tokenizer
pub struct Tokenizer {
    /// Whether to keep stopwords
    keep_stopwords: bool,

    /// English stopword list
    stopwords_en: HashSet<String>,

    /// Chinese stopword list
    stopwords_zh: HashSet<String>,
}

impl Tokenizer {
    /// Creates a new tokenizer (filtering stopwords)
    pub fn new() -> Self {
        Self {
            keep_stopwords: false,
            stopwords_en: Self::default_stopwords_en(),
            stopwords_zh: Self::default_stopwords_zh(),
        }
    }

    /// Creates a tokenizer that keeps stopwords
    pub fn with_stopwords() -> Self {
        Self {
            keep_stopwords: true,
            stopwords_en: HashSet::new(),
            stopwords_zh: HashSet::new(),
        }
    }

    /// Default English stopwords
    fn default_stopwords_en() -> HashSet<String> {
        [
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "about",
            "against",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "aren",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Default Chinese stopwords
    fn default_stopwords_zh() -> HashSet<String> {
        [
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上",
            "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这",
            "那", "里", "为", "什么", "他", "她", "它", "们", "这个", "那个", "可以", "把", "能",
            "被", "与", "及", "等", "或", "而", "但", "如", "若", "则", "因", "所以", "因为",
            "但是", "然而", "不过", "虽然", "即使", "如果", "只要",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Tokenizes text (auto-detects the language)
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();

        for word in self.tokenize_mixed(text) {
            if self.keep_stopwords || !self.is_stopword(&word) {
                terms.push(word);
            }
        }

        terms
    }

    /// Tokenizes mixed English/Chinese text
    fn tokenize_mixed(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();
        let mut current_english = String::new();
        let mut current_chinese = String::new();

        for ch in text.chars() {
            if ch.is_ascii_alphabetic() || ch.is_ascii_digit() {
                // English/digit characters
                if !current_chinese.is_empty() {
                    // First flush the accumulated Chinese
                    terms.extend(self.tokenize_chinese_segment(&current_chinese));
                    current_chinese.clear();
                }
                current_english.push(ch.to_ascii_lowercase());
            } else if Self::is_chinese_char(ch) {
                // Chinese character
                if !current_english.is_empty() {
                    // First flush the accumulated English
                    terms.push(current_english.clone());
                    current_english.clear();
                }
                current_chinese.push(ch);
            } else {
                // Other characters (punctuation, whitespace, etc.)
                if !current_english.is_empty() {
                    terms.push(current_english.clone());
                    current_english.clear();
                }
                if !current_chinese.is_empty() {
                    terms.extend(self.tokenize_chinese_segment(&current_chinese));
                    current_chinese.clear();
                }
            }
        }

        // Flush remaining characters
        if !current_english.is_empty() {
            terms.push(current_english);
        }
        if !current_chinese.is_empty() {
            terms.extend(self.tokenize_chinese_segment(&current_chinese));
        }

        terms
    }

    /// Checks whether a character is Chinese
    fn is_chinese_char(ch: char) -> bool {
        // CJK unified ideographs
        ('\u{4E00}'..='\u{9FFF}').contains(&ch) ||
        // CJK Extension A
        ('\u{3400}'..='\u{4DBF}').contains(&ch) ||
        // Chinese punctuation
        ('\u{3000}'..='\u{303F}').contains(&ch)
    }

    /// Simple Chinese tokenization (single characters + bigrams)
    fn tokenize_chinese_segment(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut terms = Vec::new();

        // Single characters
        for ch in &chars {
            terms.push(ch.to_string());
        }

        // Bigrams (n-gram)
        for i in 0..chars.len().saturating_sub(1) {
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            terms.push(bigram);
        }

        terms
    }

    /// Checks whether a term is a stopword
    fn is_stopword(&self, word: &str) -> bool {
        // Check the English stopwords first
        if self.stopwords_en.contains(word) {
            return true;
        }

        // Then check the Chinese stopwords (single characters and bigrams)
        if self.stopwords_zh.contains(word) {
            return true;
        }

        false
    }

    /// English tokenization (whitespace-split + lowercased)
    pub fn tokenize_english(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();

        for word in text.split_whitespace() {
            let word_lower = word
                .chars()
                .filter(|c| c.is_ascii_alphabetic() || c.is_ascii_digit())
                .collect::<String>()
                .to_lowercase();

            if !word_lower.is_empty()
                && (self.keep_stopwords || !self.stopwords_en.contains(&word_lower))
            {
                terms.push(word_lower);
            }
        }

        terms
    }

    /// Chinese tokenization (single characters + bigrams)
    pub fn tokenize_chinese(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();

        // Extract the Chinese characters
        let chars: Vec<char> = text
            .chars()
            .filter(|ch| Self::is_chinese_char(*ch))
            .collect();

        // Single characters
        for ch in &chars {
            let s: String = ch.to_string();
            if self.keep_stopwords || !self.stopwords_zh.contains(&s) {
                terms.push(s);
            }
        }

        // Bigrams
        for i in 0..chars.len().saturating_sub(1) {
            let bigram = format!("{}{}", chars[i], chars[i + 1]);
            if self.keep_stopwords || !self.stopwords_zh.contains(&bigram) {
                terms.push(bigram);
            }
        }

        terms
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_english() {
        let tokenizer = Tokenizer::new();

        let terms = tokenizer.tokenize_english("Hello World Rust");
        assert_eq!(terms, vec!["hello", "world", "rust"]);
    }

    #[test]
    fn test_tokenize_english_stopwords() {
        let tokenizer = Tokenizer::new();

        let terms = tokenizer.tokenize_english("The Rust is a programming language");
        // "the", "is", "a" should be filtered out
        assert!(!terms.contains(&"the".to_string()));
        assert!(!terms.contains(&"is".to_string()));
        assert!(!terms.contains(&"a".to_string()));
        assert!(terms.contains(&"rust".to_string()));
        assert!(terms.contains(&"programming".to_string()));
        assert!(terms.contains(&"language".to_string()));
    }

    #[test]
    fn test_tokenize_chinese() {
        let tokenizer = Tokenizer::new();

        let terms = tokenizer.tokenize_chinese("编程语言");
        // Single characters + bigrams
        assert!(terms.contains(&"编".to_string()));
        assert!(terms.contains(&"程".to_string()));
        assert!(terms.contains(&"语".to_string()));
        assert!(terms.contains(&"言".to_string()));
        assert!(terms.contains(&"编程".to_string()));
        assert!(terms.contains(&"程语".to_string()));
        assert!(terms.contains(&"语言".to_string()));
    }

    #[test]
    fn test_tokenize_mixed() {
        let tokenizer = Tokenizer::new();

        let terms = tokenizer.tokenize("Rust 编程语言");

        // English terms
        assert!(terms.contains(&"rust".to_string()));

        // Chinese single characters
        assert!(terms.contains(&"编".to_string()));
        assert!(terms.contains(&"程".to_string()));
        assert!(terms.contains(&"语".to_string()));
        assert!(terms.contains(&"言".to_string()));

        // Chinese bigrams
        assert!(terms.contains(&"编程".to_string()));
        assert!(terms.contains(&"语言".to_string()));
    }

    #[test]
    fn test_tokenize_with_stopwords() {
        let tokenizer = Tokenizer::with_stopwords();

        let terms = tokenizer.tokenize("The programming language");
        assert!(terms.contains(&"the".to_string()));
        assert!(terms.contains(&"programming".to_string()));
        assert!(terms.contains(&"language".to_string()));
    }

    #[test]
    fn test_chinese_stopwords() {
        let tokenizer = Tokenizer::new();

        let terms = tokenizer.tokenize_chinese("编程的语言");
        // The Chinese stopword (de) should be filtered out
        assert!(!terms.contains(&"的".to_string()));
        assert!(terms.contains(&"编".to_string()));
        assert!(terms.contains(&"程".to_string()));
    }
}

// src/retrieval/graph_rag/matcher.rs
//! Entity matching strategies for GraphRAG local queries.
//!
//! Provides the [`EntityMatcher`] trait and two implementations:
//! - [`KeywordMatcher`]: matches entities by keyword substring (default, zero-cost)
//! - [`EmbeddingMatcher`]: matches entities by embedding cosine similarity

use super::graph_store::GraphStore;
use crate::graph_rag::GraphRAGError;
use crate::hybrid::filter_by_score;
use lc_core::math::cosine_similarity;
use lc_embeddings::Embeddings;
use std::collections::{HashMap, HashSet};

/// Trait for finding relevant entities in a graph store given a query.
///
/// Implementations can use different matching strategies (keyword, embedding,
/// hybrid, etc.). The default is [`KeywordMatcher`].
pub trait EntityMatcher: Send + Sync {
    /// Find entity IDs relevant to the query, returning at most `top_k` results.
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// KeywordMatcher
// ---------------------------------------------------------------------------

/// 查询词来源类型:P2-4 用于给不同来源的命中施加不同衰减权重。
#[derive(Debug, Clone, Copy)]
enum TermKind {
    /// 查询词直接命中(权重 1.0)。
    Direct,
    /// 同义词扩展命中(衰减 `synonym_weight`)。
    Synonym,
    /// CJK 二元组命中(衰减 `cjk_bigram_weight`)。
    Bigram,
}

/// 一个待匹配查询词:文本 + 来源类型。
struct Term {
    text: String,
    kind: TermKind,
}

/// 中英混合归一化(P2-4):全角字符转半角(全角 ASCII 与半角差 0xFEE0),
/// 使 "Ｒｕｓｔ" 归一化为 "rust",与实体名的小写形式一致。
fn normalize_text(s: &str) -> String {
    s.chars()
        .map(|c| {
            let u = c as u32;
            if (0xFF01..=0xFF5E).contains(&u) {
                char::from_u32(u - 0xFEE0).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// 是否为 CJK 表意字符(中文/日文汉字等)。
fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}'
    )
}

/// 中文无空格,取 CJK 字符的相邻二元组补召回(如 "机器学习" → 机器/器学/学习),
/// 使长中文查询能命中短实体名。
fn cjk_bigrams(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().filter(|c| is_cjk(*c)).collect();
    if chars.len() < 2 {
        return Vec::new();
    }
    chars.windows(2).map(|w| w.iter().collect()).collect()
}

/// Matches entities by keyword substring search.
///
/// This is the default matcher used by GraphRAG. It splits the query into
/// keywords and scores each entity based on how many keywords match the
/// entity's name, type, and description. Name matches are weighted highest.
///
/// P2-4: 在固定 name+3/type+2/desc+1 权重之上加入三项改进,修复拍脑袋权重
/// 与子串匹配对同义词/多义词/中英混杂的漏召回:
/// - **同义词表扩展** `synonyms`:查询词命中同义词键时按等价词追加匹配
///   (命中衰减 `synonym_weight`,默认 0.7)。
/// - **中英混合归一化**:全角→半角 + 中文长查询拆 CJK 二元组,解决无空格
///   中文单 token 无法命中短实体名的问题。
/// - **TF-IDF 加权**:每个查询词按其在实体语料中的逆文档频率加权,常见词
///   (如 "Technology") 区分度小、贡献小,稀有词贡献大;`use_tfidf` 可关闭。
pub struct KeywordMatcher {
    /// Weight for name matches (default: 3).
    pub name_weight: usize,
    /// Weight for type matches (default: 2).
    pub type_weight: usize,
    /// Weight for description matches (default: 1).
    pub desc_weight: usize,
    /// 同义词表:查询词(归一化后的小写/半角形式)→ 等价词列表,等价词同样
    /// 填归一化形式。命中等价词时按等价词再匹配一次,贡献乘 `synonym_weight`。
    pub synonyms: HashMap<String, Vec<String>>,
    /// 是否启用 TF-IDF 加权(默认 true)。关闭后回落到固定权重。
    pub use_tfidf: bool,
    /// 同义词命中衰减系数(默认 0.7)。
    pub synonym_weight: f64,
    /// CJK 二元组命中衰减系数(默认 0.5)。
    pub cjk_bigram_weight: f64,
}

impl Default for KeywordMatcher {
    fn default() -> Self {
        Self {
            name_weight: 3,
            type_weight: 2,
            desc_weight: 1,
            synonyms: HashMap::new(),
            use_tfidf: true,
            synonym_weight: 0.7,
            cjk_bigram_weight: 0.5,
        }
    }
}

impl KeywordMatcher {
    /// Creates a new keyword matcher with default weights.
    pub fn new() -> Self {
        Self::default()
    }

    /// 配置同义词表(查询词 → 等价词列表)。
    pub fn with_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        self.synonyms = synonyms;
        self
    }

    /// 开关 TF-IDF 加权(默认开启)。
    pub fn with_tfidf(mut self, enabled: bool) -> Self {
        self.use_tfidf = enabled;
        self
    }

    /// 把查询拆成待匹配词序列(P2-4):直接词 + 同义词扩展 + CJK 二元组,按文本去重。
    fn build_terms(&self, query: &str) -> Vec<Term> {
        let normalized = normalize_text(query).to_lowercase();
        let tokens: Vec<String> = normalized
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|w| !w.is_empty())
            .collect();

        let mut terms: Vec<Term> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for tok in tokens {
            // 直接词优先,避免同义词与直接词文本相同时被误判成同义词。
            Self::push_term(&mut terms, &mut seen, tok.clone(), TermKind::Direct);

            // 同义词扩展:键也做归一化,容忍用户键带全角/大小写。
            let syns = self
                .synonyms
                .iter()
                .find(|(k, _)| normalize_text(k).to_lowercase() == tok)
                .map(|(_, v)| v);
            if let Some(syns) = syns {
                for syn in syns {
                    Self::push_term(&mut terms, &mut seen, syn.clone(), TermKind::Synonym);
                }
            }

            // 中文无空格,长查询拆 CJK 二元组补召回。
            if tok.chars().any(is_cjk) {
                for bg in cjk_bigrams(&tok) {
                    Self::push_term(&mut terms, &mut seen, bg, TermKind::Bigram);
                }
            }
        }
        terms
    }

    fn push_term(terms: &mut Vec<Term>, seen: &mut HashSet<String>, text: String, kind: TermKind) {
        if seen.insert(text.clone()) {
            terms.push(Term { text, kind });
        }
    }

    /// 计算每个查询词在实体语料中的平滑 IDF(P2-4)。
    ///
    /// `idf = ln((N+1)/(df+1)) + 1`,df = 命中该词的实体数。常见词 df 大、IDF 小,
    /// 稀有词 IDF 大。平滑项保证 df == N 时不归零。
    fn compute_idf(&self, terms: &[Term], store: &GraphStore) -> HashMap<String, f64> {
        let n = store.all_entities().len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for term in terms {
            df.entry(term.text.clone()).or_insert(0);
        }
        for entity in store.all_entities().values() {
            let name = normalize_text(&entity.name).to_lowercase();
            let desc = normalize_text(&entity.description).to_lowercase();
            let typ = normalize_text(&entity.entity_type).to_lowercase();
            for term in terms {
                if name.contains(&term.text)
                    || desc.contains(&term.text)
                    || typ.contains(&term.text)
                {
                    if let Some(c) = df.get_mut(&term.text) {
                        *c += 1;
                    }
                }
            }
        }
        let mut idf = HashMap::with_capacity(terms.len());
        for (text, count) in &df {
            let w = ((n + 1.0) / (*count as f64 + 1.0)).ln() + 1.0;
            idf.insert(text.clone(), w);
        }
        idf
    }

    /// 单个查询词对单个实体的得分:字段权重 × TF-IDF 权重 × 来源衰减。
    fn match_score(
        &self,
        term: &Term,
        name: &str,
        type_name: &str,
        desc: &str,
        idf_weight: f64,
    ) -> f64 {
        let mut score = 0.0f64;
        if name.contains(&term.text) {
            score += self.name_weight as f64 * idf_weight;
        }
        if type_name.contains(&term.text) {
            score += self.type_weight as f64 * idf_weight;
        }
        if desc.contains(&term.text) {
            score += self.desc_weight as f64 * idf_weight;
        }
        match term.kind {
            TermKind::Direct => score,
            TermKind::Synonym => score * self.synonym_weight,
            TermKind::Bigram => score * self.cjk_bigram_weight,
        }
    }
}

impl EntityMatcher for KeywordMatcher {
    fn find_relevant(&self, query: &str, store: &GraphStore, top_k: usize) -> Vec<String> {
        let terms = self.build_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }

        // P2-4: 预计算 TF-IDF,每个查询词按语料逆文档频率加权。
        let idf = if self.use_tfidf {
            Some(self.compute_idf(&terms, store))
        } else {
            None
        };

        let mut scored: Vec<(String, f64)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let name_lower = normalize_text(&entity.name).to_lowercase();
            let desc_lower = normalize_text(&entity.description).to_lowercase();
            let type_lower = normalize_text(&entity.entity_type).to_lowercase();

            let mut score = 0.0f64;
            for term in &terms {
                let idf_weight = idf
                    .as_ref()
                    .and_then(|m| m.get(&term.text))
                    .copied()
                    .unwrap_or(1.0);
                score += self.match_score(term, &name_lower, &type_lower, &desc_lower, idf_weight);
            }

            if score > 0.0 {
                scored.push((id.clone(), score));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(id, _)| id).collect()
    }
}

// ---------------------------------------------------------------------------
// EmbeddingMatcher
// ---------------------------------------------------------------------------

/// Matches entities by computing embedding similarity between the query and
/// entity representations (name + type + description).
///
/// Requires an [`Embeddings`] implementation to compute vectors. Embeddings
/// are cached internally to avoid recomputation across calls.
pub struct EmbeddingMatcher<E: Embeddings> {
    embeddings: E,
    /// Cached entity vectors: entity_id → embedding.
    cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
    /// 嵌入相似度最小阈值(P1-2),默认 0.0 保持旧行为。
    min_score: f64,
}

impl<E: Embeddings> EmbeddingMatcher<E> {
    /// Creates a new embedding matcher with the given embeddings backend.
    pub fn new(embeddings: E) -> Self {
        Self {
            embeddings,
            cache: std::sync::Mutex::new(HashMap::new()),
            min_score: 0.0,
        }
    }

    /// 设置嵌入相似度最小阈值(P1-2),默认 0.0 保持旧行为。
    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Returns the embedding for an entity, computing and caching it if needed.
    async fn get_entity_embedding(&self, entity_id: &str, entity_text: &str) -> Option<Vec<f32>> {
        // Check cache first
        {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(vec) = cache.get(entity_id) {
                return Some(vec.clone());
            }
        }

        // Compute and cache
        match self.embeddings.embed_query(entity_text).await {
            Ok(vec) => {
                self.cache
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(entity_id.to_string(), vec.clone());
                Some(vec)
            }
            Err(e) => {
                // 实体嵌入失败:该实体从图匹配中排除,记日志暴露降级
                log::warn!(
                    "实体 `{}` 嵌入失败,已从图匹配中排除: {}",
                    entity_id,
                    e
                );
                None
            }
        }
    }
}

impl<E: Embeddings + 'static> EntityMatcher for EmbeddingMatcher<E> {
    fn find_relevant(&self, query: &str, _store: &GraphStore, _top_k: usize) -> Vec<String> {
        // P1-4: 不再静默降级到 KeywordMatcher。
        //
        // 同步 trait 方法调不了 async `embed_query`,旧实现悄悄回落关键词匹配,
        // 用户以为在用向量匹配、实际是关键词,零提示——比报错更危险。
        // 这里拒绝静默降级:返回空结果并 `log::warn`,让失败可见。
        // 需要嵌入匹配请调用 `find_relevant_async`(GraphRAG 的 query 路径),
        // 或显式配置 `KeywordMatcher`。
        log::warn!(
            "EmbeddingMatcher::find_relevant (sync) cannot run embedding matching and no longer \
             silently falls back to keyword matching; returning empty results for query '{}'. \
             Use find_relevant_async instead, or configure KeywordMatcher for sync matching.",
            query
        );
        Vec::new()
    }
}

impl<E: Embeddings + 'static> EmbeddingMatcher<E> {
    /// Async version of entity matching using embeddings.
    ///
    /// This is the preferred method when using embedding-based matching,
    /// since embedding computation is inherently async.
    ///
    /// P0-2: 不再静默降级/静默 0 分——embedding 失败或向量维度错乱会显式报错,
    /// 让调用方知道语义匹配不可用或数据有缺陷,而不是悄悄回落 keyword
    /// 或把"维度错乱"当成"不相似"。
    pub async fn find_relevant_async(
        &self,
        query: &str,
        store: &GraphStore,
        top_k: usize,
    ) -> Result<Vec<String>, GraphRAGError> {
        let query_vec = self.embeddings.embed_query(query).await.map_err(|e| {
            GraphRAGError::QueryError(format!("EmbeddingMatcher: query embedding failed: {}", e))
        })?;

        let mut scored: Vec<(String, f64)> = Vec::new();

        for (id, entity) in store.all_entities() {
            let entity_text = format!(
                "{} {} {}",
                entity.name, entity.entity_type, entity.description
            );

            if let Some(entity_vec) = self.get_entity_embedding(id, &entity_text).await {
                match cosine_similarity(&query_vec, &entity_vec) {
                    Ok(score) => {
                        scored.push((id.clone(), score as f64));
                    }
                    // 向量维度不等是数据缺陷(如中途换 embedding 模型),
                    // 报错而非当成"不相似"静默放行。
                    Err(lc_core::math::MathError::LengthMismatch(a, b)) => {
                        return Err(GraphRAGError::QueryError(format!(
                            "EmbeddingMatcher: vector dimension mismatch {} vs {} (embedding model changed?)",
                            a, b
                        )));
                    }
                }
            }
        }

        let mut scored = filter_by_score(scored, self.min_score);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(top_k).map(|(id, _)| id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_rag::graph_store::{Entity, Relation};
    use lc_embeddings::MockEmbeddings;

    fn make_test_store() -> GraphStore {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "Python".into(),
            entity_type: "Technology".into(),
            description: "A scripting language".into(),
        });
        store.add_entity(Entity {
            id: "e3".into(),
            name: "Alice".into(),
            entity_type: "Person".into(),
            description: "A developer who uses Rust".into(),
        });
        store.add_entity(Entity {
            id: "e4".into(),
            name: "Tokio".into(),
            entity_type: "Library".into(),
            description: "An async runtime for Rust".into(),
        });
        store.add_relation(Relation {
            source: "e3".into(),
            target: "e1".into(),
            relation_type: "uses".into(),
            description: "Alice uses Rust".into(),
            doc_id: None,
        });
        store
    }

    #[test]
    fn test_keyword_matcher_basic() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("Rust programming", &store, 10);
        assert!(!results.is_empty());
        // "Rust" entity should rank first (name match + description match)
        assert_eq!(results[0], "e1");
    }

    #[test]
    fn test_keyword_matcher_top_k() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("Technology", &store, 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_keyword_matcher_no_match() {
        let store = make_test_store();
        let matcher = KeywordMatcher::new();
        let results = matcher.find_relevant("cooking recipe", &store, 10);
        assert!(results.is_empty());
    }

    /// P1-4: 同步 `find_relevant` 不再静默降级——"Rust" 在 store 中
    /// 明明命中 KeywordMatcher(e1),但 EmbeddingMatcher 同步路径必须返回空,
    /// 拒绝悄悄回落关键词匹配,让"嵌入匹配不可用"可见。
    #[test]
    fn test_embedding_matcher_sync_returns_empty_not_keyword_fallback() {
        let store = make_test_store();
        let matcher = EmbeddingMatcher::new(MockEmbeddings::new(8));
        let results = matcher.find_relevant("Rust", &store, 10);
        assert!(
            results.is_empty(),
            "sync find_relevant must NOT silently fall back to keyword matching"
        );
    }

    /// P1-4: 嵌入匹配请走 async 路径——它仍返回真正的相似实体。
    ///
    /// MockEmbeddings 对相同文本产出相同向量,因此 query 与实体文本完全一致时
    /// 余弦 = 1.0 > min_score(0.0),必被召回,断言确定。
    #[tokio::test]
    async fn test_embedding_matcher_async_still_works() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "A systems programming language".into(),
        });
        let matcher = EmbeddingMatcher::new(MockEmbeddings::new(8));
        let query = "Rust Technology A systems programming language";
        let results = matcher
            .find_relevant_async(query, &store, 10)
            .await
            .unwrap();
        assert_eq!(results, vec!["e1".to_string()]);
    }

    #[test]
    fn test_keyword_matcher_custom_weights() {
        let store = make_test_store();
        let matcher = KeywordMatcher {
            name_weight: 10,
            type_weight: 1,
            desc_weight: 0,
            ..Default::default()
        };
        let results = matcher.find_relevant("Rust", &store, 10);
        assert!(!results.is_empty());
        assert_eq!(results[0], "e1");
    }

    /// P2-4: 同义词表扩展——查询词命中同义词键时按等价词匹配,补漏召回。
    #[test]
    fn test_keyword_matcher_synonym_expansion() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "PostgreSQL".into(),
            entity_type: "Database".into(),
            description: "relational database".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "机器学习".into(),
            entity_type: "Technology".into(),
            description: "AI 领域".into(),
        });

        let synonyms: HashMap<String, Vec<String>> =
            HashMap::from([("数据库".to_string(), vec!["database".to_string()])]);
        let matcher = KeywordMatcher::new().with_synonyms(synonyms);

        // "数据库" 直接子串命中不了 PostgreSQL,靠同义词 "database" 命中 type/desc。
        let results = matcher.find_relevant("数据库", &store, 10);
        assert!(
            results.contains(&"e1".to_string()),
            "同义词 'database' 应能召回 PostgreSQL(e1)"
        );
    }

    /// P2-4: 中英混合归一化——全角字符转半角后能匹配实体名。
    #[test]
    fn test_keyword_matcher_fullwidth_normalization() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "Rust".into(),
            entity_type: "Technology".into(),
            description: "systems language".into(),
        });

        let matcher = KeywordMatcher::new();
        // "Ｒｕｓｔ" 全角 → 归一化 "rust"。
        let results = matcher.find_relevant("Ｒｕｓｔ", &store, 10);
        assert_eq!(results, vec!["e1".to_string()]);
    }

    /// P2-4: 中文无空格,长查询拆 CJK 二元组,能命中短实体名。
    #[test]
    fn test_keyword_matcher_cjk_bigram_recall() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "机器学习".into(),
            entity_type: "Technology".into(),
            description: "AI 领域".into(),
        });

        let matcher = KeywordMatcher::new();
        // 直接子串 "机器学习算法" ⊄ "机器学习",靠二元组 "机器"/"器学"/"学习" 召回。
        let results = matcher.find_relevant("机器学习算法", &store, 10);
        assert!(
            results.contains(&"e1".to_string()),
            "CJK 二元组应能召回 '机器学习' 实体"
        );
    }

    /// P2-4: TF-IDF 属性——常见词的 IDF 低于稀有词。
    #[test]
    fn test_keyword_matcher_tfidf_common_lower_than_rare() {
        let mut store = GraphStore::new();
        for name in ["Rust", "Python", "Ruby"] {
            store.add_entity(Entity {
                id: name.to_lowercase(),
                name: name.to_string(),
                entity_type: "Technology".to_string(),
                description: String::new(),
            });
        }

        let matcher = KeywordMatcher::new();
        let terms = matcher.build_terms("rust technology");
        let idf = matcher.compute_idf(&terms, &store);

        // "technology" 出现在 3 个实体的 type → IDF 低;"rust" 只在 e1 → IDF 高。
        let tech_idf = idf["technology"];
        let rust_idf = idf["rust"];
        assert!(
            rust_idf > tech_idf,
            "稀有词 'rust' 的 IDF ({:.3}) 应高于常见词 'technology' ({:.3})",
            rust_idf,
            tech_idf
        );
    }

    /// P2-4: TF-IDF 改变排序——固定权重并列时,稀有词命中方优先。
    #[test]
    fn test_keyword_matcher_tfidf_breaks_fixed_weight_tie() {
        let mut store = GraphStore::new();
        store.add_entity(Entity {
            id: "e1".into(),
            name: "data".into(),
            entity_type: "Technology".into(),
            description: "machine learning".into(),
        });
        store.add_entity(Entity {
            id: "e2".into(),
            name: "learning".into(),
            entity_type: "Technology".into(),
            description: "data".into(),
        });
        store.add_entity(Entity {
            id: "e3".into(),
            name: "extra".into(),
            entity_type: "Technology".into(),
            description: "data warehouse".into(),
        });

        let matcher = KeywordMatcher::new();
        // 固定权重下 e1/e2 并列(4 分);TF-IDF 下 "learning" 更稀有(只 2 实体),
        // e2 的 name 命中稀有词 → 应优先。
        let results = matcher.find_relevant("data learning", &store, 10);
        assert_eq!(results[0], "e2");
    }

    /// P0-2: 收敛到 lc-core 单一实现(Result<f32, MathError> 契约)。
    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_norm() {
        // 零向量不是维度错误——lc-core 返回 Ok(0.0)。
        let sim = cosine_similarity(&[], &[]).unwrap();
        assert_eq!(sim, 0.0);
    }

    /// P0-2: 维度不等必须报错,不再是静默 0.0(否则"维度错乱"被当成"不相似")。
    #[test]
    fn test_cosine_similarity_different_lengths_errors() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }
}

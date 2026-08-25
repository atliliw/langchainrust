//! BLEU 评测器:经典机器翻译/文本生成指标。
//!
//! n-gram 精率的几何平均 + 短句惩罚(brevity penalty)。
//! 完全相同为 1.0,无任何 n-gram 匹配为 0.0。

use async_trait::async_trait;
use std::collections::HashMap;

use super::{EvalError, Evaluator, Score};

/// BLEU 评测器(默认 BLEU-4)。
pub struct Bleu {
    max_n: usize,
    /// 字符级分词(中文等无空格语言用,每个字符一个 token)
    char_level: bool,
    /// 平滑:某阶 n-gram 无匹配时不直接归零,短句更友好
    smoothing: bool,
}

impl Default for Bleu {
    fn default() -> Self {
        Self::new()
    }
}

impl Bleu {
    /// 创建默认 BLEU-4 评测器。
    pub fn new() -> Self {
        Self {
            max_n: 4,
            char_level: false,
            smoothing: false,
        }
    }

    /// 使用 BLEU-n(默认 4)
    pub fn with_max_n(mut self, n: usize) -> Self {
        self.max_n = n.max(1);
        self
    }

    /// 字符级分词:中文等无空格语言按字符切(否则整句成一个 token,BLEU 失效)
    pub fn with_char_level(mut self, v: bool) -> Self {
        self.char_level = v;
        self
    }

    /// 开启平滑:某阶 n-gram 无匹配时给小值而非整体归零,短句不被一刀切
    pub fn with_smoothing(mut self, v: bool) -> Self {
        self.smoothing = v;
        self
    }

    /// corpus 级 BLEU:跨多条样例聚合 n-gram 匹配计数,再整体算几何均值 +
    /// corpus 级 brevity penalty。
    ///
    /// P2-1:句子级 BLEU 的 brevity penalty 会把短句一刀切(如 BLEU-4 下
    /// "the cat" 直接归零);corpus 聚合用总长度算惩罚、各阶 n-gram 计数合并,
    /// 短句的匹配仍能贡献低阶精度。开启 `with_smoothing` 后某阶无匹配给小值
    /// 而非整体归零。
    ///
    /// `predictions` 与 `references` 长度必须一致(逐条对应),否则 panic。
    pub fn corpus_bleu(&self, predictions: &[&str], references: &[&str]) -> f64 {
        assert_eq!(
            predictions.len(),
            references.len(),
            "predictions and references sample counts do not match"
        );
        if predictions.is_empty() {
            return 0.0;
        }
        let mut total = vec![0usize; self.max_n];
        let mut matches = vec![0usize; self.max_n];
        let mut pred_len = 0usize;
        let mut ref_len = 0usize;
        for (pred, reference) in predictions.iter().zip(references) {
            let pred_t = tokenize(pred, self.char_level);
            let ref_t = tokenize(reference, self.char_level);
            pred_len += pred_t.len();
            ref_len += ref_t.len();
            for n in 1..=self.max_n {
                let pred_grams = ngrams(&pred_t, n);
                let ref_grams = ngrams(&ref_t, n);
                for (g, &c) in &pred_grams {
                    total[n - 1] += c;
                    let r = ref_grams.get(g).copied().unwrap_or(0);
                    matches[n - 1] += c.min(r);
                }
            }
        }
        if pred_len == 0 {
            return 0.0;
        }
        let mut log_precisions: Vec<f64> = Vec::new();
        for n in 0..self.max_n {
            let t = total[n];
            let m = matches[n];
            let p = if t == 0 {
                // 该阶整条语料都没有 n-gram(所有预测都太短):平滑时跳过不惩罚,否则归零
                if self.smoothing {
                    continue;
                }
                return 0.0;
            } else if m == 0 {
                if self.smoothing {
                    // 平滑:0 匹配给小值,避免 log(0) 把整体归零
                    0.5 / t as f64
                } else {
                    return 0.0;
                }
            } else {
                m as f64 / t as f64
            };
            log_precisions.push(p.ln());
        }
        if log_precisions.is_empty() {
            return 0.0;
        }
        let geo_mean = log_precisions.iter().sum::<f64>() / log_precisions.len() as f64;
        // corpus 级 brevity penalty:总预测长度 vs 总参考长度
        let bp = if pred_len > ref_len {
            1.0
        } else {
            (1.0 - ref_len as f64 / pred_len as f64).exp()
        };
        (bp * geo_mean.exp()).clamp(0.0, 1.0)
    }
}

/// 分词:默认按空白切分并小写化;char_level 时按字符切(中文用)。
fn tokenize(s: &str, char_level: bool) -> Vec<String> {
    if char_level {
        s.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| c.to_lowercase().collect::<String>())
            .collect()
    } else {
        s.split_whitespace().map(|w| w.to_lowercase()).collect()
    }
}

fn ngrams(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
    let mut m = HashMap::new();
    if tokens.len() < n {
        return m;
    }
    for i in 0..=tokens.len() - n {
        let g: Vec<String> = tokens[i..i + n].to_vec();
        *m.entry(g).or_insert(0) += 1;
    }
    m
}

#[async_trait]
impl Evaluator for Bleu {
    async fn eval(
        &self,
        _input: &str,
        prediction: &str,
        reference: &str,
    ) -> Result<Score, EvalError> {
        let pred = tokenize(prediction, self.char_level);
        let ref_t = tokenize(reference, self.char_level);
        let plen = pred.len();
        let rlen = ref_t.len();
        if plen == 0 || rlen == 0 {
            return Ok(Score::new(0.0).with_label("empty"));
        }

        let mut log_precisions: Vec<f64> = Vec::new();
        for n in 1..=self.max_n {
            let pred_grams = ngrams(&pred, n);
            let ref_grams = ngrams(&ref_t, n);
            let mut matches = 0usize;
            let mut total = 0usize;
            for (g, &c) in &pred_grams {
                total += c;
                let r = ref_grams.get(g).copied().unwrap_or(0);
                matches += c.min(r);
            }
            if total == 0 {
                // 该阶无 n-gram(预测太短没产生):平滑时跳过不惩罚,否则归零
                if self.smoothing {
                    continue;
                }
                return Ok(Score::new(0.0).with_label("no_ngram_match"));
            }
            let p = if matches == 0 {
                if self.smoothing {
                    // 平滑:0 匹配给小值,避免 log(0) 把整体归零
                    0.5 / total as f64
                } else {
                    return Ok(Score::new(0.0).with_label("no_ngram_match"));
                }
            } else {
                matches as f64 / total as f64
            };
            log_precisions.push(p.ln());
        }

        let geo_mean = log_precisions.iter().sum::<f64>() / log_precisions.len() as f64;
        // brevity penalty:预测比参考短则惩罚
        let bp = if plen > rlen {
            1.0
        } else {
            (1.0 - rlen as f64 / plen as f64).exp()
        };
        let bleu = bp * geo_mean.exp();
        Ok(Score::new(bleu.clamp(0.0, 1.0)).with_label("bleu"))
    }

    fn name(&self) -> &str {
        "bleu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bleu_identical() {
        let ev = Bleu::new();
        let s = ev
            .eval("", "the cat sat on the mat", "the cat sat on the mat")
            .await
            .unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bleu_partial() {
        let ev = Bleu::new();
        let s = ev
            .eval("", "the cat sat on the mat", "the cat sat on a mat")
            .await
            .unwrap();
        assert!(s.value > 0.0 && s.value < 1.0);
    }

    #[tokio::test]
    async fn test_bleu_no_match() {
        let ev = Bleu::new();
        let s = ev
            .eval(
                "",
                "completely different words here",
                "the cat sat on the mat",
            )
            .await
            .unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bleu_empty() {
        let ev = Bleu::new();
        let s = ev.eval("", "", "ref").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bleu_brevity_penalty() {
        // 预测比参考短,即使词都匹配,bleu 也被惩罚 < 1
        let ev = Bleu::new().with_max_n(1);
        let s = ev
            .eval("", "the cat", "the cat sat on the mat")
            .await
            .unwrap();
        assert!(s.value < 1.0);
    }

    #[tokio::test]
    async fn test_bleu_char_level_chinese() {
        // 中文无空格,默认分词整句成一个 token;字符级才能算 n-gram
        let ev = Bleu::new().with_char_level(true).with_max_n(2);
        let s = ev.eval("", "猫坐在垫子上", "猫坐在垫子上").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bleu_smoothing_avoids_zero() {
        // 短句(词数 < 4)默认 BLEU-4 因高阶 n-gram 缺失归零
        let strict = Bleu::new();
        let s = strict.eval("", "the cat", "the cat").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
        // 开启平滑后不为零
        let smooth = Bleu::new().with_smoothing(true);
        let s2 = smooth.eval("", "the cat", "the cat").await.unwrap();
        assert!(s2.value > 0.0);
    }

    /// P2-1: corpus 级 BLEU,完全相同语料 = 1.0。
    #[test]
    fn test_corpus_bleu_identical() {
        let ev = Bleu::new();
        let v = ev.corpus_bleu(
            &["the cat", "the dog sat on the mat"],
            &["the cat", "the dog sat on the mat"],
        );
        assert!((v - 1.0).abs() < 1e-9);
    }

    /// P2-1: corpus 聚合下,短句"the cat"不再因缺 4-gram 把整体归零。
    #[tokio::test]
    async fn test_corpus_bleu_short_sentence_aggregated() {
        // 句子级 strict BLEU-4:"the cat" 无 4-gram,直接归零
        let strict = Bleu::new();
        let s = strict.eval("", "the cat", "the cat").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
        // corpus 级:短句的匹配贡献低阶精度,整体不再为零
        let v = strict.corpus_bleu(
            &["the cat", "the dog sat on the mat"],
            &["the cat", "the dog sat on the mat"],
        );
        assert!((v - 1.0).abs() < 1e-9);
    }

    /// P2-1: 某阶全语料无匹配时,strict 归零;平滑给小值而非整体归零。
    #[test]
    fn test_corpus_bleu_smoothing() {
        let preds = &["the cat", "completely different"];
        let refs = &["the cat", "the dog"];
        let strict = Bleu::new();
        let v0 = strict.corpus_bleu(preds, refs);
        assert!((v0 - 0.0).abs() < 1e-9, "strict 应为 0,实际 {v0}");
        let smooth = Bleu::new().with_smoothing(true);
        let v1 = smooth.corpus_bleu(preds, refs);
        assert!((v1 - 0.5).abs() < 1e-9, "平滑后应为 0.5,实际 {v1}");
    }

    /// P2-1: 空语料返回 0.0,不 panic。
    #[test]
    fn test_corpus_bleu_empty() {
        let v = Bleu::new().corpus_bleu(&[], &[]);
        assert!((v - 0.0).abs() < 1e-9);
    }
}

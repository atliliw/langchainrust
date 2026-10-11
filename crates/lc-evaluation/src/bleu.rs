//! BLEU evaluator: the classic machine-translation / text-generation metric.
//!
//! Geometric mean of n-gram precisions + a brevity penalty.
//! Identical text scores 1.0; no n-gram overlap scores 0.0.

use async_trait::async_trait;
use std::collections::HashMap;

use super::{EvalError, Evaluator, Score};

/// BLEU evaluator (BLEU-4 by default).
pub struct Bleu {
    max_n: usize,
    /// Character-level tokenization (for whitespace-less languages such as Chinese; one token per char)
    char_level: bool,
    /// Smoothing: an n-gram order with no match gets a small value instead of a hard zero, friendlier for short sentences
    smoothing: bool,
    /// Effective order (default `true`): an n-gram order the prediction physically cannot
    /// produce (it has fewer than `n` tokens) is *skipped* rather than zeroing the whole score.
    /// This makes BLEU-4 gracefully degrade to a smaller n for short sentences ("the cat" scores
    /// ~1.0 instead of a hard 0.0). Set to `false` for the classic BLEU, which hard-zeros.
    effective_order: bool,
}

impl Default for Bleu {
    fn default() -> Self {
        Self::new()
    }
}

impl Bleu {
    /// Creates the default BLEU-4 evaluator (with effective order, see [`with_effective_order`](Self::with_effective_order)).
    pub fn new() -> Self {
        Self {
            max_n: 4,
            char_level: false,
            smoothing: false,
            effective_order: true,
        }
    }

    /// Uses BLEU-n (default 4)
    pub fn with_max_n(mut self, n: usize) -> Self {
        self.max_n = n.max(1);
        self
    }

    /// Character-level tokenization: whitespace-less languages such as Chinese split per char (otherwise the whole sentence becomes one token and BLEU breaks)
    pub fn with_char_level(mut self, v: bool) -> Self {
        self.char_level = v;
        self
    }

    /// Enables smoothing: an order with no n-gram match gets a small value instead of a whole-zero, so short sentences are not cut off wholesale
    pub fn with_smoothing(mut self, v: bool) -> Self {
        self.smoothing = v;
        self
    }

    /// Enables/disables effective order (default enabled).
    ///
    /// When `true`, an n-gram order the prediction cannot reach (fewer than `n` tokens) is
    /// skipped rather than zeroing the whole result, so BLEU-4 degrades to a smaller n for short
    /// sentences. This is distinct from [`with_smoothing`](Self::with_smoothing): smoothing gives
    /// a small value to an order that *exists* but matched nothing; effective order ignores an
    /// order that does not exist at all. Pass `false` to recover the classic BLEU behavior.
    pub fn with_effective_order(mut self, v: bool) -> Self {
        self.effective_order = v;
        self
    }

    /// Corpus-level BLEU: aggregates n-gram match counts across examples, then computes a single
    /// geometric mean + corpus-level brevity penalty.
    ///
    /// P2-1: a sentence-level brevity penalty cuts short sentences off wholesale (e.g. "the cat"
    /// scores a hard zero under BLEU-4); corpus aggregation computes the penalty from total lengths
    /// and merges per-order n-gram counts, so short sentences still contribute low-order precision.
    /// With `with_effective_order` (default) an order the predictions cannot reach is skipped, and
    /// with `with_smoothing` an order that exists but has no match gets a small value instead of a
    /// whole zero.
    ///
    /// `predictions` and `references` must have the same length (one-to-one), otherwise
    /// [`EvalError::LengthMismatch`](crate::EvalError::LengthMismatch) is returned.
    pub fn corpus_bleu(&self, predictions: &[&str], references: &[&str]) -> Result<f64, EvalError> {
        if predictions.len() != references.len() {
            return Err(EvalError::LengthMismatch {
                predictions: predictions.len(),
                references: references.len(),
            });
        }
        if predictions.is_empty() {
            return Ok(0.0);
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
            return Ok(0.0);
        }
        let mut log_precisions: Vec<f64> = Vec::new();
        for n in 0..self.max_n {
            let t = total[n];
            let m = matches[n];
            let p = if t == 0 {
                // no n-gram at this order across the whole corpus (all predictions too short):
                // effective order skips it (no penalty), else smoothing skips it; otherwise a hard zero
                if self.effective_order || self.smoothing {
                    continue;
                }
                return Ok(0.0);
            } else if m == 0 {
                if self.smoothing {
                    // smoothing: 0 matches get a small value, avoiding log(0) zeroing the whole result
                    0.5 / t as f64
                } else {
                    return Ok(0.0);
                }
            } else {
                m as f64 / t as f64
            };
            log_precisions.push(p.ln());
        }
        if log_precisions.is_empty() {
            return Ok(0.0);
        }
        let geo_mean = log_precisions.iter().sum::<f64>() / log_precisions.len() as f64;
        // corpus-level brevity penalty: total prediction length vs total reference length
        let bp = if pred_len > ref_len {
            1.0
        } else {
            (1.0 - ref_len as f64 / pred_len as f64).exp()
        };
        Ok((bp * geo_mean.exp()).clamp(0.0, 1.0))
    }
}

/// Tokenizes: splits on whitespace and lowercases by default; with char_level splits per char (for Chinese).
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
            if pred_grams.is_empty() {
                // prediction has fewer than `n` tokens, so this n-gram order does not exist at all:
                // effective order skips it (no penalty), else smoothing skips it; otherwise a hard zero
                if self.effective_order || self.smoothing {
                    continue;
                }
                return Ok(Score::new(0.0).with_label("no_ngram_match"));
            }
            let mut matches = 0usize;
            let mut total = 0usize;
            for (g, &c) in &pred_grams {
                total += c;
                let r = ref_grams.get(g).copied().unwrap_or(0);
                matches += c.min(r);
            }
            let p = if matches == 0 {
                if self.smoothing {
                    // smoothing: 0 matches get a small value, avoiding log(0) zeroing the whole result
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
        // brevity penalty: penalize when the prediction is shorter than the reference
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
        // prediction shorter than reference: even with all words matching, bleu is penalized below 1
        let ev = Bleu::new().with_max_n(1);
        let s = ev
            .eval("", "the cat", "the cat sat on the mat")
            .await
            .unwrap();
        assert!(s.value < 1.0);
    }

    #[tokio::test]
    async fn test_bleu_char_level_chinese() {
        // Chinese has no spaces; default tokenization makes the whole sentence one token; char-level is required for n-grams
        let ev = Bleu::new().with_char_level(true).with_max_n(2);
        let s = ev.eval("", "猫坐在垫子上", "猫坐在垫子上").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn test_bleu_smoothing_avoids_zero() {
        // a 2-token sentence has no 3-/4-grams; hard classic BLEU-4 zeroes it out
        let classic = Bleu::new().with_effective_order(false);
        let s = classic.eval("", "the cat", "the cat").await.unwrap();
        assert!(
            (s.value - 0.0).abs() < 1e-9,
            "classic 4-gram 应为 0,实际 {}",
            s.value
        );
        // default effective order degrades to the reachable n, so short sentences are no longer cut off
        let effective = Bleu::new();
        let s2 = effective.eval("", "the cat", "the cat").await.unwrap();
        assert!(
            (s2.value - 1.0).abs() < 1e-9,
            "effective order 应退化为小 n 满分,实际 {}",
            s2.value
        );
        // with smoothing enabled it is also no longer zero
        let smooth = Bleu::new().with_effective_order(false).with_smoothing(true);
        let s3 = smooth.eval("", "the cat", "the cat").await.unwrap();
        assert!(s3.value > 0.0);
    }

    /// Effective order is distinct from smoothing: an order the prediction *can produce* but which
    /// matches nothing is a genuine zero-precision order (smoothed small, not skipped), whereas an
    /// order the prediction *cannot produce* at all is skipped.
    #[tokio::test]
    async fn effective_order_skips_unreachable_orders_but_not_zero_match_orders() {
        // "red blue" (2 tokens) vs a long reference: only n=1,2 exist on the prediction side.
        let eff = Bleu::new();
        // n=1,2 both match "the cat" exactly; n=3,4 are unreachable and skipped -> perfect
        let s = eff.eval("", "the cat", "the cat").await.unwrap();
        assert!((s.value - 1.0).abs() < 1e-9, "got {}", s.value);

        // Effective order does NOT excuse a genuinely zero-match order: a 2-token prediction that
        // shares no n-gram with the reference still scores a (smoothed) low value via smoothing,
        // not a free pass.
        let smooth = Bleu::new().with_effective_order(true).with_smoothing(true);
        let s2 = smooth.eval("", "red blue", "the cat ate").await.unwrap();
        assert!(
            s2.value > 0.0 && s2.value <= 1.0,
            "zero-match order must not be skipped, got {}",
            s2.value
        );
    }

    /// P2-1: corpus-level BLEU, identical corpora = 1.0.
    #[test]
    fn test_corpus_bleu_identical() {
        let ev = Bleu::new();
        let v = ev
            .corpus_bleu(
                &["the cat", "the dog sat on the mat"],
                &["the cat", "the dog sat on the mat"],
            )
            .unwrap();
        assert!((v - 1.0).abs() < 1e-9);
    }

    /// P2-1 + effective order: the short sentence "the cat" no longer zeroes the whole result.
    #[tokio::test]
    async fn test_corpus_bleu_short_sentence_aggregated() {
        // sentence-level classic BLEU-4 (effective order off): "the cat" has no 4-gram, hard zero
        let classic = Bleu::new().with_effective_order(false);
        let s = classic.eval("", "the cat", "the cat").await.unwrap();
        assert!((s.value - 0.0).abs() < 1e-9);
        // corpus-level with the default effective order: unreachable orders are skipped and the
        // short sentence's matches contribute low-order precision, so the whole is 1.0
        let v = Bleu::new()
            .corpus_bleu(
                &["the cat", "the dog sat on the mat"],
                &["the cat", "the dog sat on the mat"],
            )
            .unwrap();
        assert!((v - 1.0).abs() < 1e-9);
    }

    /// P2-1: when an order has no match across the whole corpus, classic strict zeroes out; smoothing
    /// gives a small value instead of a whole zero. Under the default effective order the unreachable
    /// high-order n-grams are skipped, so the short corpus scores without a hard zero.
    #[test]
    fn test_corpus_bleu_smoothing() {
        let preds = &["the cat", "completely different"];
        let refs = &["the cat", "the dog"];
        // classic BLEU-4 (no effective order): missing 3-/4-grams zero the whole result
        let classic = Bleu::new().with_effective_order(false);
        let v0 = classic.corpus_bleu(preds, refs).unwrap();
        assert!((v0 - 0.0).abs() < 1e-9, "classic 应为 0,实际 {v0}");
        // default effective order: unreachable 3-/4-grams skipped, p1=p2=0.5 -> 0.5
        let effective = Bleu::new();
        let v1 = effective.corpus_bleu(preds, refs).unwrap();
        assert!((v1 - 0.5).abs() < 1e-9, "effective 应为 0.5,实际 {v1}");
        // smoothing lifts the classic zero via a small value
        let smooth = Bleu::new().with_effective_order(false).with_smoothing(true);
        let v2 = smooth.corpus_bleu(preds, refs).unwrap();
        assert!((v2 - 0.5).abs() < 1e-9, "平滑后应为 0.5,实际 {v2}");
    }

    /// P2-1: an empty corpus returns 0.0, no panic.
    #[test]
    fn test_corpus_bleu_empty() {
        let v = Bleu::new().corpus_bleu(&[], &[]).unwrap();
        assert!((v - 0.0).abs() < 1e-9);
    }

    /// S6: mismatched prediction/reference counts return LengthMismatch instead of panicking.
    #[test]
    fn test_corpus_bleu_length_mismatch_returns_err() {
        let ev = Bleu::new();
        let err = ev.corpus_bleu(&["a", "b"], &["a"]).unwrap_err();
        assert!(matches!(
            err,
            EvalError::LengthMismatch {
                predictions: 2,
                references: 1
            }
        ));
    }
}

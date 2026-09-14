// lc-rag/src/mmr.rs
//! Maximum Marginal Relevance (MMR) 重排。
//!
//! RRF 融合只按相关性排序,同主题的 top-k 可能挤满一个话题。MMR 在相关性里
//! 注入多样性:逐位置贪心选「既相关、又和已选结果最不相似」的一项 ——
//! 用一个 λ 在「纯相关」与「纯去重」之间滑动:
//!
//! ```text
//! score(item) = λ·relevance(item) − (1−λ)·max{ sim(item, s) : s ∈ selected }
//! ```
//!
//! λ=1 → 纯相关排序(等价不重排);λ=0 → 强制多样性(几乎只看互不相似);
//! 常规取值 0.5~0.7。纯算法,无新依赖,输入给「(id, 相关分, 内容向量)」三元组,
//! 便于用余弦相似矩阵直接构造单测。

/// 两向量余弦相似度,a、b 等长(调用方保证);任一侧全零返回 0.0(零向量无方向)。
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    if na < f64::EPSILON || nb < f64::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// 第 `idx` 个候选相对当前已选集 `selected` 的 MMR 增益分。
fn mmr_gain<V: AsRef<[f32]>>(
    candidates: &[(String, f64, V)],
    selected: &[usize],
    idx: usize,
    lambda: f32,
) -> f64 {
    let relevance = candidates[idx].1;
    let max_sim = selected
        .iter()
        .map(|&s| cosine_similarity(candidates[idx].2.as_ref(), candidates[s].2.as_ref()))
        .fold(0.0f64, f64::max);
    lambda as f64 * relevance - (1.0 - lambda) as f64 * max_sim
}

/// 对所有 `candidates` 做 MMR 贪心选择,返回重排后的 id 列表(最多 `k` 项)。
///
/// - `lambda ∈ \[0,1\]`:越高越看重相关、越低越看重多样(越界自动 clamp)。
/// - `k == 0` 或空输入返回空;`k` 超出候选数时取全部。
pub fn mmr<V: AsRef<[f32]>>(candidates: &[(String, f64, V)], lambda: f32, k: usize) -> Vec<String> {
    if candidates.is_empty() || k == 0 {
        return Vec::new();
    }
    let lambda = lambda.clamp(0.0, 1.0);
    let k = k.min(candidates.len());

    let mut unselected: Vec<usize> = (0..candidates.len()).collect();
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    while selected.len() < k {
        // 每轮贪心选 MMR 增益最大的未选项。
        let next = unselected
            .iter()
            .copied()
            .max_by(|&a, &b| {
                mmr_gain(candidates, &selected, a, lambda)
                    .partial_cmp(&mmr_gain(candidates, &selected, b, lambda))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_default();
        unselected.retain(|&i| i != next);
        selected.push(next);
    }

    selected
        .into_iter()
        .map(|i| candidates[i].0.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造一个手写余弦相似矩阵可推演的局面,验证 MMR 的「多样优先」与
    /// 全局正确性,而非只测同一条贪心路径。
    ///
    /// 候选向量(单位方向,非零):
    /// - A = [1.0](一维正)
    /// - B = [-1.0](一维负,与 A 完全反相位 → 余弦 -1)
    /// - C = A 全张量复制(A 自身 → 余弦 +1)
    ///
    /// 给定相关分:rel(A)=0.9, rel(B)=0.5, rel(C)=0.8,k=3。
    /// | 对 | 余弦 |
    /// |----|------|
    /// | A·C | 1.0 |
    /// | A·B | -1.0 |
    /// | B·C | -1.0 |
    ///
    /// 贪心(λ=0.5):
    ///  1. 首项:selected 空 → 纯相关 ⇒ A(0.9)。
    ///  2. 第二项:B 增益 = 0.5·0.5 − 0.5·max(cos(B,A)) = 0.25 − 0.5·(−1) = 0.75;
    ///     C 增益 = 0.5·0.8 − 0.5·cos(C,A) = 0.4 − 0.5·1 = −0.1 ⇒ 选 B。
    ///  3. 第三项:C 增益 = 0.5·0.8 − 0.5·max(cos(C,A),cos(C,B)) = 0.4 − 0.5·1 = −0.1
    ///     (仅剩 C)⇒ 选 C。顺序:[A, B, C]。
    ///
    /// 对照 λ=1(纯相关):[A, C, B]。两组都要求与手算一致。
    #[test]
    fn mmr_trades_diversity_against_relevance_exactly() {
        let v_a = vec![1.0f32];
        let v_b = vec![-1.0f32];
        let v_c = v_a.clone();
        let candidates: Vec<(String, f64, Vec<f32>)> = vec![
            ("A".into(), 0.9, v_a),
            ("B".into(), 0.5, v_b),
            ("C".into(), 0.8, v_c),
        ];

        // λ=0.5:第一选 A(纯相关),第二被迫转向与 A 反相位的 B(多样性胜出)。
        assert_eq!(mmr(&candidates, 0.5, 3), vec!["A", "B", "C"]);

        // λ=1:纯相关排序,多样性完全不介入。
        assert_eq!(mmr(&candidates, 1.0, 3), vec!["A", "C", "B"]);
    }

    /// k 截断:λ=0.5 下取 k=1 只留首项(纯相关 A)。
    #[test]
    fn mmr_truncates_to_k() {
        let candidates = vec![
            ("A".to_string(), 0.9, vec![1.0f32]),
            ("B".to_string(), 0.5, vec![-1.0f32]),
        ];
        assert_eq!(mmr(&candidates, 0.5, 1), vec!["A"]);
    }

    /// 空输入与 k=0 都是空结果,不 panic。
    #[test]
    fn mmr_handles_empty_and_zero() {
        assert!(mmr::<Vec<f32>>(&[], 0.5, 3).is_empty());
        let candidates = vec![("A".to_string(), 0.9, vec![1.0f32])];
        assert!(mmr(&candidates, 0.5, 0).is_empty());
    }

    /// 首项选择与空已选集分支:即便有与 A 完全相似的 C,首项也按纯相关抓
    /// 全局最高分(与手算一致)。
    #[test]
    fn mmr_first_pick_is_pure_relevance() {
        let candidates = vec![
            ("A".to_string(), 0.9, vec![1.0f32]),
            ("C".to_string(), 0.8, vec![1.0f32]), // 与 A 余弦=1
        ];
        assert_eq!(mmr(&candidates, 0.5, 2), vec!["A", "C"]);
    }

    /// 余弦相似度单元:全零向量返回 0(不 panic),单位向量对返回 ±1。
    #[test]
    fn cosine_zero_and_unit_vectors() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[0.5, 0.5]), 0.0);
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-9);
    }
}

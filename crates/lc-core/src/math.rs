// src/core/math.rs
//! Shared math utilities.

/// Error type for math operations.
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum MathError {
    /// The input vectors have different lengths.
    #[error("vector length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1], where 1 means identical direction.
/// Returns an error if vectors have different lengths.
/// Returns 0.0 if either vector has zero norm (below epsilon threshold).
///
/// Internally computes in f64 for precision, but accepts and returns f32
/// to match the project's embedding type.
///
/// # Examples
/// ```no_run
/// use lc_core::math::cosine_similarity;
///
/// let a = vec![1.0_f32, 0.0, 0.0];
/// let b = vec![0.0_f32, 1.0, 0.0];
/// let sim = cosine_similarity(&a, &b).unwrap();
/// assert!((sim - 0.0).abs() < 1e-6);
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, MathError> {
    if a.len() != b.len() {
        return Err(MathError::LengthMismatch(a.len(), b.len()));
    }

    // Compute in f64 for precision (M30), then cast back to f32.
    let dot_product: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();

    // Use a real near-zero threshold instead of f64::EPSILON: EPSILON (~2.2e-16)
    // only ever caught an *exact* zero norm, so near-degenerate vectors slipped
    // through and produced a blown-up cosine. Embeddings are L2-normalized to ~1.0,
    // so anything below MIN_NORM is genuinely degenerate — return 0.0 (no direction).
    const MIN_NORM: f64 = 1e-8;
    if norm_a < MIN_NORM || norm_b < MIN_NORM {
        return Ok(0.0);
    }

    Ok((dot_product / (norm_a * norm_b)) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_different_lengths_returns_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_near_zero_vector_is_degenerate() {
        // R2: a norm that is tiny but non-zero (below MIN_NORM) must not divide into a
        // wild cosine. Under the old f64::EPSILON gate this returned ~1.0; now it's 0.0.
        let tiny = vec![1e-10_f32, 0.0, 0.0];
        let unit = vec![1.0_f32, 0.0, 0.0];
        assert_eq!(cosine_similarity(&tiny, &unit).unwrap(), 0.0);
    }
}

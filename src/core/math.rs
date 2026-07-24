// src/core/math.rs
//! Shared math utilities.

/// Error type for math operations.
#[derive(Debug, Clone, thiserror::Error)]
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
/// ```
/// use langchainrust::core::math::cosine_similarity;
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

    // Use epsilon comparison instead of exact == 0.0 (C22)
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
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
}

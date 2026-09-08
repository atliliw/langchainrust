use super::*;

/// Builds a minimal WordPiece tokenizer (JSON via `Tokenizer::from_bytes`,
/// matching the real tokenizer.json load path). Vocab: `[UNK]=0, hello=1, world=2`,
/// with `[PAD]=3` appended when `with_pad_in_vocab=true`.
fn tiny_tokenizer(with_pad_in_vocab: bool) -> tokenizers::Tokenizer {
    let mut vocab = serde_json::json!({
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
    });
    if with_pad_in_vocab {
        vocab["[PAD]"] = serde_json::json!(3);
    }
    let json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordPiece",
            "vocab": vocab,
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100
        }
    });
    tokenizers::Tokenizer::from_bytes(json.to_string().as_bytes())
        .expect("tiny tokenizer should deserialize")
}

#[test]
fn test_l2_normalize() {
    let mut v = vec![3.0, 4.0];
    crate::l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5);
    assert!((v[0] - 0.6).abs() < 1e-5);
    assert!((v[1] - 0.8).abs() < 1e-5);
}

#[test]
fn test_l2_normalize_zero() {
    let mut v = vec![0.0, 0.0, 0.0];
    crate::l2_normalize(&mut v);
    assert!(v.iter().all(|x| *x == 0.0));
}

/// P2-2: a real WordPiece tokenizer emits deterministic token IDs (with no post_processor,
/// `add_special_tokens=true` appends no [CLS]/[SEP]).
#[test]
fn test_tokenize_real_wordpiece() {
    let tok = tiny_tokenizer(false);
    let enc = tok.encode("hello world", true).unwrap();
    assert_eq!(enc.get_ids(), &[1u32, 2u32]);
    assert_eq!(enc.get_attention_mask(), &[1u32, 1u32]);
}

/// P2-2: out-of-vocab words fall back to [UNK]=0.
#[test]
fn test_tokenize_unknown_word_uses_unk() {
    let tok = tiny_tokenizer(false);
    let enc = tok.encode("zzzznotinvocab", true).unwrap();
    assert_eq!(enc.get_ids(), &[0u32]);
}

/// P2-4: batch pad alignment — short rows get pad_id, mask gets 0, long rows unchanged.
#[test]
fn test_build_batch_tensors_pads_to_longest() {
    let tok = tiny_tokenizer(false);
    let encodings = tok
        .encode_batch(vec!["hello".to_string(), "hello world".to_string()], true)
        .unwrap();
    let (input_ids, attention_mask, token_type_ids, max_len) =
        LocalInner::build_batch_tensors(&encodings, 8, 0);
    assert_eq!(max_len, 2);
    // "hello" → [1, PAD(0)], "hello world" → [1, 2]
    assert_eq!(input_ids, vec![1, 0, 1, 2]);
    assert_eq!(attention_mask, vec![1, 0, 1, 1]);
    assert_eq!(token_type_ids, vec![0, 0, 0, 0]);
}

#[test]
fn test_resolve_pad_id_with_pad_token() {
    let tok = tiny_tokenizer(true);
    assert_eq!(LocalInner::resolve_pad_id(&tok), 3);
}

#[test]
fn test_resolve_pad_id_defaults_zero() {
    let tok = tiny_tokenizer(false);
    assert_eq!(LocalInner::resolve_pad_id(&tok), 0);
}

/// P2-4: 3D masked mean pooling — pad positions with mask=0 do not participate in the mean.
#[test]
fn test_pool_rows_3d_masked() {
    // shape [2, 3, 2]: two rows of 3 positions each, dim=2.
    let shape = vec![2usize, 3, 2];
    let data = vec![
        // row0: tokens [1,2,3]
        1.0, 10.0, 2.0, 20.0, 3.0, 30.0, // row1
        4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
    ];
    let masks = vec![vec![1, 1, 1], vec![1, 0, 0]];
    let rows = LocalInner::pool_rows(&shape, &data, &masks, 2, 3).unwrap();
    assert_eq!(rows.len(), 2);
    // row0 mean = ((1+2+3)/3, (10+20+30)/3) = (2, 20)
    assert!((rows[0][0] - 2.0).abs() < 1e-5);
    assert!((rows[0][1] - 20.0).abs() < 1e-5);
    // row1 takes only the first position = (4, 40)
    assert!((rows[1][0] - 4.0).abs() < 1e-5);
    assert!((rows[1][1] - 40.0).abs() < 1e-5);
}

/// P2-4: 2D `[batch, dim]` output is split by row directly.
#[test]
fn test_pool_rows_2d() {
    let shape = vec![2usize, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let masks = vec![vec![1], vec![1]];
    let rows = LocalInner::pool_rows(&shape, &data, &masks, 2, 1).unwrap();
    assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

/// P0-1 alignment contract: model output rows != input batch → explicit BatchMismatch.
#[test]
fn test_pool_rows_batch_mismatch() {
    let shape = vec![3usize, 2, 2];
    let data = vec![0.0; 12];
    let masks = vec![vec![1], vec![1]];
    let err = LocalInner::pool_rows(&shape, &data, &masks, 2, 1).unwrap_err();
    assert!(matches!(
        err,
        EmbeddingError::BatchMismatch {
            expected: 2,
            actual: 3
        }
    ));
}

/// 0.21.0 S5.2: 	oken_rows_from_3d keeps only kept positions, slices rows by dim.
#[test]
fn test_token_rows_from_3d_filters_and_slices() {
    // shape [1, 3, 2]: 3 positions, dim 2. keep = [true, false, true].
    let shape = vec![1usize, 3, 2];
    let data = vec![1.0, 2.0, 9.0, 9.0, 3.0, 4.0];
    let keep = vec![true, false, true];
    let rows = LocalInner::token_rows_from_3d(&shape, &data, &keep, 3).unwrap();
    assert_eq!(rows, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
}

/// 0.21.0 S5.2: non-3D output (already-pooled 2D model) errors explicitly —
/// no token-level information to expose.
#[test]
fn test_token_rows_from_3d_rejects_2d() {
    let shape = vec![1usize, 2];
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let keep = vec![true, true];
    let err = LocalInner::token_rows_from_3d(&shape, &data, &keep, 2).unwrap_err();
    assert!(matches!(err, EmbeddingError::ParseError(_)));
}

/// 0.21.0 S5.2: batch != 1 is a BatchMismatch (the path encodes a single text).
#[test]
fn test_token_rows_from_3d_rejects_batch() {
    let shape = vec![2usize, 2, 2];
    let data = vec![0.0; 8];
    let keep = vec![true, true];
    let err = LocalInner::token_rows_from_3d(&shape, &data, &keep, 2).unwrap_err();
    assert!(matches!(
        err,
        EmbeddingError::BatchMismatch {
            expected: 1,
            actual: 2
        }
    ));
}

/// 0.21.0 S5.2: all positions excluded -> EmptyInput (never an empty vector list).
#[test]
fn test_token_rows_from_3d_empty_input() {
    let shape = vec![1usize, 2, 2];
    let data = vec![0.0; 4];
    let keep = vec![false, false];
    let err = LocalInner::token_rows_from_3d(&shape, &data, &keep, 2).unwrap_err();
    assert!(matches!(err, EmbeddingError::EmptyInput));
}

/// 0.21.0 S5.2: short tensor data errors instead of panicking on the slice.
#[test]
fn test_token_rows_from_3d_short_data() {
    let shape = vec![1usize, 2, 2];
    let data = vec![1.0]; // need 4
    let keep = vec![true, true];
    let err = LocalInner::token_rows_from_3d(&shape, &data, &keep, 2).unwrap_err();
    assert!(matches!(err, EmbeddingError::ParseError(_)));
}

/// 0.21.0 S5.2: char offsets -> byte spans through the real WordPiece tokenizer
/// (this fixture tokenizer has no special tokens, so offsets cover real words).
#[test]
fn test_token_level_offsets_to_byte_spans() {
    let tok = tiny_tokenizer(false);
    let enc = tok.encode("hello world", true).unwrap();
    let offsets = enc.get_offsets().to_vec();
    let spans = crate::char_spans_to_byte_spans("hello world", &offsets);
    assert_eq!(spans.len(), 2);
    assert_eq!(&"hello world"[spans[0].start..spans[0].end], "hello");
    assert_eq!(&"hello world"[spans[1].start..spans[1].end], "world");
}

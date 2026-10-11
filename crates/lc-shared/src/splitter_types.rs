// lc-shared/src/splitter_types.rs
//! Text splitter types shared across crates.
//!
//! `TextSplitter` trait and `RecursiveCharacterSplitter` are needed by
//! both `lc-vector-stores` and `lc-rag`, so they live here to break
//! the circular dependency.

use crate::document_types::Document;
use serde_json::Value;

/// Text splitter trait
pub trait TextSplitter: Send + Sync {
    /// Split text into chunks
    fn split_text(&self, text: &str) -> Vec<String>;

    /// Split a document into smaller documents
    fn split_document(&self, document: &Document) -> Vec<Document> {
        let chunks = self.split_text(&document.content);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut metadata = document.metadata.clone();
                // Only insert the chunk index when the user hasn't already
                // provided a "chunk" key — never silently overwrite it.
                metadata
                    .entry("chunk".to_string())
                    .or_insert(Value::String(i.to_string()));

                Document {
                    content: chunk,
                    metadata,
                    id: None,
                }
            })
            .collect()
    }
}

/// A source-span wrapper used internally by the recursive splitter before
/// overlap is applied: the raw chunk text plus its contiguous `[start, end)`
/// char range within the original input. Char (not byte) offsets.
#[derive(Clone, Debug)]
struct SrcSpan {
    text: String,
    start: usize,
    end: usize,
}

/// A split chunk plus its provenance in the original input
/// (produced by [`RecursiveCharacterSplitter::split_text_with_indices`]).
///
/// `start_char`/`start_line`/`end_line` describe the contiguous source region
/// the chunk's **leading-content-then-trailing-content** maps back to.
/// Prepended overlap is attributed to the preceding chunk, so spans are disjoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkSpan {
    /// The emitted chunk text (identical to what [`TextSplitter::split_text`] returns).
    pub text: String,
    /// 0-based character offset of the chunk's leading content in the original text.
    pub start_char: usize,
    /// 1-based line where the chunk's leading content begins.
    pub start_line: usize,
    /// 1-based line where the chunk's trailing content ends.
    pub end_line: usize,
}

/// Maps a char offset to its 1-based line number by binary-searching precomputed
/// line-start char offsets (no per-chunk O(n) rescan).
struct LineIndex {
    /// Char offset of each line's first character; the first entry is always 0.
    line_starts: Vec<usize>,
}

impl LineIndex {
    fn from(text: &str) -> Self {
        let mut line_starts = vec![0usize];
        for (i, ch) in text.chars().enumerate() {
            if ch == '\n' {
                line_starts.push(i + 1);
            }
        }
        Self { line_starts }
    }

    /// 1-based line number containing `char_offset`.
    fn line_of(&self, char_offset: usize) -> usize {
        // `partition_point` returns the count of line-starts <= char_offset,
        // which equals the 1-based line index (line_starts[0] = 0 always <= offset).
        self.line_starts
            .partition_point(|&s| s <= char_offset)
            .max(1)
    }
}

/// Recursive character splitter
///
/// Splits text by separator priority, recursively trying smaller separators.
pub struct RecursiveCharacterSplitter {
    /// Chunk size (character count)
    chunk_size: usize,

    /// Chunk overlap (character count)
    chunk_overlap: usize,

    /// Separator list (by priority)
    separators: Vec<String>,
}

impl RecursiveCharacterSplitter {
    /// Create a new recursive character splitter
    ///
    /// `chunk_size` is clamped to a minimum of 1: a value of `0` — whether
    /// passed directly or via a user-supplied separator list that shrank the
    /// effective window to nothing — used to make a single character always
    /// exceed the quota and recurse forever, blowing the stack.
    ///
    /// `chunk_overlap` is clamped to `[0, chunk_size-1]`: an overlap equal to
    /// or larger than the chunk would re-emit the entire previous chunk (and
    /// with no room left for new content), so it is capped just under the
    /// chunk size to keep the overlap meaningful and `chunk_size` a true upper
    /// bound on each emitted chunk.
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        let chunk_size = chunk_size.max(1);
        let chunk_overlap = chunk_overlap.min(chunk_size.saturating_sub(1));
        Self {
            chunk_size,
            chunk_overlap,
            separators: Self::default_separators(),
        }
    }

    /// The default separator set by priority.
    fn default_separators() -> Vec<String> {
        vec![
            "\n\n".to_string(), // paragraph
            "\n".to_string(),   // line
            "。".to_string(),   // Chinese period
            ".".to_string(),    // English period
            " ".to_string(),    // space
            "".to_string(),     // character
        ]
    }

    /// Create with default parameters (chunk_size=1000, chunk_overlap=200)
    pub fn with_defaults() -> Self {
        Self::new(1000, 200)
    }

    /// Set custom separators
    ///
    /// An empty separator list would degrade the recursive splitter to a
    /// character-level split with no way to recurse further — which, combined
    /// with a too-large chunk, could recurse without making progress. Guard by
    /// falling back to the default separator set on an empty list.
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        if separators.is_empty() {
            self.separators = Self::default_separators();
        } else {
            self.separators = separators;
        }
        self
    }

    /// Split text (internal recursive method)
    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        self.split_text_recursive_spanned(text, separators, 0)
            .into_iter()
            .map(|s| s.text)
            .collect()
    }

    /// Split `text` by `separator`, yielding each part and its char range
    /// within `text`. An empty separator splits every character into its own
    /// single-char part. (Char offsets, not byte offsets, so slicing stays
    /// on UTF-8 scalar boundaries for multi-byte input.)
    fn split_segments(text: &str, separator: &str) -> Vec<SrcSpan> {
        if separator.is_empty() {
            text.chars()
                .enumerate()
                .map(|(i, c)| SrcSpan {
                    text: c.to_string(),
                    start: i,
                    end: i + 1,
                })
                .collect()
        } else {
            let mut segments = Vec::new();
            let mut pos = 0usize;
            for part in text.split(separator) {
                let n = part.chars().count();
                segments.push(SrcSpan {
                    text: part.to_string(),
                    start: pos,
                    end: pos + n,
                });
                pos += n + separator.chars().count();
            }
            segments
        }
    }

    /// Same recursive split as [`Self::split_text_recursive`], but each
    /// returned chunk also carries the `[start, end)` char range it occupies
    /// in the original `text`. `base_char` is the char offset of `text` within
    /// the original document (used when recursing into a sub-slice).
    fn split_text_recursive_spanned(
        &self,
        text: &str,
        separators: &[String],
        base_char: usize,
    ) -> Vec<SrcSpan> {
        let mut chunks: Vec<SrcSpan> = Vec::new();

        if text.is_empty() {
            return chunks;
        }

        // If text is already small enough, return as-is
        if text.chars().count() <= self.chunk_size {
            chunks.push(SrcSpan {
                text: text.to_string(),
                start: base_char,
                end: base_char + text.chars().count(),
            });
            return chunks;
        }

        // Find a suitable separator
        let separator = separators
            .iter()
            .find(|s| text.contains(s.as_str()))
            .cloned()
            .unwrap_or_default();

        let segments = Self::split_segments(text, &separator);

        // Merge segments into chunks, tracking source ranges. When a separator
        // is re-inserted (`sep + split`), the merged span starts at the
        // separator's position, i.e. `\n\n` chars before the segment itself.
        let mut current: Option<SrcSpan> = None;

        for seg in segments {
            let (split_with_sep, sp_start) = if separator.is_empty() || current.is_none() {
                (seg.text.clone(), seg.start)
            } else {
                (
                    format!("{}{}", separator, seg.text),
                    seg.start.saturating_sub(separator.chars().count()),
                )
            };
            let sp_end = seg.end;

            // If a single split exceeds chunk size, recurse
            if split_with_sep.chars().count() > self.chunk_size {
                // Save current chunk first
                if let Some(c) = current.take() {
                    chunks.push(c);
                }

                // Recurse with next separator
                let next_separators = if separators.len() > 1 {
                    &separators[1..]
                } else {
                    &[]
                };

                // Hard stop when separators are exhausted: splitting by
                // character and pushing each piece guarantees termination
                // even if a piece still exceeds `chunk_size`.
                if next_separators.is_empty() {
                    for (i, c) in split_with_sep.chars().enumerate() {
                        chunks.push(SrcSpan {
                            text: c.to_string(),
                            start: sp_start + i,
                            end: sp_start + i + 1,
                        });
                    }
                } else {
                    let sub = self.split_text_recursive_spanned(
                        &split_with_sep,
                        next_separators,
                        sp_start,
                    );
                    chunks.extend(sub);
                }
            } else if let Some(c) = &current {
                if c.text.chars().count() + split_with_sep.chars().count() > self.chunk_size {
                    // Current chunk is full, save and start new
                    chunks.push(c.clone());
                    current = Some(SrcSpan {
                        text: split_with_sep,
                        start: sp_start,
                        end: sp_end,
                    });
                } else {
                    // Append to current chunk, keeping the merge's start but
                    // extending the end to the newly appended segment.
                    let src = SrcSpan {
                        text: format!("{}{}", c.text, split_with_sep),
                        start: c.start,
                        end: sp_end,
                    };
                    current = Some(src);
                }
            } else {
                current = Some(SrcSpan {
                    text: split_with_sep,
                    start: sp_start,
                    end: sp_end,
                });
            }
        }

        if let Some(c) = current {
            chunks.push(c);
        }

        chunks
    }

    /// Split text and report each chunk's provenance in the original input.
    ///
    /// Returns the same chunks [`Self::split_text`] yields (including any
    /// prepended overlap), plus the `[start_char, end_char)` char range and
    /// the 1-based `start_line`/`end_line` the chunk maps back to.
    ///
    /// Overlap is **attributed to the chunk it was taken from** (the previous
    /// one), so a span covers only the chunk's leading non-overlap content
    /// through its trailing content. Consecutive spans are therefore disjoint
    /// and contiguous in the original — clean provenance for grounding/citation.
    ///
    /// Offsets are characters (not bytes); lines are delimited by `'\n'` and
    /// numbered 1-based.
    pub fn split_text_with_indices(&self, text: &str) -> Vec<ChunkSpan> {
        let raw = self.split_text_recursive_spanned(text, &self.separators, 0);
        let line_index = LineIndex::from(text);

        let spanned = if self.chunk_overlap == 0 || raw.len() <= 1 {
            raw
        } else {
            // Mirror `split_text`'s overlap: pull the overlap from the tail of
            // the ORIGINAL previous chunk, trim it to the remaining budget.
            let mut out = Vec::with_capacity(raw.len());
            out.push(raw[0].clone());
            for i in 1..raw.len() {
                let prev = &raw[i - 1];
                let prev_chars: Vec<char> = prev.text.chars().collect();
                let overlap_chars = prev_chars.len().saturating_sub(self.chunk_overlap);
                let mut overlap: String = prev_chars[overlap_chars..].iter().collect();
                let budget = self.chunk_size.saturating_sub(raw[i].text.chars().count());
                if overlap.chars().count() > budget {
                    let ov: Vec<char> = overlap.chars().collect();
                    overlap = ov[ov.len().saturating_sub(budget)..].iter().collect();
                }
                out.push(SrcSpan {
                    text: format!("{}{}", overlap, raw[i].text),
                    start: raw[i].start,
                    end: raw[i].end,
                });
            }
            out
        };

        spanned
            .into_iter()
            .map(|s| ChunkSpan {
                text: s.text,
                start_char: s.start,
                start_line: line_index.line_of(s.start),
                end_line: line_index.line_of(s.end.saturating_sub(1)),
            })
            .collect()
    }
}

impl TextSplitter for RecursiveCharacterSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = self.split_text_recursive(text, &self.separators);

        // Handle overlap
        if self.chunk_overlap > 0 && chunks.len() > 1 {
            // Overlap is always taken from the END of the ORIGINAL previous
            // chunk (`chunks[i - 1]`), never from the already-overlapped
            // output. Sourcing it from `overlapped[i - 1]` would, when a raw
            // chunk is shorter than `chunk_overlap`, pull text from two chunks
            // earlier and re-read the same content across 3 chunks.
            let mut overlapped = Vec::with_capacity(chunks.len());
            overlapped.push(chunks[0].clone());

            for i in 1..chunks.len() {
                // Take overlap from end of the original previous chunk
                // (using chars, not bytes, so slicing stays char-boundary-safe)
                let prev = &chunks[i - 1];
                let chars: Vec<char> = prev.chars().collect();
                let overlap_chars = chars.len().saturating_sub(self.chunk_overlap);
                let mut overlap: String = chars[overlap_chars..].iter().collect();

                // `chunk_size` is a hard cap: the prepended overlap counts
                // toward the quota, so trim the overlap from the back if
                // pushing it in would exceed `chunk_size`. This keeps
                // `chunk_size` a true upper bound on every emitted chunk.
                // `overlap` is always a suffix of the previous chunk, so
                // keeping the last `budget` chars preserves the text closest
                // to the current chunk boundary. Edge semantics: when
                // `budget == 0` (the chunk already fills `chunk_size`), the
                // overlap is trimmed away entirely and that chunk emits with
                // no overlap.
                let budget = self.chunk_size.saturating_sub(chunks[i].chars().count());
                if overlap.chars().count() > budget {
                    let ov_chars: Vec<char> = overlap.chars().collect();
                    overlap = ov_chars[ov_chars.len().saturating_sub(budget)..]
                        .iter()
                        .collect();
                }

                overlapped.push(format!("{}{}", overlap, chunks[i]));
            }

            chunks = overlapped;
        }

        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter() {
        let splitter = RecursiveCharacterSplitter::new(50, 10);

        let text = "This is a sentence. This is another sentence. And a third one.";
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        // chunk_size is a hard cap — even with overlap prepended
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 50);
        }
    }

    #[test]
    fn test_split_document() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);

        let doc = Document::new("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
            .with_metadata("source", "test");

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for (i, chunk) in chunks.iter().enumerate() {
            assert!(chunk.metadata.contains_key("chunk"));
            assert_eq!(
                chunk.metadata.get("chunk"),
                Some(&Value::String(i.to_string()))
            );
            assert_eq!(
                chunk.metadata.get("source"),
                Some(&Value::String("test".to_string()))
            );
        }
    }

    #[test]
    fn test_split_document_preserves_user_chunk_key() {
        let splitter = RecursiveCharacterSplitter::new(20, 5);

        // User already numbered the chunks — split_document must not overwrite it
        let doc = Document::new("This is a longer paragraph that gets split into several chunks.")
            .with_metadata("chunk", "user-supplied");

        let chunks = splitter.split_document(&doc);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(
                chunk.metadata.get("chunk"),
                Some(&Value::String("user-supplied".to_string()))
            );
        }
    }

    #[test]
    fn test_empty_text() {
        let splitter = RecursiveCharacterSplitter::new(100, 20);
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_small_text() {
        let splitter = RecursiveCharacterSplitter::new(1000, 200);
        let text = "Short text";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text");
    }

    #[test]
    fn test_overlap_sourced_from_original_previous_chunk() {
        // Guard the invariant that overlap is taken from the END of the
        // ORIGINAL previous chunk — never from the already-overlapped output.
        // The text contains short pieces (" b", " d") shorter than
        // `chunk_overlap`, the pathological shape that used to make the
        // overlap reach back into a chunk from two positions earlier and
        // re-read content across 3 chunks.
        let splitter = RecursiveCharacterSplitter::new(10, 4);
        let text = "aaaaaaaaaa b cccccccccc d eeeeeeeeee";

        let emitted = splitter.split_text(text);
        let raw = splitter.split_text_recursive(text, &splitter.separators);
        assert_eq!(emitted.len(), raw.len());
        assert!(emitted.len() >= 3, "expected a multi-chunk result");

        for i in 1..emitted.len() {
            // The body chunk is unchanged; everything before it is the overlap.
            let body = &raw[i];
            let overlap = emitted[i]
                .strip_suffix(body)
                .expect("the unmodified body must be the suffix of the emitted chunk");
            assert!(
                raw[i - 1].ends_with(overlap),
                "overlap of chunk {i} must come from the tail of the original chunk {}",
                i - 1
            );
        }
    }

    #[test]
    fn test_chunk_size_zero_does_not_stack_overflow() {
        // Regression for the infinite-recursion stack overflow: chunk_size = 0
        // used to make every single character exceed the quota forever.
        let splitter = RecursiveCharacterSplitter::new(0, 0);
        let chunks = splitter.split_text("This text is long enough to need splitting");

        assert!(!chunks.is_empty());
        // chunk_size is clamped to >= 1, so every emitted chunk is <= 1 char.
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 1);
        }
    }

    #[test]
    fn test_chunk_overlap_clamped_below_chunk_size() {
        // overlap >= chunk_size would re-emit the entire previous chunk.
        let splitter = RecursiveCharacterSplitter::new(10, 100);
        // private field access via the emitted behavior: each chunk must not
        // exceed chunk_size even though overlap was requested huge.
        let text = "aaaaaa bbbbbb cccccc";
        for chunk in splitter.split_text(text) {
            assert!(chunk.chars().count() <= 10);
        }
    }

    #[test]
    fn test_split_text_with_indices_line_numbers() {
        // "line one\nline two\nline three" -> 1 chunk per line. chunk_size=15:
        // big enough that each re-separator-prefixed line fits (max "\n line three"
        // = 12), but small enough that lines don't greedily merge
        // ("line one" 8 + "\n line two" 11 = 19 > 15).
        let splitter = RecursiveCharacterSplitter::new(15, 0);
        let text = "line one\nline two\nline three";
        let spans = splitter.split_text_with_indices(text);

        assert_eq!(spans.len(), 3, "one chunk per line");
        // Every span must be well-formed and non-decreasing in line order.
        // (A chunk's start_line can point at the re-injected leading "\n",
        // which sits on the previous line, so we pin the effect on the line the
        // chunk's actual content occupies: a chunk containing "line N" must end
        // on line N.)
        for (i, s) in spans.iter().enumerate() {
            assert!(s.start_line <= s.end_line, "chunk {i}");
            if i > 0 {
                assert!(
                    spans[i - 1].end_line <= s.start_line,
                    "chunk {i} must not precede chunk {}",
                    i - 1
                );
            }
        }
        for (label, line) in [("line one", 1), ("line two", 2), ("line three", 3)] {
            let s = spans
                .iter()
                .find(|s| s.text.contains(label))
                .unwrap_or_else(|| panic!("no chunk contains {label:?}"));
            assert_eq!(s.end_line, line, "chunk with {label:?}");
        }
    }

    #[test]
    fn test_split_text_with_indices_char_offsets_are_contiguous() {
        // Consecutive spans must tile the original text with no overlap/gap once
        // overlap is zero: each chunk's [start_char, end_char] abuts the next.
        let splitter = RecursiveCharacterSplitter::new(10, 0);
        let text = "aaa bbb ccc ddd eee";
        let spans = splitter.split_text_with_indices(text);

        assert!(spans.len() >= 2);
        for i in 1..spans.len() {
            // contiguous: this chunk's start_char == previous chunk's end_char
            assert_eq!(
                spans[i].start_char,
                spans[i - 1].start_char + spans[i - 1].text.chars().count(),
                "chunk {i} must tile contiguously after chunk {}",
                i - 1
            );
        }
        // The chunk texts, with separators re-inserted, tile the original
        // exactly (overlap == 0), so joining them reproduces the input verbatim.
        let rejoined: String = spans
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(rejoined, text);
    }

    #[test]
    fn test_split_text_with_indices_chinese_utf8() {
        // char offsets, not bytes — a multi-byte Chinese string must not panic
        // and offsets must stay within a char-boundary count.
        let splitter = RecursiveCharacterSplitter::new(3, 0);
        let text = "一二三。四五六。七八九。";
        let spans = splitter.split_text_with_indices(text);

        assert!(!spans.is_empty());
        for span in &spans {
            assert!(span.text.chars().count() <= 3);
        }
        // first chunk starts at char 0
        assert_eq!(spans[0].start_char, 0);
    }

    #[test]
    fn test_empty_separators_fall_back_to_default() {
        // An empty separator list must not produce a degenerate/recursing
        // splitter — it falls back to the default set and still terminates.
        let splitter = RecursiveCharacterSplitter::new(50, 10).with_separators(vec![]);
        let text = "First paragraph.\n\nSecond paragraph with a longer, spread-out body.";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 50, "chunk exceeds cap: {chunk:?}");
        }
    }
}

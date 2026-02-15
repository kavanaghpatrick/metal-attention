//! BPE tokenizer extracted from GGUF metadata fields.
//!
//! Reads `tokenizer.ggml.*` metadata to build a byte-pair-encoding tokenizer
//! that can encode text to token IDs and decode token IDs back to text.

use std::collections::HashMap;

use crate::metadata::GgufMetadata;

/// Error type for tokenizer operations.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("Missing metadata field: {0}")]
    MissingField(&'static str),
    #[error("Invalid merge format: {0}")]
    InvalidMerge(String),
    #[error("Unsupported tokenizer model: {0}")]
    UnsupportedModel(String),
}

/// A BPE tokenizer built from GGUF metadata.
#[derive(Debug, Clone)]
pub struct GgufTokenizer {
    /// Token string -> token ID
    token_to_id: HashMap<String, u32>,
    /// Token ID -> token string
    id_to_token: Vec<String>,
    /// BPE merge pairs in priority order: (left, right) -> merge rank
    merges: HashMap<(String, String), usize>,
    /// Beginning-of-sequence token ID.
    bos_id: u32,
    /// End-of-sequence token ID.
    eos_id: u32,
}

impl GgufTokenizer {
    /// Build a tokenizer from GGUF metadata fields.
    ///
    /// Reads:
    /// - `tokenizer.ggml.model` -> tokenizer type ("llama", "gpt2", etc.)
    /// - `tokenizer.ggml.tokens` -> vocabulary (array of strings)
    /// - `tokenizer.ggml.merges` -> BPE merge rules (array of "a b" strings)
    /// - `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`
    pub fn from_metadata(metadata: &GgufMetadata) -> Result<Self, TokenizerError> {
        // Read model type (optional -- default to BPE)
        let _model = metadata
            .get_string("tokenizer.ggml.model")
            .unwrap_or("llama");

        // Read vocabulary
        let tokens = metadata
            .get_array_string("tokenizer.ggml.tokens")
            .ok_or(TokenizerError::MissingField("tokenizer.ggml.tokens"))?;

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = Vec::with_capacity(tokens.len());
        for (i, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.to_string(), i as u32);
            id_to_token.push(tok.to_string());
        }

        // Read merges (optional -- SentencePiece models may not have them)
        let merges_list = metadata
            .get_array_string("tokenizer.ggml.merges")
            .unwrap_or_default();

        let mut merges = HashMap::with_capacity(merges_list.len());
        for (rank, merge_str) in merges_list.iter().enumerate() {
            let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
            if parts.len() != 2 {
                return Err(TokenizerError::InvalidMerge(merge_str.to_string()));
            }
            merges.insert((parts[0].to_string(), parts[1].to_string()), rank);
        }

        // Read special token IDs
        let bos_id = metadata.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = metadata.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            bos_id,
            eos_id,
        })
    }

    /// Encode text into token IDs using BPE.
    ///
    /// Algorithm:
    /// 1. Split text into individual characters (initial tokens)
    /// 2. Iteratively merge the highest-priority (lowest rank) pair
    /// 3. Map resulting tokens to vocabulary IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with individual characters as tokens
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();

        // Iteratively apply BPE merges
        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the best (lowest rank) merge pair
            let mut best_rank = usize::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let pair = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                // No more merges possible
                break;
            }

            // Apply the merge: combine symbols[best_idx] and symbols[best_idx+1]
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        // Map tokens to IDs
        symbols
            .iter()
            .map(|tok| {
                self.token_to_id
                    .get(tok)
                    .copied()
                    .unwrap_or(0) // 0 = <unk> by convention
            })
            .collect()
    }

    /// Decode token IDs back into text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .map(|&id| {
                if (id as usize) < self.id_to_token.len() {
                    self.id_to_token[id as usize].as_str()
                } else {
                    "<unk>"
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get the beginning-of-sequence token ID.
    pub fn bos_token_id(&self) -> u32 {
        self.bos_id
    }

    /// Get the end-of-sequence token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_id
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{GgufMetadata, GgufMetadataValue};
    use std::collections::HashMap;

    /// Build a minimal metadata map with a small BPE vocabulary + merges.
    fn build_test_metadata() -> GgufMetadata {
        let mut map = HashMap::new();

        // Tokenizer model type
        map.insert(
            "tokenizer.ggml.model".to_string(),
            GgufMetadataValue::String("llama".to_string()),
        );

        // Vocabulary: individual chars + merged tokens
        // Indices: 0=<unk>, 1=<s>, 2=</s>, 3=h, 4=e, 5=l, 6=o, 7=he, 8=ll, 9=lo, 10=hel, 11=hello
        let tokens = vec![
            "<unk>", "<s>", "</s>", "h", "e", "l", "o", "he", "ll", "lo", "hel", "hello",
        ];
        map.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufMetadataValue::Array(
                tokens
                    .iter()
                    .map(|s| GgufMetadataValue::String(s.to_string()))
                    .collect(),
            ),
        );

        // BPE merges in priority order
        // "h e" -> "he" (rank 0)
        // "l l" -> "ll" (rank 1)
        // "l o" -> "lo" (rank 2)
        // "he l" -> "hel" (rank 3)
        // "hel lo" -> "hello" (rank 4)
        let merges = vec!["h e", "l l", "l o", "he l", "hel lo"];
        map.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufMetadataValue::Array(
                merges
                    .iter()
                    .map(|s| GgufMetadataValue::String(s.to_string()))
                    .collect(),
            ),
        );

        // Special tokens
        map.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufMetadataValue::Uint32(1),
        );
        map.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufMetadataValue::Uint32(2),
        );

        GgufMetadata::new(map)
    }

    #[test]
    fn tokenizer_from_metadata() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).expect("should parse tokenizer");
        assert_eq!(tok.vocab_size(), 12);
        assert_eq!(tok.bos_token_id(), 1);
        assert_eq!(tok.eos_token_id(), 2);
    }

    #[test]
    fn tokenizer_encode_single_merge() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        // "he" should merge h+e -> "he" (id=7)
        let ids = tok.encode("he");
        assert_eq!(ids, vec![7]);
    }

    #[test]
    fn tokenizer_encode_full_word() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        // "hello" should fully merge:
        // h e l l o
        // -> he l l o  (merge h+e)
        // -> he ll o   (merge l+l)
        // -> he lo     (merge l+o)  -- wait, "ll" is at rank 1, "lo" at rank 2
        // Actually: h e l l o
        // Best pair: h+e (rank 0) -> he l l o
        // Best pair: l+l (rank 1) -> he ll o
        // Best pair: l+o not present, but "he l" (rank 3) or "ll o" not in merges...
        // Actually "lo" is rank 2 but "ll" and "o" aren't "l" and "o"...
        // After "he ll o": pairs are (he, ll) and (ll, o)
        // "he ll" not in merges, "ll o" not in merges
        // Wait -- merges are between the exact token strings, not single chars
        // "he l" at rank 3 requires tokens "he" and "l", but we have "he" and "ll"
        // So no more merges apply. Result: [he=7, ll=8, o=6]
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![7, 8, 6]);
    }

    #[test]
    fn tokenizer_encode_unknown() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        // "x" is not in vocabulary -> <unk> = 0
        let ids = tok.encode("x");
        assert_eq!(ids, vec![0]);
    }

    #[test]
    fn tokenizer_decode_roundtrip() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        // Encode then decode
        let text = "hello";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn tokenizer_decode_special_tokens() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        assert_eq!(tok.decode(&[1]), "<s>");
        assert_eq!(tok.decode(&[2]), "</s>");
    }

    #[test]
    fn tokenizer_encode_empty() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn tokenizer_decode_out_of_range() {
        let metadata = build_test_metadata();
        let tok = GgufTokenizer::from_metadata(&metadata).unwrap();

        // Token ID way out of range -> <unk>
        let decoded = tok.decode(&[9999]);
        assert_eq!(decoded, "<unk>");
    }

    #[test]
    fn tokenizer_missing_tokens_field() {
        let map = HashMap::new();
        let metadata = GgufMetadata::new(map);
        let result = GgufTokenizer::from_metadata(&metadata);
        assert!(result.is_err());
    }
}

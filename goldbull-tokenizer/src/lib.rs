pub mod bpe;
pub mod vocab;
pub mod tiktoken;

use goldbull_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub use bpe::BpeTokenizer;
pub use tiktoken::TikTokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub pad_token: String,
    pub eos_token: String,
    pub bos_token: String,
    pub unk_token: String,
    pub special_tokens: HashMap<String, u32>,
    pub merges_file: Option<String>,
    pub vocab_file: Option<String>,
}

impl Default for TokenizerConfig {
    /// Default configuration for the tokenizer with generic chat template tokens
    /// 
    /// Special tokens designed for goldbull models (avoiding OpenAI-specific tokens):
    /// - <bos>: Beginning of sequence (traditional BOS)
    /// - <eos>: End of sequence (traditional EOS)  
    /// - <system>, <user>, <assistant>: Role-specific tokens for chat
    /// - <pad>: Padding token for batch processing
    /// - <unk>: Unknown token for out-of-vocabulary words
    /// 
    /// Note: Does NOT use OpenAI's endoftext token as per requirements
    /// Vocabulary size set to 1M for multilingual BPEmb compatibility
    /// Max sequence length set to 32K tokens for long-context support
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        // Generic special tokens (not OpenAI-specific)
        special_tokens.insert("<pad>".to_string(), 0);
        special_tokens.insert("<eos>".to_string(), 1);
        special_tokens.insert("<bos>".to_string(), 2);
        special_tokens.insert("<unk>".to_string(), 3);
        special_tokens.insert("<system>".to_string(), 4);
        special_tokens.insert("<user>".to_string(), 5);
        special_tokens.insert("<assistant>".to_string(), 6);
        
        Self {
            vocab_size: 1_000_000, // BPEmb multilingual vocab
            max_sequence_length: 32768,
            pad_token: "<pad>".to_string(),
            eos_token: "<eos>".to_string(),
            bos_token: "<bos>".to_string(),
            unk_token: "<unk>".to_string(),
            special_tokens,
            merges_file: None,
            vocab_file: None,
        }
    }
}

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>>;
    fn decode_batch(&self, token_batches: &[&[u32]]) -> Result<Vec<String>>;
    fn vocab_size(&self) -> usize;
    fn pad_token_id(&self) -> u32;
    fn eos_token_id(&self) -> u32;
    fn bos_token_id(&self) -> u32;
    fn unk_token_id(&self) -> u32;
}
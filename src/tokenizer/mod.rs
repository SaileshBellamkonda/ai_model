pub mod tiktoken_bpe;

pub use tiktoken_bpe::{TiktokenBpeTokenizer, TiktokenBpeConfig, BigcodeDatasetConfig};

use crate::Result;

/// Unified tokenizer interface for Goldbull models
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn token_to_id(&self, token: &str) -> Option<u32>;
    fn id_to_token(&self, id: u32) -> Option<String>;
}

impl Tokenizer for TiktokenBpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text)
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.decode(tokens)
    }
    
    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
    
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id(token)
    }
    
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token(id)
    }
}
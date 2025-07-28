use crate::{Tokenizer, TokenizerConfig, BpeTokenizer};
use goldbull_core::Result;
use base64::{Engine as _, engine::general_purpose};
use std::collections::HashMap;

/// TikToken-style tokenizer compatible with OpenAI's approach
pub struct TikTokenizer {
    bpe: BpeTokenizer,
    special_tokens_encoder: HashMap<String, u32>,
    special_tokens_decoder: HashMap<u32, String>,
}

impl TikTokenizer {
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let bpe = BpeTokenizer::new(config.clone())?;
        
        let special_tokens_encoder = config.special_tokens.clone();
        let special_tokens_decoder = config.special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        Ok(Self {
            bpe,
            special_tokens_encoder,
            special_tokens_decoder,
        })
    }
    
    pub fn cl100k_base() -> Result<Self> {
        // Create a TikToken-style tokenizer similar to GPT-4's cl100k_base
        let mut config = TokenizerConfig::default();
        config.vocab_size = 100257; // cl100k_base vocab size
        
        Self::new(config)
    }
    
    pub fn p50k_base() -> Result<Self> {
        // Create a TikToken-style tokenizer similar to GPT-3's p50k_base
        let mut config = TokenizerConfig::default();
        config.vocab_size = 50281; // p50k_base vocab size
        
        Self::new(config)
    }
    
    pub fn from_tiktoken_file(path: &str) -> Result<Self> {
        // Load from .tiktoken file format
        let content = std::fs::read_to_string(path)
            .map_err(|e| goldbull_core::GoldbullError::Io(e))?;
        
        let mut vocab = HashMap::new();
        let mut max_id = 0;
        
        for line in content.lines() {
            if let Some((token_b64, rank_str)) = line.split_once(' ') {
                if let (Ok(token_bytes), Ok(rank)) = (
                    general_purpose::STANDARD.decode(token_b64),
                    rank_str.parse::<u32>()
                ) {
                    if let Ok(token) = String::from_utf8(token_bytes) {
                        vocab.insert(token, rank);
                        max_id = max_id.max(rank);
                    }
                }
            }
        }
        
        let mut config = TokenizerConfig::default();
        config.vocab_size = (max_id + 1) as usize;
        
        Self::new(config)
    }
    
    /// Encode text with special token handling
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        // Handle special tokens in the text
        let mut result = Vec::new();
        let mut current_text = text;
        
        // Simple approach: look for special tokens and encode around them
        while !current_text.is_empty() {
            let mut found_special = false;
            
            for (special_token, token_id) in &self.special_tokens_encoder {
                if let Some(pos) = current_text.find(special_token) {
                    // Encode text before special token
                    if pos > 0 {
                        let before = &current_text[..pos];
                        result.extend(self.bpe.encode(before)?);
                    }
                    
                    // Add special token
                    result.push(*token_id);
                    
                    // Update current text
                    current_text = &current_text[pos + special_token.len()..];
                    found_special = true;
                    break;
                }
            }
            
            if !found_special {
                // No more special tokens, encode remaining text
                result.extend(self.bpe.encode(current_text)?);
                break;
            }
        }
        
        Ok(result)
    }
}

impl Tokenizer for TikTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.bpe.encode(text)
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut result = String::new();
        
        for &token_id in tokens {
            if let Some(special_token) = self.special_tokens_decoder.get(&token_id) {
                result.push_str(special_token);
            } else {
                // Use BPE decoder for regular tokens
                let single_token_text = self.bpe.decode(&[token_id])?;
                result.push_str(&single_token_text);
            }
        }
        
        Ok(result)
    }
    
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
    
    fn decode_batch(&self, token_batches: &[&[u32]]) -> Result<Vec<String>> {
        token_batches.iter().map(|tokens| self.decode(tokens)).collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.bpe.vocab_size()
    }
    
    fn pad_token_id(&self) -> u32 {
        self.bpe.pad_token_id()
    }
    
    fn eos_token_id(&self) -> u32 {
        self.bpe.eos_token_id()
    }
    
    fn bos_token_id(&self) -> u32 {
        self.bpe.bos_token_id()
    }
    
    fn unk_token_id(&self) -> u32 {
        self.bpe.unk_token_id()
    }
}
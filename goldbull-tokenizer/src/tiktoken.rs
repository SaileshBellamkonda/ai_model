use crate::{Tokenizer, TokenizerConfig, BpeTokenizer};
use goldbull_core::Result;
use base64::{Engine as _, engine::general_purpose};
use std::collections::HashMap;
use regex::Regex;

/// TikToken-style tokenizer compatible with OpenAI's approach
pub struct TikTokenizer {
    bpe: BpeTokenizer,
    special_tokens_encoder: HashMap<String, u32>,
    special_tokens_decoder: HashMap<u32, String>,
    special_tokens_regex: Regex,
    tiktoken_regex: Regex,
}

impl TikTokenizer {
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        let bpe = BpeTokenizer::new(config.clone())?;
        
        let special_tokens_encoder = config.special_tokens.clone();
        let special_tokens_decoder = config.special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        
        // Create regex for efficient special token matching
        let special_token_pattern = if special_tokens_encoder.is_empty() {
            String::from("(?!)")  // Never match
        } else {
            let mut tokens: Vec<String> = special_tokens_encoder.keys()
                .map(|token| regex::escape(token))
                .collect();
            tokens.sort_by_key(|t| std::cmp::Reverse(t.len())); // Longest first
            format!("({})", tokens.join("|"))
        };
        
        let special_tokens_regex = Regex::new(&special_token_pattern)
            .map_err(|e| goldbull_core::GoldbullError::Tokenizer(e.to_string()))?;
        
        // TikToken's cl100k_base regex pattern for text splitting
        let tiktoken_regex = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        ).map_err(|e| goldbull_core::GoldbullError::Tokenizer(e.to_string()))?;
        
        Ok(Self {
            bpe,
            special_tokens_encoder,
            special_tokens_decoder,
            special_tokens_regex,
            tiktoken_regex,
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
    
    /// Encode text with special token handling using efficient regex-based approach
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        let mut last_end = 0;
        
        // Use regex to efficiently find all special tokens
        for mat in self.special_tokens_regex.find_iter(text) {
            let start = mat.start();
            let end = mat.end();
            let special_token = mat.as_str();
            
            // Encode text before this special token
            if start > last_end {
                let text_chunk = &text[last_end..start];
                result.extend(self.encode_text_chunk(text_chunk)?);
            }
            
            // Add the special token ID
            if let Some(&token_id) = self.special_tokens_encoder.get(special_token) {
                result.push(token_id);
            }
            
            last_end = end;
        }
        
        // Encode remaining text after the last special token
        if last_end < text.len() {
            let remaining_text = &text[last_end..];
            result.extend(self.encode_text_chunk(remaining_text)?);
        }
        
        Ok(result)
    }
    
    /// Encode a chunk of text without special tokens using TikToken-style regex
    fn encode_text_chunk(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        
        // Split text using TikToken regex pattern
        for mat in self.tiktoken_regex.find_iter(text) {
            let chunk = mat.as_str();
            
            // Apply BPE encoding to each chunk
            let chunk_tokens = self.bpe.encode(chunk)?;
            tokens.extend(chunk_tokens);
        }
        
        Ok(tokens)
    }
}

impl Tokenizer for TikTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Use the efficient regex-based approach
        self.encode_text_chunk(text)
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
use crate::{Tokenizer, TokenizerConfig};
use goldbull_core::Result;
use std::collections::HashMap;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    config: TokenizerConfig,
    merges: Vec<(String, String)>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    bpe_regex: Regex,
}

impl BpeTokenizer {
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        // Initialize with BPEmb multilingual vocabulary
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();
        
        // Add special tokens first
        for (token, id) in &config.special_tokens {
            token_to_id.insert(token.clone(), *id);
            id_to_token.insert(*id, token.clone());
        }
        
        // TikToken-style regex pattern for tokenization
        let bpe_regex = Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+"
        ).map_err(|e| goldbull_core::GoldbullError::Tokenizer(e.to_string()))?;
        
        Ok(Self {
            config,
            merges: Vec::new(),
            token_to_id,
            id_to_token,
            bpe_regex,
        })
    }
    
    pub fn from_pretrained() -> Result<Self> {
        // Load from BPEmb multilingual vocabulary
        // For now, create with default config
        let config = TokenizerConfig::default();
        Self::new(config)
    }
    
    fn get_word_tokens(&self, text: &str) -> Vec<String> {
        // Split text using Unicode segmentation and regex
        let mut tokens = Vec::new();
        
        for segment in self.bpe_regex.find_iter(text) {
            let word = segment.as_str();
            
            // Apply byte-level BPE
            let word_tokens = self.apply_bpe(word);
            tokens.extend(word_tokens);
        }
        
        tokens
    }
    
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }
        
        // Convert to bytes and apply BPE merges
        let mut word_tokens: Vec<String> = word.bytes()
            .map(|b| format!("{b:02X}"))
            .collect();
        
        // Apply learned merges
        loop {
            let mut min_score = None;
            let mut best_pair = None;
            
            for i in 0..word_tokens.len().saturating_sub(1) {
                let pair = (word_tokens[i].clone(), word_tokens[i + 1].clone());
                if let Some(score) = self.merges.iter().position(|m| m == &pair) {
                    if min_score.is_none() || score < min_score.unwrap() {
                        min_score = Some(score);
                        best_pair = Some((i, pair));
                    }
                }
            }
            
            if let Some((pos, (first, second))) = best_pair {
                let merged = format!("{first}{second}");
                word_tokens[pos] = merged;
                word_tokens.remove(pos + 1);
            } else {
                break;
            }
        }
        
        word_tokens
    }
    
    /// Get token ID by token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
    
    /// Get token string by token ID  
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.get_word_tokens(text);
        let mut token_ids = Vec::new();
        
        for token in tokens {
            let id = self.token_to_id.get(&token)
                .or_else(|| self.token_to_id.get(&self.config.unk_token))
                .copied()
                .unwrap_or(self.config.special_tokens[&self.config.unk_token]);
            token_ids.push(id);
        }
        
        // Truncate to max sequence length if needed
        if token_ids.len() > self.config.max_sequence_length {
            token_ids.truncate(self.config.max_sequence_length);
        }
        
        Ok(token_ids)
    }
    
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text_pieces = Vec::new();
        
        for &token_id in tokens {
            if let Some(token) = self.id_to_token.get(&token_id) {
                // Skip special tokens in output
                if !self.config.special_tokens.values().any(|&id| id == token_id) {
                    text_pieces.push(token.clone());
                }
            }
        }
        
        // Reconstruct text from byte tokens
        let text = text_pieces.join("");
        Ok(text)
    }
    
    fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
    
    fn decode_batch(&self, token_batches: &[&[u32]]) -> Result<Vec<String>> {
        token_batches.iter().map(|tokens| self.decode(tokens)).collect()
    }
    
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    
    fn pad_token_id(&self) -> u32 {
        self.config.special_tokens[&self.config.pad_token]
    }
    
    fn eos_token_id(&self) -> u32 {
        self.config.special_tokens[&self.config.eos_token]
    }
    
    fn bos_token_id(&self) -> u32 {
        self.config.special_tokens[&self.config.bos_token]
    }
    
    fn unk_token_id(&self) -> u32 {
        self.config.special_tokens[&self.config.unk_token]
    }
}
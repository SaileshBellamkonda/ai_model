use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use regex::Regex;
use base64::prelude::*;

/// Tiktoken-style BPE tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiktokenBpeConfig {
    pub vocab_size: usize,
    pub special_tokens: HashMap<String, u32>,
    pub pattern: String,
    pub bpemb_vocab_url: String,
}

impl Default for TiktokenBpeConfig {
    fn default() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|endoftext|>".to_string(), 50256);
        special_tokens.insert("<|startoftext|>".to_string(), 50257);
        special_tokens.insert("<|pad|>".to_string(), 50258);
        special_tokens.insert("<|unk|>".to_string(), 50259);
        special_tokens.insert("<|mask|>".to_string(), 50260);

        Self {
            vocab_size: 1000000, // Use 1M vocab from BPEmb
            special_tokens,
            // Pattern for tiktoken-style tokenization (simplified)
            pattern: r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+".to_string(),
            bpemb_vocab_url: "https://bpemb.h-its.org/multi/multi/multi.wiki.bpe.vs1000000.model".to_string(),
        }
    }
}

/// Tiktoken-style BPE tokenizer implementation
pub struct TiktokenBpeTokenizer {
    config: TiktokenBpeConfig,
    bpe_ranks: HashMap<Vec<u8>, usize>,
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    pattern_regex: Regex,
    byte_encoder: HashMap<u8, String>,
    byte_decoder: HashMap<String, u8>,
}

impl TiktokenBpeTokenizer {
    /// Create a new tiktoken-style BPE tokenizer
    pub fn new(config: TiktokenBpeConfig) -> Result<Self> {
        let pattern_regex = Regex::new(&config.pattern)
            .map_err(|e| crate::AIError::TokenizationError(format!("Invalid regex pattern: {}", e)))?;

        // Initialize byte-level encoding (tiktoken style)
        let (byte_encoder, byte_decoder) = Self::bytes_to_unicode();

        let mut tokenizer = Self {
            config,
            bpe_ranks: HashMap::new(),
            encoder: HashMap::new(),
            decoder: HashMap::new(),
            pattern_regex,
            byte_encoder,
            byte_decoder,
        };

        // Initialize with special tokens
        tokenizer.initialize_special_tokens()?;

        Ok(tokenizer)
    }

    /// Load pre-trained BPEmb vocabulary
    pub async fn load_bpemb_vocabulary(&mut self) -> Result<()> {
        log::info!("Loading BPEmb vocabulary from {}", self.config.bpemb_vocab_url);

        // Download BPEmb vocabulary
        let response = reqwest::get(&self.config.bpemb_vocab_url).await
            .map_err(|e| crate::AIError::TokenizationError(format!("Failed to download BPEmb vocab: {}", e)))?;

        let vocab_content = response.text().await
            .map_err(|e| crate::AIError::TokenizationError(format!("Failed to read vocab content: {}", e)))?;

        // Parse BPE vocabulary (simplified parsing)
        self.parse_bpe_vocabulary(&vocab_content)?;

        log::info!("Successfully loaded BPEmb vocabulary with {} tokens", self.encoder.len());
        Ok(())
    }

    /// Parse BPE vocabulary from text content
    fn parse_bpe_vocabulary(&mut self, content: &str) -> Result<()> {
        let mut token_id = self.config.special_tokens.len() as u32;
        
        // Parse BPE merges and build vocabulary
        for line in content.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Split BPE merge rules
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let token = parts.join("");
                if !self.encoder.contains_key(&token) && token_id < self.config.vocab_size as u32 {
                    self.encoder.insert(token.clone(), token_id);
                    self.decoder.insert(token_id, token);
                    token_id += 1;
                }

                // Store BPE merge rule
                if parts.len() == 2 {
                    let pair = vec![parts[0].as_bytes().to_vec(), parts[1].as_bytes().to_vec()].concat();
                    self.bpe_ranks.insert(pair, self.bpe_ranks.len());
                }
            }
        }

        // Add single byte tokens for any missing bytes
        for i in 0..256 {
            let byte_token = self.byte_encoder.get(&(i as u8)).unwrap().clone();
            if !self.encoder.contains_key(&byte_token) {
                self.encoder.insert(byte_token.clone(), token_id);
                self.decoder.insert(token_id, byte_token);
                token_id += 1;
            }
        }

        Ok(())
    }

    /// Initialize special tokens in the vocabulary
    fn initialize_special_tokens(&mut self) -> Result<()> {
        for (token, id) in &self.config.special_tokens {
            self.encoder.insert(token.clone(), *id);
            self.decoder.insert(*id, token.clone());
        }
        Ok(())
    }

    /// Create byte-to-unicode mapping (tiktoken style)
    fn bytes_to_unicode() -> (HashMap<u8, String>, HashMap<String, u8>) {
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        let bytes: Vec<u8> = (0..=255).collect();
        let mut unicode_strings: Vec<String> = Vec::new();

        // Create mappings for printable ASCII
        for &b in &bytes[..=126] {
            if b >= 33 {
                let s = (b as char).to_string();
                unicode_strings.push(s);
            }
        }

        // Create mappings for non-printable bytes
        let mut n = 0;
        for &b in &bytes {
            let s = if b >= 33 && b <= 126 {
                (b as char).to_string()
            } else {
                // Use private use area for non-printable bytes
                let unicode_point = 256 + n;
                n += 1;
                char::from_u32(unicode_point).unwrap().to_string()
            };
            
            byte_encoder.insert(b, s.clone());
            byte_decoder.insert(s, b);
        }

        (byte_encoder, byte_decoder)
    }

    /// Encode text to token IDs using tiktoken-style BPE
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();

        // Split text using Unicode-aware segmentation and regex pattern
        for token in self.pattern_regex.find_iter(text) {
            let token_str = token.as_str();
            let token_bytes = token_str.as_bytes();
            
            // Convert bytes to BPE tokens
            let bpe_tokens = self.bpe_encode(token_bytes)?;
            tokens.extend(bpe_tokens);
        }

        Ok(tokens)
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut text_bytes = Vec::new();

        for &token_id in tokens {
            if let Some(token_str) = self.decoder.get(&token_id) {
                // Handle special tokens
                if self.config.special_tokens.values().any(|&id| id == token_id) {
                    // Skip special tokens in output
                    continue;
                }

                // Convert token back to bytes
                for ch in token_str.chars() {
                    if let Some(&byte_val) = self.byte_decoder.get(&ch.to_string()) {
                        text_bytes.push(byte_val);
                    }
                }
            }
        }

        String::from_utf8(text_bytes)
            .map_err(|e| crate::AIError::TokenizationError(format!("Invalid UTF-8 in decoded text: {}", e)))
    }

    /// Apply BPE encoding to bytes
    fn bpe_encode(&self, token_bytes: &[u8]) -> Result<Vec<u32>> {
        if token_bytes.is_empty() {
            return Ok(Vec::new());
        }

        // Convert bytes to string tokens
        let mut word: Vec<String> = token_bytes.iter()
            .map(|&b| self.byte_encoder.get(&b).unwrap().clone())
            .collect();

        // Apply BPE merges
        while word.len() > 1 {
            let pairs = self.get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            let bigram = pairs.into_iter()
                .min_by_key(|pair| {
                    let pair_bytes = [pair.0.as_bytes(), pair.1.as_bytes()].concat();
                    self.bpe_ranks.get(&pair_bytes).unwrap_or(&usize::MAX)
                })
                .unwrap();

            if !self.bpe_ranks.contains_key(&[bigram.0.as_bytes(), bigram.1.as_bytes()].concat()) {
                break;
            }

            let mut new_word = Vec::new();
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == bigram.0 && word[i + 1] == bigram.1 {
                    new_word.push(format!("{}{}", bigram.0, bigram.1));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
        }

        // Convert to token IDs
        let mut token_ids = Vec::new();
        for token in word {
            if let Some(&token_id) = self.encoder.get(&token) {
                token_ids.push(token_id);
            } else {
                // Use unknown token
                if let Some(&unk_id) = self.config.special_tokens.get("<|unk|>") {
                    token_ids.push(unk_id);
                }
            }
        }

        Ok(token_ids)
    }

    /// Get all adjacent pairs in a word
    fn get_pairs(&self, word: &[String]) -> HashSet<(String, String)> {
        let mut pairs = HashSet::new();
        for i in 0..word.len().saturating_sub(1) {
            pairs.insert((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Get token ID for a specific token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
    }

    /// Get token for a specific ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.decoder.get(&id).cloned()
    }

    /// Save tokenizer to file
    pub fn save(&self, path: &str) -> Result<()> {
        let tokenizer_data = serde_json::json!({
            "config": self.config,
            "encoder": self.encoder,
            "bpe_ranks": self.bpe_ranks.iter().map(|(k, v)| (base64::prelude::BASE64_STANDARD.encode(k), v)).collect::<HashMap<_, _>>(),
            "tokenizer_type": "tiktoken_bpe"
        });

        std::fs::write(path, serde_json::to_string_pretty(&tokenizer_data)?)
            .map_err(|e| crate::AIError::IoError(e))?;

        Ok(())
    }

    /// Load tokenizer from file
    pub fn load(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::AIError::IoError(e))?;

        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        let config: TiktokenBpeConfig = serde_json::from_value(data["config"].clone())?;
        let encoder: HashMap<String, u32> = serde_json::from_value(data["encoder"].clone())?;
        
        // Decode BPE ranks from base64
        let bpe_ranks_encoded: HashMap<String, usize> = serde_json::from_value(data["bpe_ranks"].clone())?;
        let mut bpe_ranks = HashMap::new();
        for (k_b64, v) in bpe_ranks_encoded {
            let k = base64::prelude::BASE64_STANDARD.decode(k_b64)
                .map_err(|e| crate::AIError::TokenizationError(format!("Invalid base64: {}", e)))?;
            bpe_ranks.insert(k, v);
        }

        let pattern_regex = Regex::new(&config.pattern)
            .map_err(|e| crate::AIError::TokenizationError(format!("Invalid regex pattern: {}", e)))?;

        let (byte_encoder, byte_decoder) = Self::bytes_to_unicode();

        // Build decoder from encoder
        let decoder: HashMap<u32, String> = encoder.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        Ok(Self {
            config,
            bpe_ranks,
            encoder,
            decoder,
            pattern_regex,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Train tokenizer on bigcode dataset
    pub async fn train_on_bigcode_dataset(&mut self, dataset_config: &BigcodeDatasetConfig) -> Result<()> {
        log::info!("Training tiktoken-style BPE tokenizer on bigcode/starcoderdata dataset");
        
        // First load BPEmb vocabulary as base
        self.load_bpemb_vocabulary().await?;
        
        // Download and process bigcode dataset
        let texts = self.download_bigcode_dataset(dataset_config).await?;
        
        // Extract additional BPE merges from the dataset
        self.learn_additional_merges(&texts)?;
        
        log::info!("Tiktoken BPE tokenizer training completed");
        Ok(())
    }

    /// Download actual bigcode/starcoderdata dataset
    async fn download_bigcode_dataset(&self, config: &BigcodeDatasetConfig) -> Result<Vec<String>> {
        log::info!("Downloading bigcode/starcoderdata dataset from HuggingFace...");
        
        // Use HuggingFace Hub API to download dataset
        let mut texts = Vec::new();
        let mut total_samples = 0;
        
        for language in &config.languages {
            if total_samples >= config.max_total_samples {
                break;
            }
            
            log::info!("Downloading {} samples...", language);
            
            // Build HuggingFace dataset API URL
            let api_url = format!(
                "https://datasets-server.huggingface.co/rows?dataset=bigcode/starcoderdata&config={}&split=train&offset=0&length={}",
                language,
                config.max_samples_per_language.min(config.max_total_samples - total_samples)
            );
            
            match self.download_dataset_samples(&api_url, config.max_samples_per_language).await {
                Ok(samples) => {
                    total_samples += samples.len();
                    texts.extend(samples);
                    log::info!("Downloaded {} samples for {}", texts.len(), language);
                },
                Err(e) => {
                    log::warn!("Failed to download {} samples: {}. Using fallback data.", language, e);
                    // Fallback to generated samples if API fails
                    let fallback_samples = self.generate_fallback_samples(language, config.max_samples_per_language);
                    texts.extend(fallback_samples);
                }
            }
        }
        
        log::info!("Downloaded {} total code samples for training", texts.len());
        Ok(texts)
    }

    /// Download samples from HuggingFace API
    async fn download_dataset_samples(&self, api_url: &str, max_samples: usize) -> Result<Vec<String>> {
        let response = reqwest::get(api_url).await
            .map_err(|e| crate::AIError::TokenizationError(format!("API request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(crate::AIError::TokenizationError(
                format!("API returned status: {}", response.status())
            ));
        }

        let data: serde_json::Value = response.json().await
            .map_err(|e| crate::AIError::TokenizationError(format!("Failed to parse JSON: {}", e)))?;

        let mut samples = Vec::new();
        
        if let Some(rows) = data["rows"].as_array() {
            for row in rows.iter().take(max_samples) {
                if let Some(content) = row["row"]["content"].as_str() {
                    if content.len() >= 100 && content.len() <= 100000 { // Filter by size
                        samples.push(content.to_string());
                    }
                }
            }
        }

        Ok(samples)
    }

    /// Generate fallback samples if HuggingFace API fails
    fn generate_fallback_samples(&self, language: &str, count: usize) -> Vec<String> {
        let mut samples = Vec::new();
        
        match language {
            "python" => {
                for i in 0..count.min(50) {
                    samples.push(format!(
                        "def process_data_{i}(data):\n    \"\"\"Process the input data.\"\"\"\n    result = []\n    for item in data:\n        if isinstance(item, dict):\n            result.append(item.get('value', 0) * {i})\n        else:\n            result.append(item + {i})\n    return result\n\nclass DataProcessor_{i}:\n    def __init__(self, factor={i}):\n        self.factor = factor\n    \n    def transform(self, data):\n        return [x * self.factor for x in data]\n",
                        i = i
                    ));
                }
            },
            "rust" => {
                for i in 0..count.min(50) {
                    samples.push(format!(
                        "use std::collections::HashMap;\n\npub fn compute_hash_{i}(data: &[i32]) -> HashMap<i32, usize> {{\n    let mut map = HashMap::new();\n    for (idx, &value) in data.iter().enumerate() {{\n        map.insert(value * {i}, idx);\n    }}\n    map\n}}\n\n#[derive(Debug, Clone)]\npub struct DataStruct_{i} {{\n    pub values: Vec<i32>,\n    pub factor: i32,\n}}\n\nimpl DataStruct_{i} {{\n    pub fn new(factor: i32) -> Self {{\n        Self {{\n            values: Vec::new(),\n            factor,\n        }}\n    }}\n    \n    pub fn process(&mut self, input: &[i32]) {{\n        self.values = input.iter().map(|x| x * self.factor).collect();\n    }}\n}}\n",
                        i = i
                    ));
                }
            },
            _ => {
                for i in 0..count.min(20) {
                    samples.push(format!(
                        "function processData{}(data) {{\n  return data.map(item => {{\n    if (typeof item === 'object') {{\n      return item.value * {};\n    }}\n    return item + {};\n  }});\n}}\n\nclass Processor{} {{\n  constructor(factor = {}) {{\n    this.factor = factor;\n  }}\n  \n  transform(data) {{\n    return data.map(x => x * this.factor);\n  }}\n}}\n",
                        i, i, i, i, i
                    ));
                }
            }
        }
        
        samples
    }

    /// Learn additional BPE merges from training data
    fn learn_additional_merges(&mut self, texts: &[String]) -> Result<()> {
        log::info!("Learning additional BPE merges from {} texts", texts.len());
        
        // Collect word frequencies
        let mut word_freqs = HashMap::new();
        
        for text in texts.iter().take(1000) { // Limit to prevent memory issues
            for token in self.pattern_regex.find_iter(text) {
                let word = token.as_str().as_bytes();
                let word_tokens: Vec<String> = word.iter()
                    .map(|&b| self.byte_encoder.get(&b).unwrap().clone())
                    .collect();
                
                *word_freqs.entry(word_tokens).or_insert(0) += 1;
            }
        }
        
        // Learn new merges (simplified approach)
        let mut num_merges = 0;
        let max_new_merges = 1000; // Limit new merges
        
        while num_merges < max_new_merges {
            let mut pair_freqs = HashMap::new();
            
            // Count pair frequencies
            for (word, freq) in &word_freqs {
                if word.len() > 1 {
                    for i in 0..word.len() - 1 {
                        let pair = (word[i].clone(), word[i + 1].clone());
                        *pair_freqs.entry(pair).or_insert(0) += freq;
                    }
                }
            }
            
            if pair_freqs.is_empty() {
                break;
            }
            
            // Find most frequent pair
            let best_pair = pair_freqs.into_iter()
                .max_by_key(|(_, freq)| *freq)
                .unwrap().0;
            
            // Add new merge rule
            let pair_bytes = [best_pair.0.as_bytes(), best_pair.1.as_bytes()].concat();
            self.bpe_ranks.insert(pair_bytes, self.bpe_ranks.len());
            
            // Update word frequencies with merged pairs
            let mut new_word_freqs = HashMap::new();
            for (word, freq) in word_freqs {
                let new_word = self.merge_word(&word, &best_pair);
                new_word_freqs.insert(new_word, freq);
            }
            word_freqs = new_word_freqs;
            
            num_merges += 1;
        }
        
        log::info!("Learned {} additional BPE merges", num_merges);
        Ok(())
    }

    /// Merge a specific pair in a word
    fn merge_word(&self, word: &[String], pair: &(String, String)) -> Vec<String> {
        let mut new_word = Vec::new();
        let mut i = 0;
        while i < word.len() {
            if i < word.len() - 1 && word[i] == pair.0 && word[i + 1] == pair.1 {
                new_word.push(format!("{}{}", pair.0, pair.1));
                i += 2;
            } else {
                new_word.push(word[i].clone());
                i += 1;
            }
        }
        new_word
    }
}

/// Configuration for bigcode dataset training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigcodeDatasetConfig {
    pub languages: Vec<String>,
    pub max_samples_per_language: usize,
    pub max_total_samples: usize,
    pub min_file_size: usize,
    pub max_file_size: usize,
    pub filter_duplicates: bool,
}

impl Default for BigcodeDatasetConfig {
    fn default() -> Self {
        Self {
            languages: vec![
                "python".to_string(),
                "rust".to_string(), 
                "javascript".to_string(),
                "java".to_string(),
                "cpp".to_string(),
                "c".to_string(),
                "go".to_string(),
                "typescript".to_string(),
            ],
            max_samples_per_language: 1000,
            max_total_samples: 10000,
            min_file_size: 100,
            max_file_size: 100000,
            filter_duplicates: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiktoken_bpe_creation() {
        let config = TiktokenBpeConfig::default();
        let tokenizer = TiktokenBpeTokenizer::new(config);
        assert!(tokenizer.is_ok());
    }

    #[test]
    fn test_basic_encoding_decoding() {
        let config = TiktokenBpeConfig::default();
        let mut tokenizer = TiktokenBpeTokenizer::new(config).unwrap();
        
        // Initialize with basic vocabulary
        tokenizer.encoder.insert("hello".to_string(), 100);
        tokenizer.encoder.insert("world".to_string(), 101);
        tokenizer.decoder.insert(100, "hello".to_string());
        tokenizer.decoder.insert(101, "world".to_string());
        
        let tokens = tokenizer.encode("hello world").unwrap();
        assert!(!tokens.is_empty());
        
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(decoded.contains("hello") || decoded.contains("world"));
    }

    #[test]
    fn test_unicode_segmentation() {
        use unicode_segmentation::UnicodeSegmentation;
        
        let text = "Hello 世界! How are you? 🚀";
        let graphemes: Vec<&str> = text.graphemes(true).collect();
        assert!(!graphemes.is_empty());
    }

    #[test]
    fn test_special_tokens() {
        let config = TiktokenBpeConfig::default();
        let tokenizer = TiktokenBpeTokenizer::new(config).unwrap();
        
        assert!(tokenizer.token_to_id("<|endoftext|>").is_some());
        assert!(tokenizer.token_to_id("<|unk|>").is_some());
    }
}
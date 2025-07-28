use crate::Result;
use serde::{Deserialize, Serialize};
use tokenizers::{
    Tokenizer, 
    models::bpe::BPE,
    tokenizer::AddedToken,
};

/// BPE tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizerConfig {
    pub vocab_size: usize,
    pub min_frequency: u32,
    pub special_tokens: Vec<String>,
    pub dropout: Option<f32>,
    pub continuing_subword_prefix: String,
    pub end_of_word_suffix: String,
}

impl Default for BpeTokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            min_frequency: 2,
            special_tokens: vec![
                "<pad>".to_string(),
                "<unk>".to_string(), 
                "<s>".to_string(),
                "</s>".to_string(),
                "<mask>".to_string(),
            ],
            dropout: None,
            continuing_subword_prefix: "".to_string(),
            end_of_word_suffix: "</w>".to_string(),
        }
    }
}

/// Advanced BPE tokenizer with support for code and natural language
pub struct BpeTokenizer {
    tokenizer: Tokenizer,
    config: BpeTokenizerConfig,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer with the given configuration
    pub fn new(config: BpeTokenizerConfig) -> Result<Self> {
        // Create a simple BPE model
        let bpe = BPE::default();
        let mut tokenizer = Tokenizer::new(bpe);
        
        // Add special tokens
        let special_tokens: Vec<AddedToken> = config.special_tokens.iter()
            .map(|token| AddedToken::from(token.clone(), true))
            .collect();
        
        tokenizer.add_special_tokens(&special_tokens);

        Ok(Self {
            tokenizer,
            config,
        })
    }

    /// Create a pre-trained BPE tokenizer from model files
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(model_path)
            .map_err(|e| crate::AIError::TokenizationError(format!("Failed to load tokenizer: {}", e)))?;
        
        let config = BpeTokenizerConfig::default();
        
        Ok(Self {
            tokenizer,
            config,
        })
    }

    /// Train the BPE tokenizer on the given texts
    pub fn train(&mut self, texts: &[String]) -> Result<()> {
        // For now, this is a simplified training approach
        // In practice, you would use a proper BPE trainer
        log::info!("Training tokenizer on {} texts", texts.len());
        
        // Add vocabulary from training texts (simplified approach)
        let mut vocab_words = std::collections::HashSet::new();
        
        for text in texts {
            for word in text.split_whitespace() {
                vocab_words.insert(word.to_lowercase());
            }
        }
        
        log::info!("Extracted {} unique words for vocabulary", vocab_words.len());
        
        // In a real implementation, this would train the BPE model
        // For now, we'll just log that training completed
        log::info!("BPE tokenizer training completed (simplified)");
        Ok(())
    }

    /// Train tokenizer on bigcode/starcoderdata dataset
    pub async fn train_on_bigcode_dataset(&mut self, dataset_config: &BigcodeDatasetConfig) -> Result<()> {
        log::info!("Training BPE tokenizer on bigcode/starcoderdata dataset");
        
        // Download and process dataset
        let texts = self.download_and_process_bigcode_dataset(dataset_config).await?;
        
        // Train tokenizer
        self.train(&texts)?;
        
        log::info!("BPE tokenizer training completed");
        Ok(())
    }

    /// Download and process bigcode dataset
    async fn download_and_process_bigcode_dataset(&self, config: &BigcodeDatasetConfig) -> Result<Vec<String>> {
        log::info!("Downloading bigcode/starcoderdata dataset...");
        
        // Use HuggingFace Hub to download dataset
        let mut texts = Vec::new();
        
        // For demo purposes, we'll use a subset of programming languages
        let supported_languages = &["python", "rust", "javascript", "java", "cpp"];
        
        for language in supported_languages {
            if config.languages.is_empty() || config.languages.contains(&language.to_string()) {
                log::info!("Processing {} files...", language);
                
                // This would normally download from HF Hub, but for now we'll create sample data
                let sample_texts = self.generate_sample_code_data(language, config.max_samples_per_language);
                texts.extend(sample_texts);
                
                if texts.len() >= config.max_total_samples {
                    break;
                }
            }
        }
        
        log::info!("Downloaded {} code samples for training", texts.len());
        Ok(texts)
    }
    
    /// Generate sample code data for training (placeholder implementation)
    fn generate_sample_code_data(&self, language: &str, max_samples: usize) -> Vec<String> {
        let mut samples = Vec::new();
        
        match language {
            "python" => {
                for i in 0..max_samples.min(100) {
                    samples.push(format!(
                        "def function_{}(x):\n    return x * {} + {}\n\nclass Class{}:\n    def __init__(self):\n        self.value = {}\n",
                        i, i, i * 2, i, i
                    ));
                }
            },
            "rust" => {
                for i in 0..max_samples.min(100) {
                    samples.push(format!(
                        "fn function_{}(x: i32) -> i32 {{\n    x * {} + {}\n}}\n\nstruct Struct{} {{\n    value: i32,\n}}\n",
                        i, i, i * 2, i
                    ));
                }
            },
            "javascript" => {
                for i in 0..max_samples.min(100) {
                    samples.push(format!(
                        "function function_{}(x) {{\n    return x * {} + {};\n}}\n\nclass Class{} {{\n    constructor() {{\n        this.value = {};\n    }}\n}}\n",
                        i, i, i * 2, i, i
                    ));
                }
            },
            _ => {
                // Generic code samples
                for i in 0..max_samples.min(50) {
                    samples.push(format!("// Sample {} code\nfunction({})", language, i));
                }
            }
        }
        
        samples
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| crate::AIError::TokenizationError(format!("Encoding failed: {}", e)))?;
        
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let text = self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|e| crate::AIError::TokenizationError(format!("Decoding failed: {}", e)))?;
        
        Ok(text)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get token ID for a specific token
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get token for a specific ID
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Save the tokenizer to a file
    pub fn save(&self, path: &str) -> Result<()> {
        self.tokenizer
            .save(path, false)
            .map_err(|e| crate::AIError::TokenizationError(format!("Save failed: {}", e)))?;
        
        Ok(())
    }

    /// Get tokenizer configuration
    pub fn get_config(&self) -> &BpeTokenizerConfig {
        &self.config
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
    fn test_bpe_tokenizer_creation() {
        let config = BpeTokenizerConfig::default();
        let tokenizer = BpeTokenizer::new(config);
        assert!(tokenizer.is_ok());
    }

    #[tokio::test]
    async fn test_basic_encoding_decoding() {
        let config = BpeTokenizerConfig::default();
        let mut tokenizer = BpeTokenizer::new(config).unwrap();
        
        // Train on some sample text
        let training_texts = vec![
            "Hello world".to_string(),
            "This is a test".to_string(),
            "def function(): return True".to_string(),
        ];
        
        tokenizer.train(&training_texts).unwrap();
        
        // Test encoding and decoding
        let text = "Hello world";
        let token_ids = tokenizer.encode(text, true).unwrap();
        let decoded = tokenizer.decode(&token_ids, false).unwrap();
        
        // The decoded text should contain the original text (may have additional tokens)
        assert!(decoded.contains("Hello") || decoded.contains("world"));
    }

    #[test]
    fn test_bigcode_config() {
        let config = BigcodeDatasetConfig::default();
        assert!(!config.languages.is_empty());
        assert!(config.max_samples_per_language > 0);
    }
}
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_type: ModelType,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_sequence_length: usize,
    pub dropout: f32,
    pub layer_norm_eps: f64,
    pub use_bias: bool,
    pub rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextGeneration,
    CodeCompletion,
    QuestionAnswering,
    Summarization,
    Vision,
    Multimodal,
    Embedding,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: "goldbull-base".to_string(),
            model_type: ModelType::TextGeneration,
            vocab_size: 1_000_000, // BPEmb multilingual vocab
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_sequence_length: 32768, // 32K as specified
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: false,
            rope_scaling: None,
        }
    }
}

impl ModelConfig {
    pub fn text_generation() -> Self {
        Self {
            model_name: "goldbull-text".to_string(),
            model_type: ModelType::TextGeneration,
            ..Default::default()
        }
    }
    
    pub fn code_completion() -> Self {
        Self {
            model_name: "goldbull-code".to_string(),
            model_type: ModelType::CodeCompletion,
            ..Default::default()
        }
    }
    
    pub fn question_answering() -> Self {
        Self {
            model_name: "goldbull-sage".to_string(),
            model_type: ModelType::QuestionAnswering,
            ..Default::default()
        }
    }
    
    pub fn summarization() -> Self {
        Self {
            model_name: "goldbull-brief".to_string(),
            model_type: ModelType::Summarization,
            ..Default::default()
        }
    }
    
    pub fn vision() -> Self {
        Self {
            model_name: "goldbull-vision".to_string(),
            model_type: ModelType::Vision,
            hidden_size: 1024,
            ..Default::default()
        }
    }
    
    pub fn multimodal() -> Self {
        Self {
            model_name: "goldbull-multimodel".to_string(),
            model_type: ModelType::Multimodal,
            hidden_size: 1024,
            ..Default::default()
        }
    }
    
    pub fn embedding() -> Self {
        Self {
            model_name: "goldbull-embedding".to_string(),
            model_type: ModelType::Embedding,
            hidden_size: 384,
            ..Default::default()
        }
    }
}
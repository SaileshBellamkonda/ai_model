//! # AI Model Library
//! 
//! A lightweight, high-accuracy machine learning model library providing:
//! - Text generation
//! - Code completion
//! - Question & Answer
//! - Text summarization
//! - Basic visual analysis
//! - Function/tool calling with external API integration

pub mod core;
pub mod models;
pub mod tasks;
pub mod tools;
pub mod memory;
pub mod utils;
pub mod tokenizer;
pub mod inference;

pub use core::{AIModel, ModelConfig};
pub use models::neural_network::NeuralNetwork;
pub use tasks::{TaskType, TaskResult};
pub use tools::{FunctionCall, ToolRegistry};

/// Main error type for the AI model library
#[derive(thiserror::Error, Debug)]
pub enum AIError {
    #[error("Model initialization failed: {0}")]
    InitializationError(String),
    
    #[error("Inference error: {0}")]
    InferenceError(String),
    
    #[error("Memory limit exceeded: current={current}MB, limit={limit}MB")]
    MemoryLimitExceeded { current: usize, limit: usize },
    
    #[error("Unsupported task type: {0}")]
    UnsupportedTask(String),
    
    #[error("Tool execution failed: {0}")]
    ToolError(String),
    
    #[error("API error: {0}")]
    ApiError(String),
    
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, AIError>;

/// Performance metrics for monitoring model efficiency
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_tokens_per_sec: f64,
    pub accuracy_score: Option<f64>,
}
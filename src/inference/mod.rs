use crate::Result;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

pub mod llama_cpp;
pub mod onnx;

/// Trait for different inference engines
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Initialize the inference engine with a model
    async fn load_model(&mut self, model_path: &str) -> Result<()>;
    
    /// Generate text given a prompt
    async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String>;
    
    /// Complete code given partial code
    async fn complete_code(&self, partial_code: &str, language: &str) -> Result<String>;
    
    /// Get model information
    fn get_model_info(&self) -> Result<ModelInfo>;
    
    /// Check if model is loaded
    fn is_loaded(&self) -> bool;
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub architecture: String,
    pub parameters: u64,
    pub context_length: usize,
    pub vocab_size: usize,
}

/// Configuration for inference engines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub engine_type: InferenceEngineType,
    pub model_path: String,
    pub device: Device,
    pub max_memory_mb: usize,
    pub num_threads: usize,
    pub use_gpu: bool,
    pub gpu_layers: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InferenceEngineType {
    LlamaCpp,
    Onnx,
    Native,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda,
    Metal,
    OpenCL,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            engine_type: InferenceEngineType::Native,
            model_path: String::new(),
            device: Device::Cpu,
            max_memory_mb: 2048,
            num_threads: num_cpus::get(),
            use_gpu: false,
            gpu_layers: None,
        }
    }
}

/// Factory for creating inference engines
pub struct InferenceEngineFactory;

impl InferenceEngineFactory {
    pub fn create_engine(config: &InferenceConfig) -> Result<Box<dyn InferenceEngine>> {
        match config.engine_type {
            InferenceEngineType::LlamaCpp => {
                Ok(Box::new(llama_cpp::LlamaCppEngine::new(config.clone())?))
            },
            InferenceEngineType::Onnx => {
                Ok(Box::new(onnx::OnnxEngine::new(config.clone())?))
            },
            InferenceEngineType::Native => {
                Err(crate::AIError::InferenceError(
                    "Native engine not implemented in this module".to_string()
                ))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert!(matches!(config.engine_type, InferenceEngineType::Native));
        assert!(matches!(config.device, Device::Cpu));
        assert!(config.max_memory_mb > 0);
    }

    #[test]
    fn test_model_info_creation() {
        let info = ModelInfo {
            name: "test-model".to_string(),
            architecture: "transformer".to_string(),
            parameters: 7_000_000_000,
            context_length: 2048,
            vocab_size: 32000,
        };
        
        assert_eq!(info.name, "test-model");
        assert_eq!(info.parameters, 7_000_000_000);
    }
}
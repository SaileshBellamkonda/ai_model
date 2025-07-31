use serde::{Deserialize, Serialize};
use candle_core::DType;

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
    #[serde(skip, default = "default_dtype")]
    pub dtype: DType,
    // CPU optimization settings
    pub cpu_optimization: CpuOptimization,
    pub memory_optimization: MemoryOptimization,
    pub quantization: QuantizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimization {
    pub num_threads: Option<usize>, // None = auto-detect
    pub use_simd: bool,
    pub cache_friendly_layout: bool,
    pub progressive_loading: bool,
    pub memory_mapping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    pub max_memory_mb: usize,
    pub gradient_checkpointing: bool,
    pub tensor_pooling: bool,
    pub dynamic_batching: bool,
    pub memory_cleanup_threshold: f32, // Cleanup when usage > threshold
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enable_quantization: bool,
    pub weight_dtype: WeightDType,
    pub activation_dtype: ActivationDType,
    pub quantization_aware_training: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightDType {
    F32,
    F16,
    BF16,
    I8,
    I4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationDType {
    F32,
    F16,
    BF16,
}

impl Default for CpuOptimization {
    fn default() -> Self {
        Self {
            num_threads: None, // Auto-detect
            use_simd: true,
            cache_friendly_layout: true,
            progressive_loading: true,
            memory_mapping: true,
        }
    }
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024, // 1GB default limit
            gradient_checkpointing: true,
            tensor_pooling: true,
            dynamic_batching: true,
            memory_cleanup_threshold: 0.8, // Cleanup at 80% usage
        }
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            enable_quantization: true,
            weight_dtype: WeightDType::F16,
            activation_dtype: ActivationDType::F16,
            quantization_aware_training: false,
        }
    }
}

fn default_dtype() -> DType {
    DType::F32
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
            vocab_size: 32_000, // Reduced for memory efficiency
            hidden_size: 256,   // Reduced for CPU optimization
            num_layers: 6,      // Reduced for faster inference
            num_attention_heads: 8, // Optimized for CPU cache
            intermediate_size: 1024, // Reduced from 3072
            max_sequence_length: 2048, // Reduced for memory efficiency
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            use_bias: false,
            rope_scaling: None,
            dtype: DType::F16, // Use FP16 for memory efficiency
            cpu_optimization: CpuOptimization::default(),
            memory_optimization: MemoryOptimization::default(),
            quantization: QuantizationConfig::default(),
        }
    }
}

impl ModelConfig {
    pub fn text_generation() -> Self {
        Self {
            model_name: "goldbull-text".to_string(),
            model_type: ModelType::TextGeneration,
            vocab_size: 32_000,
            hidden_size: 384,
            num_layers: 8,
            num_attention_heads: 8,
            intermediate_size: 1536,
            max_sequence_length: 2048,
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 512, // 512MB for text model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn code_completion() -> Self {
        Self {
            model_name: "goldbull-code".to_string(),
            model_type: ModelType::CodeCompletion,
            vocab_size: 49_152, // Optimized for code tokens
            hidden_size: 512,
            num_layers: 8,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_sequence_length: 4096, // Longer context for code
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 768, // 768MB for code model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn question_answering() -> Self {
        Self {
            model_name: "goldbull-sage".to_string(),
            model_type: ModelType::QuestionAnswering,
            vocab_size: 32_000,
            hidden_size: 256,
            num_layers: 6,
            num_attention_heads: 8,
            intermediate_size: 1024,
            max_sequence_length: 1024, // Shorter for Q&A
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 256, // 256MB for Q&A model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn summarization() -> Self {
        Self {
            model_name: "goldbull-brief".to_string(),
            model_type: ModelType::Summarization,
            vocab_size: 32_000,
            hidden_size: 384,
            num_layers: 6,
            num_attention_heads: 8,
            intermediate_size: 1536,
            max_sequence_length: 2048,
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 512, // 512MB for summarization model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn vision() -> Self {
        Self {
            model_name: "goldbull-vision".to_string(),
            model_type: ModelType::Vision,
            vocab_size: 1000, // ImageNet classes
            hidden_size: 512,
            num_layers: 6,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_sequence_length: 196, // 14x14 patches for 224x224 image
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 512, // 512MB for vision model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn multimodal() -> Self {
        Self {
            model_name: "goldbull-multimodel".to_string(),
            model_type: ModelType::Multimodal,
            vocab_size: 32_000,
            hidden_size: 512,
            num_layers: 8,
            num_attention_heads: 8,
            intermediate_size: 2048,
            max_sequence_length: 2048,
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 1024, // 1GB for multimodal model
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    pub fn embedding() -> Self {
        Self {
            model_name: "goldbull-embedding".to_string(),
            model_type: ModelType::Embedding,
            vocab_size: 32_000,
            hidden_size: 256, // Smaller embeddings for efficiency
            num_layers: 4,
            num_attention_heads: 4,
            intermediate_size: 1024,
            max_sequence_length: 512,
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 128, // 128MB for embedding model
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a lightweight configuration for systems with very low memory (< 1GB)
    pub fn lightweight() -> Self {
        Self {
            model_name: "goldbull-micro".to_string(),
            model_type: ModelType::TextGeneration,
            vocab_size: 16_000, // Very small vocab
            hidden_size: 128,   // Minimal hidden size
            num_layers: 4,      // Few layers
            num_attention_heads: 4, // Minimal heads
            intermediate_size: 512, // Small FFN
            max_sequence_length: 512, // Short sequences
            dtype: DType::F16,
            memory_optimization: MemoryOptimization {
                max_memory_mb: 64, // 64MB for micro model
                gradient_checkpointing: true,
                tensor_pooling: true,
                dynamic_batching: true,
                memory_cleanup_threshold: 0.6, // Aggressive cleanup
            },
            quantization: QuantizationConfig {
                enable_quantization: true,
                weight_dtype: WeightDType::I8, // 8-bit weights
                activation_dtype: ActivationDType::F16,
                quantization_aware_training: false,
            },
            ..Default::default()
        }
    }
    
    /// Get estimated memory usage in MB
    pub fn estimated_memory_mb(&self) -> f64 {
        let params = self.estimate_parameters();
        let bytes_per_param = match self.quantization.weight_dtype {
            WeightDType::F32 => 4.0,
            WeightDType::F16 | WeightDType::BF16 => 2.0,
            WeightDType::I8 => 1.0,
            WeightDType::I4 => 0.5,
        };
        (params as f64 * bytes_per_param) / (1024.0 * 1024.0)
    }
    
    /// Estimate total parameters
    pub fn estimate_parameters(&self) -> usize {
        // Embeddings: vocab_size * hidden_size
        let embedding_params = self.vocab_size * self.hidden_size;
        
        // Each transformer layer
        let layer_params = self.num_layers * (
            // Self-attention: 4 * hidden_size^2 (Q, K, V, O projections)
            4 * self.hidden_size * self.hidden_size +
            // Feed-forward: 2 * hidden_size * intermediate_size  
            2 * self.hidden_size * self.intermediate_size +
            // Layer norms: 2 * hidden_size (input_ln + post_attn_ln)
            2 * self.hidden_size
        );
        
        // Output projection: hidden_size * vocab_size
        let output_params = self.hidden_size * self.vocab_size;
        
        embedding_params + layer_params + output_params
    }
}
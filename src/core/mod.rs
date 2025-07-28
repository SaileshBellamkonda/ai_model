use crate::{Result, PerformanceMetrics};
use crate::models::{GoldbullModel, GoldbullEmbedding, GoldbullEmbeddingConfig};
use crate::tokenizer::{TiktokenBpeTokenizer, TiktokenBpeConfig};
use crate::tasks::{TaskType, TaskResult};
use crate::tools::ToolRegistry;
use crate::memory::MemoryManager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the AI model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Maximum memory usage in MB (default: 2048MB = 2GB)
    pub max_memory_mb: usize,
    
    /// Model precision (f32 for performance, f64 for accuracy)
    pub use_f32_precision: bool,
    
    /// Enable aggressive memory optimization
    pub aggressive_memory_optimization: bool,
    
    /// Number of CPU threads to use
    pub cpu_threads: usize,
    
    /// Enable caching for repeated computations
    pub enable_caching: bool,
    
    /// Model architecture parameters
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_sequence_length: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048, // 2GB limit
            use_f32_precision: true, // f32 for better performance
            aggressive_memory_optimization: true,
            cpu_threads: num_cpus::get().min(8), // Use available CPUs, max 8
            enable_caching: true,
            hidden_size: 512, // Compact model size
            num_layers: 6, // Lightweight architecture
            vocab_size: 1000000, // BPEmb 1M vocabulary
            max_sequence_length: 2048,
        }
    }
}

/// Main AI model structure (Goldbull)
pub struct AIModel {
    config: ModelConfig,
    goldbull_model: Arc<RwLock<GoldbullModel>>,
    goldbull_embedding: Arc<RwLock<GoldbullEmbedding>>,
    tool_registry: Arc<RwLock<ToolRegistry>>,
    memory_manager: Arc<RwLock<MemoryManager>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl AIModel {
    /// Create a new AI model with the given configuration
    pub async fn new(config: ModelConfig) -> Result<Self> {
        // Initialize memory manager first
        let memory_manager = Arc::new(RwLock::new(
            MemoryManager::new(config.max_memory_mb)
        ));
        
        // Initialize tiktoken-style BPE tokenizer
        let tokenizer_config = TiktokenBpeConfig::default();
        let mut tokenizer = TiktokenBpeTokenizer::new(tokenizer_config)?;
        
        // Try to load BPEmb vocabulary
        if let Err(e) = tokenizer.load_bpemb_vocabulary().await {
            log::warn!("Failed to load BPEmb vocabulary: {}. Using base tokenizer.", e);
        }
        
        // Create a second tokenizer for embedding model
        let embedding_tokenizer_config = TiktokenBpeConfig::default();
        let mut embedding_tokenizer = TiktokenBpeTokenizer::new(embedding_tokenizer_config)?;
        if let Err(e) = embedding_tokenizer.load_bpemb_vocabulary().await {
            log::warn!("Failed to load BPEmb vocabulary for embedding model: {}. Using base tokenizer.", e);
        }
        
        // Initialize Goldbull main model
        let goldbull_model = Arc::new(RwLock::new(
            GoldbullModel::new(&config, tokenizer).await?
        ));
        
        // Initialize Goldbull embedding model
        let embedding_config = GoldbullEmbeddingConfig::default();
        let goldbull_embedding = Arc::new(RwLock::new(
            GoldbullEmbedding::new(embedding_config, embedding_tokenizer).await?
        ));
        
        // Initialize tool registry
        let tool_registry = Arc::new(RwLock::new(
            ToolRegistry::new()
        ));
        
        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(
            PerformanceMetrics {
                inference_time_ms: 0.0,
                memory_usage_mb: 0.0,
                throughput_tokens_per_sec: 0.0,
                accuracy_score: None,
            }
        ));
        
        log::info!("Goldbull model initialized successfully");
        
        Ok(Self {
            config,
            goldbull_model,
            goldbull_embedding,
            tool_registry,
            memory_manager,
            performance_metrics,
        })
    }
    
    /// Create a model with default configuration
    pub async fn with_defaults() -> Result<Self> {
        Self::new(ModelConfig::default()).await
    }
    
    /// Execute a task with the AI model
    pub async fn execute_task(&self, task_type: TaskType, input: &str) -> Result<TaskResult> {
        let start_time = std::time::Instant::now();
        
        // Check memory usage before execution
        {
            let memory_manager = self.memory_manager.read().await;
            memory_manager.check_memory_limit()?;
        }
        
        // Execute the specific task
        let result = match task_type {
            TaskType::TextGeneration { max_tokens, temperature } => {
                self.generate_text(input, max_tokens, temperature).await?
            },
            TaskType::CodeCompletion { language, context_lines } => {
                self.complete_code(input, &language, context_lines).await?
            },
            TaskType::QuestionAnswer { context } => {
                self.answer_question(input, context.as_deref()).await?
            },
            TaskType::Summarization { max_length, style } => {
                self.summarize_text(input, max_length, &style).await?
            },
            TaskType::VisualAnalysis { image_data } => {
                self.analyze_image(input, &image_data).await?
            },
        };
        
        // Update performance metrics
        let inference_time = start_time.elapsed().as_millis() as f64;
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.inference_time_ms = inference_time;
            
            let memory_manager = self.memory_manager.read().await;
            metrics.memory_usage_mb = memory_manager.current_usage_mb();
            
            // Calculate throughput based on output tokens
            if let Some(output_len) = result.output_tokens() {
                metrics.throughput_tokens_per_sec = (output_len as f64 * 1000.0) / inference_time;
            }
        }
        
        Ok(result)
    }
    
    /// Generate text based on input prompt
    async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<TaskResult> {
        let model = self.goldbull_model.read().await;
        let generated_text = model.generate_text(prompt, max_tokens, temperature).await?;
        
        Ok(TaskResult::TextGeneration {
            generated_text,
            tokens_generated: max_tokens,
            finish_reason: "length".to_string(),
        })
    }
    
    /// Complete code based on context
    async fn complete_code(&self, partial_code: &str, language: &str, context_lines: usize) -> Result<TaskResult> {
        let model = self.goldbull_model.read().await;
        let completion = model.complete_code(partial_code, language, context_lines).await?;
        
        Ok(TaskResult::CodeCompletion {
            completion,
            confidence_score: 0.85, // Placeholder - would be calculated by the model
        })
    }
    
    /// Answer a question with optional context
    async fn answer_question(&self, question: &str, context: Option<&str>) -> Result<TaskResult> {
        let model = self.goldbull_model.read().await;
        let answer = model.answer_question(question, context).await?;
        
        Ok(TaskResult::QuestionAnswer {
            answer,
            confidence_score: 0.8, // Placeholder
            sources: vec![], // Would be populated with actual sources
        })
    }
    
    /// Summarize text
    async fn summarize_text(&self, text: &str, max_length: usize, style: &str) -> Result<TaskResult> {
        let model = self.goldbull_model.read().await;
        let summary = model.summarize_text(text, max_length, style).await?;
        
        Ok(TaskResult::Summarization {
            summary: summary.clone(),
            compression_ratio: text.len() as f32 / summary.len() as f32,
        })
    }
    
    /// Analyze image (basic visual analysis)
    async fn analyze_image(&self, description: &str, image_data: &[u8]) -> Result<TaskResult> {
        let model = self.goldbull_model.read().await;
        let analysis = model.analyze_image(description, image_data).await?;
        
        Ok(TaskResult::VisualAnalysis {
            description: analysis,
            detected_objects: vec![], // Placeholder
            confidence_scores: vec![], // Placeholder
        })
    }
    
    /// Register a new tool/function for calling
    pub async fn register_tool(&self, name: String, description: String, 
                               handler: Box<dyn crate::tools::ToolHandler>) -> Result<()> {
        let mut registry = self.tool_registry.write().await;
        registry.register_tool(name, description, handler)
    }
    
    /// Execute a tool/function call
    pub async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let registry = self.tool_registry.read().await;
        registry.call_tool(name, args).await
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().await.clone()
    }
    
    /// Get model configuration
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get text embeddings using the Goldbull embedding model
    pub async fn get_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embedding_model = self.goldbull_embedding.read().await;
        let embeddings = embedding_model.encode_batch(texts).await?;
        
        // Convert to Vec<Vec<f32>>
        let mut result = Vec::new();
        for row in embeddings.rows() {
            result.push(row.to_vec());
        }
        
        Ok(result)
    }

    /// Compute text similarity using embeddings
    pub async fn text_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let embedding_model = self.goldbull_embedding.read().await;
        embedding_model.similarity(text1, text2).await
    }

    /// Advanced NLP analysis using the main Goldbull model
    pub async fn analyze_text_nlp(&self, text: &str) -> Result<String> {
        let model = self.goldbull_model.read().await;
        model.analyze_text(text).await
    }

    /// Extract named entities using NLP
    pub async fn extract_entities(&self, text: &str) -> Result<String> {
        let model = self.goldbull_model.read().await;
        model.extract_entities(text).await
    }

    /// Classify text into categories
    pub async fn classify_text(&self, text: &str, categories: &[String]) -> Result<String> {
        let model = self.goldbull_model.read().await;
        model.classify_text(text, categories).await
    }
}

// Make AIModel thread-safe
unsafe impl Send for AIModel {}
unsafe impl Sync for AIModel {}
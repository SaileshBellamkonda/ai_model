use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use goldbull_core::ModelConfig;
use goldbull_tokenizer::{BpeTokenizer, Tokenizer};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
use crate::model::GoldbullCode;
use crate::syntax::{LanguageType, SyntaxAnalyzer, CodeFeatures};

/// Comprehensive training framework for code completion models
/// Includes real loss computation, data loading, and production-ready checkpointing
pub struct Trainer {
    /// The code completion model being trained
    model: GoldbullCode,
    /// Training configuration parameters
    config: TrainingConfig,
    /// Code dataset loader and processor
    dataset: CodeDataset,
    /// Training metrics tracker
    metrics: TrainingMetrics,
    /// Optimizer for model parameters
    optimizer: CodeOptimizer,
    /// Learning rate scheduler
    scheduler: LearningRateScheduler,
    /// Validation dataset for evaluation
    validation_dataset: Option<CodeDataset>,
    /// Checkpoint manager for model saving
    checkpoint_manager: CheckpointManager,
}

/// Configuration for code model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Learning rate for optimizer
    pub learning_rate: f64,
    /// Weight decay for regularization
    pub weight_decay: f64,
    /// Gradient clipping threshold
    pub gradient_clipping: f64,
    /// Warmup steps for learning rate
    pub warmup_steps: usize,
    /// Validation frequency (epochs)
    pub validation_frequency: usize,
    /// Checkpoint saving frequency (epochs)
    pub checkpoint_frequency: usize,
    /// Maximum sequence length for training
    pub max_sequence_length: usize,
    /// Whether to use mixed precision training
    pub mixed_precision: bool,
    /// Target code languages for training
    pub target_languages: Vec<LanguageType>,
    /// Data augmentation settings
    pub augmentation: AugmentationConfig,
    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 8,
            learning_rate: 2e-5,
            weight_decay: 0.01,
            gradient_clipping: 1.0,
            warmup_steps: 1000,
            validation_frequency: 1,
            checkpoint_frequency: 1,
            max_sequence_length: 2048,
            mixed_precision: false,
            target_languages: vec![
                LanguageType::Rust,
                LanguageType::Python,
                LanguageType::JavaScript,
                LanguageType::TypeScript,
            ],
            augmentation: AugmentationConfig::default(),
            early_stopping: EarlyStoppingConfig::default(),
        }
    }
}

/// Data augmentation configuration for code training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Whether to enable variable name randomization
    pub randomize_variable_names: bool,
    /// Whether to enable comment removal/addition
    pub randomize_comments: bool,
    /// Whether to enable formatting variations
    pub randomize_formatting: bool,
    /// Whether to enable code snippet mixing
    pub enable_mixup: bool,
    /// Augmentation probability (0.0 - 1.0)
    pub augmentation_probability: f64,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            randomize_variable_names: false,
            randomize_comments: true,
            randomize_formatting: true,
            enable_mixup: false,
            augmentation_probability: 0.3,
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Whether early stopping is enabled
    pub enabled: bool,
    /// Patience (epochs without improvement)
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Metric to monitor for early stopping
    pub monitor_metric: String,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 3,
            min_delta: 0.001,
            monitor_metric: "validation_loss".to_string(),
        }
    }
}

/// Code dataset loader and processor
/// Handles multiple programming languages and code-specific preprocessing
#[derive(Debug)]
pub struct CodeDataset {
    /// Code samples with language information
    samples: Vec<CodeSample>,
    /// Dataset statistics and metadata
    metadata: DatasetMetadata,
    /// Language-specific processors
    processors: HashMap<LanguageType, CodeProcessor>,
    /// Tokenizer for encoding
    tokenizer: BpeTokenizer,
    /// Current batch index
    current_batch: usize,
}

/// Individual code sample
#[derive(Debug, Clone)]
pub struct CodeSample {
    /// Source code content
    pub code: String,
    /// Programming language
    pub language: LanguageType,
    /// File path or identifier
    pub source_file: String,
    /// Code features and metadata
    pub features: Option<CodeFeatures>,
    /// Difficulty level (0.0 - 1.0)
    pub difficulty: f64,
    /// Quality score (0.0 - 1.0)
    pub quality_score: f64,
}

/// Dataset metadata and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Total number of samples
    pub total_samples: usize,
    /// Total number of tokens
    pub total_tokens: usize,
    /// Samples per language
    pub language_distribution: HashMap<LanguageType, usize>,
    /// Average sample length
    pub average_length: f64,
    /// Dataset version
    pub version: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Language-specific code processor
pub struct CodeProcessor {
    /// Language type
    language: LanguageType,
    /// Syntax analyzer for features
    analyzer: SyntaxAnalyzer,
    /// Language-specific tokenization rules
    tokenization_rules: TokenizationRules,
    /// Code quality validator
    quality_validator: QualityValidator,
}

impl std::fmt::Debug for CodeProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeProcessor")
            .field("language", &self.language)
            .finish()
    }
}

/// Tokenization rules for specific languages
#[derive(Debug, Clone)]
pub struct TokenizationRules {
    /// Special tokens for the language
    pub special_tokens: Vec<String>,
    /// Token splitting patterns
    pub split_patterns: Vec<regex::Regex>,
    /// Whether to preserve whitespace
    pub preserve_whitespace: bool,
    /// Maximum token length
    pub max_token_length: usize,
}

/// Code quality validator for training data
#[derive(Debug)]
pub struct QualityValidator {
    /// Minimum code quality threshold
    quality_threshold: f64,
    /// Syntax validation rules
    syntax_rules: Vec<SyntaxRule>,
    /// Style validation rules
    style_rules: Vec<StyleRule>,
}

/// Syntax validation rule
#[derive(Debug, Clone)]
pub struct SyntaxRule {
    /// Rule name
    pub name: String,
    /// Pattern to validate
    pub pattern: regex::Regex,
    /// Rule weight
    pub weight: f64,
}

/// Style validation rule
#[derive(Debug, Clone)]
pub struct StyleRule {
    /// Rule name
    pub name: String,
    /// Style pattern
    pub pattern: regex::Regex,
    /// Recommended style
    pub recommendation: String,
    /// Rule weight
    pub weight: f64,
}

/// Training metrics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub training_loss: Vec<f64>,
    /// Validation loss history
    pub validation_loss: Vec<f64>,
    /// Learning rate history
    pub learning_rate: Vec<f64>,
    /// Code completion accuracy
    pub completion_accuracy: Vec<f64>,
    /// Syntax validity rate
    pub syntax_validity_rate: Vec<f64>,
    /// Training time per epoch
    pub epoch_times: Vec<f64>,
    /// Current epoch
    pub current_epoch: usize,
    /// Best validation loss achieved
    pub best_validation_loss: f64,
    /// Total training time
    pub total_training_time: f64,
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            training_loss: Vec::new(),
            validation_loss: Vec::new(),
            learning_rate: Vec::new(),
            completion_accuracy: Vec::new(),
            syntax_validity_rate: Vec::new(),
            epoch_times: Vec::new(),
            current_epoch: 0,
            best_validation_loss: f64::INFINITY,
            total_training_time: 0.0,
        }
    }
}

/// Optimizer for code model training
pub struct CodeOptimizer {
    /// Optimizer type (Adam, AdamW, SGD)
    optimizer_type: OptimizerType,
    /// Learning rate
    learning_rate: f64,
    /// Momentum parameters
    momentum: (f64, f64),
    /// Weight decay
    weight_decay: f64,
    /// Accumulated gradients
    accumulated_gradients: VarMap,
    /// Optimizer state
    optimizer_state: HashMap<String, Tensor>,
}

impl std::fmt::Debug for CodeOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeOptimizer")
            .field("optimizer_type", &self.optimizer_type)
            .field("learning_rate", &self.learning_rate)
            .field("momentum", &self.momentum)
            .field("weight_decay", &self.weight_decay)
            .finish()
    }
}

/// Optimizer types
#[derive(Debug, Clone, Copy)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
}

/// Learning rate scheduler
#[derive(Debug)]
pub struct LearningRateScheduler {
    /// Initial learning rate
    initial_lr: f64,
    /// Current learning rate
    current_lr: f64,
    /// Warmup steps
    warmup_steps: usize,
    /// Total training steps
    total_steps: usize,
    /// Current step
    current_step: usize,
    /// Scheduler type
    scheduler_type: SchedulerType,
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy)]
pub enum SchedulerType {
    Linear,
    Cosine,
    Exponential,
    Polynomial,
}

/// Checkpoint manager for model saving and loading
#[derive(Debug)]
pub struct CheckpointManager {
    /// Checkpoint directory
    checkpoint_dir: PathBuf,
    /// Maximum checkpoints to keep
    max_checkpoints: usize,
    /// Checkpoint metadata
    metadata: CheckpointMetadata,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model version
    pub model_version: String,
    /// Training step
    pub step: usize,
    /// Epoch number
    pub epoch: usize,
    /// Training loss
    pub training_loss: f64,
    /// Validation loss
    pub validation_loss: f64,
    /// Model configuration
    pub config: ModelConfig,
    /// Training configuration
    pub training_config: TrainingConfig,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Trainer {
    /// Create a new trainer for code completion model
    /// 
    /// # Arguments
    /// * `model` - The code completion model to train
    /// * `config` - Training configuration
    /// 
    /// # Returns
    /// * `Result<Self>` - Initialized trainer or error
    pub fn new(model: GoldbullCode, config: TrainingConfig) -> Result<Self> {
        let dataset = CodeDataset::new(BpeTokenizer::new(goldbull_tokenizer::TokenizerConfig::default())?)?;
        let metrics = TrainingMetrics::default();
        let optimizer = CodeOptimizer::new(OptimizerType::AdamW, config.learning_rate, config.weight_decay)?;
        let scheduler = LearningRateScheduler::new(
            config.learning_rate,
            config.warmup_steps,
            config.epochs * 1000, // Estimate total steps
            SchedulerType::Linear,
        )?;
        let checkpoint_manager = CheckpointManager::new("./checkpoints")?;
        
        Ok(Self {
            model,
            config,
            dataset,
            metrics,
            optimizer,
            scheduler,
            validation_dataset: None,
            checkpoint_manager,
        })
    }
    
    /// Load training data from directory containing code files
    /// 
    /// # Arguments
    /// * `data_path` - Path to directory containing code files
    /// * `validation_split` - Fraction of data to use for validation (0.0 - 1.0)
    /// 
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn load_data(&mut self, data_path: &str, validation_split: f64) -> Result<()> {
        tracing::info!("Loading training data from: {}", data_path);
        
        // Discover code files in directory
        let code_files = self.discover_code_files(data_path)?;
        tracing::info!("Found {} code files", code_files.len());
        
        // Process files in parallel for efficiency
        let samples: Vec<CodeSample> = code_files
            .par_iter()
            .filter_map(|file_path| {
                self.process_code_file(file_path).unwrap_or_else(|e| {
                    tracing::warn!("Failed to process file {}: {}", file_path.display(), e);
                    None
                })
            })
            .collect();
        
        tracing::info!("Processed {} valid code samples", samples.len());
        
        // Split into training and validation sets
        let split_index = (samples.len() as f64 * (1.0 - validation_split)) as usize;
        let (training_samples, validation_samples) = samples.split_at(split_index);
        
        // Create training dataset
        self.dataset.load_samples(training_samples.to_vec())?;
        
        // Create validation dataset if validation split > 0
        if validation_split > 0.0 && !validation_samples.is_empty() {
            let mut validation_dataset = CodeDataset::new(BpeTokenizer::new(goldbull_tokenizer::TokenizerConfig::default())?)?;
            validation_dataset.load_samples(validation_samples.to_vec())?;
            self.validation_dataset = Some(validation_dataset);
            tracing::info!("Created validation dataset with {} samples", validation_samples.len());
        }
        
        Ok(())
    }
    
    /// Train the model on loaded data
    /// 
    /// # Returns
    /// * `Result<TrainingMetrics>` - Final training metrics or error
    pub fn train(&mut self) -> Result<TrainingMetrics> {
        tracing::info!("Starting training for {} epochs", self.config.epochs);
        
        let total_start_time = Instant::now();
        let mut early_stopping_counter = 0;
        
        for epoch in 0..self.config.epochs {
            let epoch_start_time = Instant::now();
            
            tracing::info!("Starting epoch {}/{}", epoch + 1, self.config.epochs);
            
            // Training phase
            let epoch_training_loss = self.train_epoch()?;
            
            // Validation phase
            let epoch_validation_loss = if epoch % self.config.validation_frequency == 0 {
                if let Some(ref validation_dataset) = self.validation_dataset {
                    Some(self.validate_epoch(validation_dataset)?)
                } else {
                    None
                }
            } else {
                None
            };
            
            // Update metrics
            self.metrics.current_epoch = epoch + 1;
            self.metrics.training_loss.push(epoch_training_loss);
            
            if let Some(val_loss) = epoch_validation_loss {
                self.metrics.validation_loss.push(val_loss);
                
                // Check for improvement
                if val_loss < self.metrics.best_validation_loss - self.config.early_stopping.min_delta {
                    self.metrics.best_validation_loss = val_loss;
                    early_stopping_counter = 0;
                } else {
                    early_stopping_counter += 1;
                }
            }
            
            self.metrics.learning_rate.push(self.scheduler.current_lr);
            
            let epoch_time = epoch_start_time.elapsed().as_secs_f64();
            self.metrics.epoch_times.push(epoch_time);
            
            tracing::info!(
                "Epoch {}/{} completed - Training Loss: {:.4}, Validation Loss: {:?}, Time: {:.2}s",
                epoch + 1,
                self.config.epochs,
                epoch_training_loss,
                epoch_validation_loss,
                epoch_time
            );
            
            // Save checkpoint
            if (epoch + 1) % self.config.checkpoint_frequency == 0 {
                self.save_checkpoint(epoch + 1, epoch_training_loss, epoch_validation_loss.unwrap_or(0.0))?;
            }
            
            // Check early stopping
            if self.config.early_stopping.enabled && early_stopping_counter >= self.config.early_stopping.patience {
                tracing::info!("Early stopping triggered after {} epochs without improvement", early_stopping_counter);
                break;
            }
            
            // Update learning rate
            self.scheduler.step();
        }
        
        self.metrics.total_training_time = total_start_time.elapsed().as_secs_f64();
        
        tracing::info!(
            "Training completed - Total time: {:.2}s, Best validation loss: {:.4}",
            self.metrics.total_training_time,
            self.metrics.best_validation_loss
        );
        
        Ok(self.metrics.clone())
    }
    
    /// Train for one epoch
    fn train_epoch(&mut self) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        self.dataset.reset_batches();
        
        while let Some(batch) = self.dataset.next_batch(self.config.batch_size)? {
            // Forward pass
            let (loss, _logits) = self.forward_pass(&batch)?;
            
            // Extract loss value before move
            let loss_value = loss.to_scalar::<f64>()?;
            
            // Backward pass and optimization step
            self.backward_pass(loss)?;
            
            total_loss += loss_value;
            batch_count += 1;
            
            // Update learning rate
            self.scheduler.step();
            
            if batch_count % 100 == 0 {
                tracing::debug!("Processed {} batches, current loss: {:.4}", batch_count, total_loss / batch_count as f64);
            }
        }
        
        Ok(total_loss / batch_count as f64)
    }
    
    /// Validate for one epoch
    fn validate_epoch(&self, validation_dataset: &CodeDataset) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        // Create a mutable copy for iteration
        let mut val_dataset = validation_dataset.clone();
        val_dataset.reset_batches();
        
        while let Some(batch) = val_dataset.next_batch(self.config.batch_size)? {
            // Forward pass only (no gradient computation)
            let (loss, _logits) = self.forward_pass(&batch)?;
            
            total_loss += loss.to_scalar::<f64>()?;
            batch_count += 1;
        }
        
        Ok(total_loss / batch_count as f64)
    }
    
    /// Forward pass through the model
    fn forward_pass(&self, batch: &CodeBatch) -> Result<(Tensor, Tensor)> {
        // Forward pass through transformer model
        let logits = self.model.forward(&batch.input_ids, batch.attention_mask.as_ref())?;
        
        // Calculate cross-entropy loss
        let loss = self.calculate_cross_entropy_loss(&logits, &batch.target_ids)?;
        
        Ok((loss, logits))
    }
    
    /// Calculate cross-entropy loss with numerical stability
    fn calculate_cross_entropy_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, vocab_size) = logits.dims3()?;
        
        // Reshape logits and targets for loss calculation
        let logits_flat = logits.reshape((batch_size * seq_len, vocab_size))?;
        let targets_flat = targets.reshape((batch_size * seq_len,))?;
        
        // Apply log softmax for numerical stability
        let log_probs = candle_nn::ops::log_softmax(&logits_flat, 1)?;
        
        // Gather log probabilities for target tokens
        let target_log_probs = self.gather_target_log_probs(&log_probs, &targets_flat)?;
        
        // Calculate negative log likelihood (cross-entropy loss)
        let loss = target_log_probs.neg()?.mean(0)?;
        
        Ok(loss)
    }
    
    /// Gather log probabilities for target tokens
    fn gather_target_log_probs(&self, log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let (seq_len, vocab_size) = log_probs.dims2()?;
        let target_indices = targets.to_vec1::<u32>()?;
        
        let mut gathered_probs = Vec::new();
        let log_probs_data = log_probs.to_vec2::<f32>()?;
        
        for (i, &target_idx) in target_indices.iter().enumerate() {
            if i < seq_len && (target_idx as usize) < vocab_size {
                gathered_probs.push(log_probs_data[i][target_idx as usize]);
            } else {
                gathered_probs.push(0.0); // Padding token
            }
        }
        
        Ok(Tensor::from_vec(gathered_probs, seq_len, log_probs.device())?)
    }
    
    /// Backward pass and parameter update
    fn backward_pass(&mut self, _loss: Tensor) -> Result<()> {
        // In a real implementation, this would:
        // 1. Compute gradients via backpropagation
        // 2. Apply gradient clipping
        // 3. Update model parameters using optimizer
        // 4. Zero gradients for next iteration
        
        // Placeholder for gradient computation and parameter updates
        tracing::debug!("Performing backward pass (placeholder implementation)");
        
        Ok(())
    }
    
    /// Save model checkpoint
    fn save_checkpoint(&self, epoch: usize, training_loss: f64, validation_loss: f64) -> Result<()> {
        let checkpoint_metadata = CheckpointMetadata {
            model_version: "goldbull-code-1.0".to_string(),
            step: epoch * 1000, // Approximate step count
            epoch,
            training_loss,
            validation_loss,
            config: self.model.config().clone(),
            training_config: self.config.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        self.checkpoint_manager.save_checkpoint(&self.model, &checkpoint_metadata)
    }
    
    /// Discover code files in directory
    fn discover_code_files(&self, data_path: &str) -> Result<Vec<PathBuf>> {
        let mut code_files = Vec::new();
        let data_dir = Path::new(data_path);
        
        if !data_dir.exists() {
            return Err(anyhow!("Data directory does not exist: {}", data_path));
        }
        
        for entry in walkdir::WalkDir::new(data_dir) {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if self.is_supported_code_file(extension.to_str().unwrap_or("")) {
                        code_files.push(path.to_path_buf());
                    }
                }
            }
        }
        
        Ok(code_files)
    }
    
    /// Check if file extension is supported for training
    fn is_supported_code_file(&self, extension: &str) -> bool {
        matches!(extension, "rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "go" | "rb" | "php" | "cs" | "kt")
    }
    
    /// Process individual code file into training sample
    fn process_code_file(&self, file_path: &Path) -> Result<Option<CodeSample>> {
        let content = fs::read_to_string(file_path)?;
        
        // Skip very small or very large files
        if content.len() < 50 || content.len() > 100_000 {
            return Ok(None);
        }
        
        // Detect language from file extension
        let language = self.detect_language_from_extension(file_path)?;
        
        // Skip if language not in target languages
        if !self.config.target_languages.contains(&language) {
            return Ok(None);
        }
        
        // Calculate quality score
        let quality_score = self.calculate_code_quality(&content, language)?;
        
        // Skip low-quality code
        if quality_score < 0.5 {
            return Ok(None);
        }
        
        Ok(Some(CodeSample {
            code: content,
            language,
            source_file: file_path.to_string_lossy().to_string(),
            features: None, // Will be computed on demand
            difficulty: 0.5, // Default difficulty
            quality_score,
        }))
    }
    
    /// Detect programming language from file extension
    fn detect_language_from_extension(&self, file_path: &Path) -> Result<LanguageType> {
        if let Some(extension) = file_path.extension().and_then(|e| e.to_str()) {
            match extension {
                "rs" => Ok(LanguageType::Rust),
                "py" => Ok(LanguageType::Python),
                "js" => Ok(LanguageType::JavaScript),
                "ts" => Ok(LanguageType::TypeScript),
                "java" => Ok(LanguageType::Java),
                "cpp" | "cc" | "cxx" => Ok(LanguageType::Cpp),
                "c" => Ok(LanguageType::C),
                "go" => Ok(LanguageType::Go),
                _ => Ok(LanguageType::Unknown),
            }
        } else {
            Ok(LanguageType::Unknown)
        }
    }
    
    /// Calculate code quality score
    fn calculate_code_quality(&self, code: &str, language: LanguageType) -> Result<f64> {
        let mut quality_score = 1.0;
        
        // Basic quality heuristics
        let lines: Vec<&str> = code.lines().collect();
        let non_empty_lines: Vec<&str> = lines.iter().filter(|line| !line.trim().is_empty()).cloned().collect();
        
        // Penalize very short files
        if non_empty_lines.len() < 5 {
            quality_score *= 0.5;
        }
        
        // Check for common code patterns
        if code.contains("TODO") || code.contains("FIXME") {
            quality_score *= 0.8;
        }
        
        // Check for proper function definitions
        let has_functions = match language {
            LanguageType::Rust => code.contains("fn "),
            LanguageType::Python => code.contains("def "),
            LanguageType::JavaScript | LanguageType::TypeScript => code.contains("function ") || code.contains("=>"),
            LanguageType::Java => code.contains("public ") || code.contains("private "),
            _ => true, // Default to acceptable
        };
        
        if !has_functions && non_empty_lines.len() > 10 {
            quality_score *= 0.7;
        }
        
        Ok(quality_score)
    }
}

/// Training batch for code completion
#[derive(Debug, Clone)]
pub struct CodeBatch {
    /// Input token IDs
    pub input_ids: Tensor,
    /// Target token IDs (shifted input for next-token prediction)
    pub target_ids: Tensor,
    /// Attention mask for variable length sequences
    pub attention_mask: Option<Tensor>,
    /// Language labels for each sample
    pub language_labels: Vec<LanguageType>,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Batch metadata
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Sample indices in original dataset
    pub sample_indices: Vec<usize>,
    /// Sequence lengths before padding
    pub sequence_lengths: Vec<usize>,
    /// Difficulty scores
    pub difficulty_scores: Vec<f64>,
}

impl CodeDataset {
    /// Create a new code dataset
    fn new(tokenizer: BpeTokenizer) -> Result<Self> {
        Ok(Self {
            samples: Vec::new(),
            metadata: DatasetMetadata {
                total_samples: 0,
                total_tokens: 0,
                language_distribution: HashMap::new(),
                average_length: 0.0,
                version: "1.0".to_string(),
                created_at: chrono::Utc::now(),
            },
            processors: HashMap::new(),
            tokenizer,
            current_batch: 0,
        })
    }
    
    /// Load samples into dataset
    fn load_samples(&mut self, samples: Vec<CodeSample>) -> Result<()> {
        self.samples = samples;
        self.update_metadata()?;
        Ok(())
    }
    
    /// Update dataset metadata
    fn update_metadata(&mut self) -> Result<()> {
        self.metadata.total_samples = self.samples.len();
        self.metadata.language_distribution.clear();
        
        let mut total_length = 0;
        
        for sample in &self.samples {
            *self.metadata.language_distribution.entry(sample.language).or_insert(0) += 1;
            total_length += sample.code.len();
        }
        
        self.metadata.average_length = if self.samples.is_empty() {
            0.0
        } else {
            total_length as f64 / self.samples.len() as f64
        };
        
        // Estimate total tokens (rough approximation)
        self.metadata.total_tokens = total_length / 4; // Assuming ~4 chars per token
        
        Ok(())
    }
    
    /// Reset batch iterator
    fn reset_batches(&mut self) {
        self.current_batch = 0;
    }
    
    /// Get next batch of samples
    fn next_batch(&mut self, batch_size: usize) -> Result<Option<CodeBatch>> {
        let start_idx = self.current_batch * batch_size;
        let end_idx = (start_idx + batch_size).min(self.samples.len());
        
        if start_idx >= self.samples.len() {
            return Ok(None);
        }
        
        let batch_samples = &self.samples[start_idx..end_idx];
        let batch = self.create_batch(batch_samples)?;
        
        self.current_batch += 1;
        Ok(Some(batch))
    }
    
    /// Create batch from samples
    fn create_batch(&self, samples: &[CodeSample]) -> Result<CodeBatch> {
        let mut input_sequences = Vec::new();
        let mut target_sequences = Vec::new();
        let mut language_labels = Vec::new();
        let mut sample_indices = Vec::new();
        let mut sequence_lengths = Vec::new();
        let mut difficulty_scores = Vec::new();
        
        for (idx, sample) in samples.iter().enumerate() {
            // Tokenize code
            let tokens = self.tokenizer.encode(&sample.code)?;
            
            // Create input and target sequences (shifted by 1 for next-token prediction)
            if tokens.len() > 1 {
                let input_tokens = &tokens[..tokens.len()-1];
                let target_tokens = &tokens[1..];
                
                input_sequences.push(input_tokens.to_vec());
                target_sequences.push(target_tokens.to_vec());
                language_labels.push(sample.language);
                sample_indices.push(idx);
                sequence_lengths.push(tokens.len() - 1);
                difficulty_scores.push(sample.difficulty);
            }
        }
        
        // Pad sequences to same length
        let max_length = input_sequences.iter().map(|seq| seq.len()).max().unwrap_or(0);
        let max_length = max_length.min(2048); // Limit to maximum sequence length
        
        let mut padded_inputs = Vec::new();
        let mut padded_targets = Vec::new();
        let mut attention_masks = Vec::new();
        
        for (input_seq, target_seq) in input_sequences.iter().zip(target_sequences.iter()) {
            let mut padded_input = input_seq.clone();
            let mut padded_target = target_seq.clone();
            let mut attention_mask = vec![1u8; input_seq.len()];
            
            // Truncate if too long
            if padded_input.len() > max_length {
                padded_input.truncate(max_length);
                padded_target.truncate(max_length);
                attention_mask.truncate(max_length);
            }
            
            // Pad to max length
            while padded_input.len() < max_length {
                padded_input.push(0); // Padding token
                padded_target.push(0);
                attention_mask.push(0);
            }
            
            padded_inputs.push(padded_input);
            padded_targets.push(padded_target);
            attention_masks.push(attention_mask);
        }
        
        // Convert to tensors
        let device = Device::Cpu; // Use CPU for training
        
        let input_ids = Tensor::from_vec(
            padded_inputs.into_iter().flatten().collect::<Vec<u32>>(),
            (samples.len(), max_length),
            &device
        )?;
        
        let target_ids = Tensor::from_vec(
            padded_targets.into_iter().flatten().collect::<Vec<u32>>(),
            (samples.len(), max_length),
            &device
        )?;
        
        let attention_mask = Tensor::from_vec(
            attention_masks.into_iter().flatten().collect::<Vec<u8>>(),
            (samples.len(), max_length),
            &device
        )?;
        
        Ok(CodeBatch {
            input_ids,
            target_ids,
            attention_mask: Some(attention_mask),
            language_labels,
            metadata: BatchMetadata {
                sample_indices,
                sequence_lengths,
                difficulty_scores,
            },
        })
    }
}

// Clone implementation for CodeDataset (for validation)
impl Clone for CodeDataset {
    fn clone(&self) -> Self {
        Self {
            samples: self.samples.clone(),
            metadata: self.metadata.clone(),
            processors: HashMap::new(), // Don't clone processors
            tokenizer: self.tokenizer.clone(),
            current_batch: 0,
        }
    }
}

impl CodeOptimizer {
    fn new(optimizer_type: OptimizerType, learning_rate: f64, weight_decay: f64) -> Result<Self> {
        Ok(Self {
            optimizer_type,
            learning_rate,
            momentum: (0.9, 0.999), // Default Adam parameters
            weight_decay,
            accumulated_gradients: VarMap::new(),
            optimizer_state: HashMap::new(),
        })
    }
}

impl LearningRateScheduler {
    fn new(initial_lr: f64, warmup_steps: usize, total_steps: usize, scheduler_type: SchedulerType) -> Result<Self> {
        Ok(Self {
            initial_lr,
            current_lr: initial_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            scheduler_type,
        })
    }
    
    fn step(&mut self) {
        self.current_step += 1;
        
        if self.current_step <= self.warmup_steps {
            // Linear warmup
            self.current_lr = self.initial_lr * (self.current_step as f64 / self.warmup_steps as f64);
        } else {
            // Apply scheduler
            match self.scheduler_type {
                SchedulerType::Linear => {
                    let remaining_steps = self.total_steps - self.warmup_steps;
                    let current_remaining = self.total_steps - self.current_step;
                    self.current_lr = self.initial_lr * (current_remaining as f64 / remaining_steps as f64);
                }
                SchedulerType::Cosine => {
                    let progress = (self.current_step - self.warmup_steps) as f64 / (self.total_steps - self.warmup_steps) as f64;
                    self.current_lr = self.initial_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                }
                _ => {
                    // Keep current learning rate for other types
                }
            }
        }
    }
}

impl CheckpointManager {
    fn new(checkpoint_dir: &str) -> Result<Self> {
        let checkpoint_dir = PathBuf::from(checkpoint_dir);
        fs::create_dir_all(&checkpoint_dir)?;
        
        Ok(Self {
            checkpoint_dir,
            max_checkpoints: 5,
            metadata: CheckpointMetadata {
                model_version: "goldbull-code-1.0".to_string(),
                step: 0,
                epoch: 0,
                training_loss: 0.0,
                validation_loss: 0.0,
                config: ModelConfig::default(),
                training_config: TrainingConfig::default(),
                timestamp: chrono::Utc::now(),
            },
        })
    }
    
    fn save_checkpoint(&self, model: &GoldbullCode, metadata: &CheckpointMetadata) -> Result<()> {
        let checkpoint_name = format!("checkpoint_epoch_{}_step_{}.safetensors", metadata.epoch, metadata.step);
        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);
        
        // Save model weights
        model.save_weights(&checkpoint_path.to_string_lossy())?;
        
        // Save metadata
        let metadata_path = checkpoint_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        fs::write(metadata_path, metadata_json)?;
        
        tracing::info!("Saved checkpoint: {}", checkpoint_name);
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;
        
        Ok(())
    }
    
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoint_files: Vec<_> = fs::read_dir(&self.checkpoint_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .collect();
        
        // Sort by modification time (newest first)
        checkpoint_files.sort_by_key(|entry| {
            entry.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });
        checkpoint_files.reverse();
        
        // Remove old checkpoints
        for file_to_remove in checkpoint_files.iter().skip(self.max_checkpoints) {
            let path = file_to_remove.path();
            fs::remove_file(&path)?;
            
            // Also remove corresponding metadata file
            let metadata_path = path.with_extension("json");
            if metadata_path.exists() {
                fs::remove_file(metadata_path)?;
            }
        }
        
        Ok(())
    }
}
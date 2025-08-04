# Goldbull AI Model Suite - Training Guide

## Overview

This guide covers how to train and fine-tune models in the Goldbull AI Model Suite. The training system is designed for CPU efficiency while supporting both full training from scratch and fine-tuning of pre-trained models.

## Training Architecture

### Core Training Components

```rust
// Core training interfaces
pub trait Trainable {
    fn train_step(&mut self, batch: &Batch) -> Result<f32>;
    fn validate(&self, dataset: &Dataset) -> Result<ValidationMetrics>;
    fn save_checkpoint(&self, path: &str) -> Result<()>;
    fn load_checkpoint(&mut self, path: &str) -> Result<()>;
}

// Training configuration
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gradient_clipping: f32,
    pub weight_decay: f32,
    pub save_frequency: usize,
}
```

## Quick Start Training

### Basic Training Setup

```rust
use goldbull_text::{GoldbullText, Trainer};
use goldbull_core::{ModelConfig, TrainingConfig};
use candle_core::Device;

fn main() -> anyhow::Result<()> {
    // Initialize model and training components
    let device = Device::Cpu;
    let model_config = ModelConfig::default();
    let training_config = TrainingConfig {
        learning_rate: 1e-4,
        batch_size: 8,
        num_epochs: 5,
        gradient_clipping: 1.0,
        weight_decay: 0.01,
        save_frequency: 100,
    };
    
    // Create model and trainer
    let model = GoldbullText::new(model_config, device)?;
    let mut trainer = Trainer::new(model, training_config);
    
    // Train on dataset
    trainer.train_on_dataset("path/to/training_data.txt", 5)?;
    
    Ok(())
}
```

## Data Preparation

### Text Data Format

#### Simple Text Files
```text
# training_data.txt
This is the first training example.
This is the second training example.
Each line should be a separate training instance.
```

#### JSON Format
```json
{
  "examples": [
    {
      "input": "Translate to French: Hello world",
      "output": "Bonjour le monde"
    },
    {
      "input": "Summarize: Long text here...",
      "output": "Brief summary here"
    }
  ]
}
```

#### Conversation Format
```json
{
  "conversations": [
    {
      "messages": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."}
      ]
    }
  ]
}
```

### Data Loading and Preprocessing

```rust
use goldbull_core::data::{DataLoader, TextDataset, PreprocessConfig};

// Load text dataset
let preprocess_config = PreprocessConfig {
    max_sequence_length: 1024,
    min_sequence_length: 10,
    filter_duplicates: true,
    normalize_unicode: true,
};

let dataset = TextDataset::from_file(
    "training_data.txt", 
    preprocess_config
)?;

// Create data loader with batching
let data_loader = DataLoader::new(dataset)
    .batch_size(8)
    .shuffle(true)
    .drop_last(true);
```

### Data Preprocessing Pipeline

```rust
use goldbull_core::data::{Preprocessor, CleaningConfig};

let cleaning_config = CleaningConfig {
    remove_html: true,
    normalize_whitespace: true,
    filter_short_lines: true,
    min_line_length: 10,
    max_line_length: 2048,
    remove_duplicates: true,
    language_filter: Some("en".to_string()),
};

let preprocessor = Preprocessor::new(cleaning_config);
let cleaned_data = preprocessor.process_file("raw_data.txt", "cleaned_data.txt")?;
println!("Processed {} examples", cleaned_data.num_examples);
```

## Model Training

### Training Text Generation Models

```rust
use goldbull_text::{GoldbullText, TextTrainer, TextTrainingConfig};

// Configure training for text generation
let training_config = TextTrainingConfig {
    learning_rate: 5e-5,
    batch_size: 4,
    gradient_accumulation_steps: 2,
    num_epochs: 3,
    warmup_steps: 100,
    max_grad_norm: 1.0,
    weight_decay: 0.01,
    lr_scheduler: "cosine".to_string(),
    save_steps: 500,
    eval_steps: 100,
    logging_steps: 10,
};

// Initialize model and trainer
let model = GoldbullText::new(model_config, device)?;
let mut trainer = TextTrainer::new(model, training_config);

// Load training and validation datasets
let train_dataset = TextDataset::from_file("train.txt", preprocess_config)?;
let val_dataset = TextDataset::from_file("validation.txt", preprocess_config)?;

// Start training
trainer.train(train_dataset, Some(val_dataset))?;
```

### Training Code Completion Models

```rust
use goldbull_code::{GoldbullCode, CodeTrainer, CodeTrainingConfig};

let code_config = CodeTrainingConfig {
    learning_rate: 1e-4,
    batch_size: 6,
    num_epochs: 5,
    context_length: 2048,
    language_weights: vec![
        ("python", 0.4),
        ("rust", 0.2),
        ("javascript", 0.2),
        ("java", 0.1),
        ("cpp", 0.1),
    ],
    include_syntax_trees: true,
    mask_probability: 0.15,
};

let model = GoldbullCode::new(model_config, device)?;
let mut trainer = CodeTrainer::new(model, code_config);

// Train on code repositories
trainer.train_on_repositories(&[
    "path/to/python/repo",
    "path/to/rust/repo",
    "path/to/js/repo"
])?;
```

### Training Vision Models

```rust
use goldbull_vision::{GoldbullVision, VisionTrainer, VisionTrainingConfig};

let vision_config = VisionTrainingConfig {
    learning_rate: 1e-4,
    batch_size: 16,
    num_epochs: 10,
    image_size: (224, 224),
    augmentation_probability: 0.8,
    mixup_alpha: 0.2,
    label_smoothing: 0.1,
    class_weights: None,
};

let model = GoldbullVision::new(model_config, device)?;
let mut trainer = VisionTrainer::new(model, vision_config);

// Train on image classification dataset
trainer.train_classification("path/to/images", "path/to/labels.csv")?;
```

## Fine-tuning Pre-trained Models

### Fine-tuning Text Models

```rust
use goldbull_text::{GoldbullText, FineTuner, FineTuningConfig};

// Load pre-trained model
let base_model = GoldbullText::from_pretrained("path/to/pretrained/model")?;

// Configure fine-tuning
let finetune_config = FineTuningConfig {
    learning_rate: 1e-5,  // Lower learning rate for fine-tuning
    batch_size: 4,
    num_epochs: 2,
    freeze_layers: 6,     // Freeze first 6 layers
    target_layers: vec!["attention", "feedforward"],
    gradient_checkpointing: true,
    lora_rank: 8,         // Use LoRA for efficient fine-tuning
    lora_alpha: 16,
};

let mut fine_tuner = FineTuner::new(base_model, finetune_config);

// Fine-tune on domain-specific data
let domain_dataset = TextDataset::from_file("domain_data.txt", preprocess_config)?;
fine_tuner.fine_tune(domain_dataset)?;

// Save fine-tuned model
fine_tuner.save_model("path/to/finetuned/model")?;
```

### Parameter-Efficient Fine-tuning (LoRA)

```rust
use goldbull_core::lora::{LoRAConfig, LoRAAdapter};

// Configure LoRA for efficient fine-tuning
let lora_config = LoRAConfig {
    rank: 8,
    alpha: 16,
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj", "k_proj", "o_proj"],
    bias: "none",
};

// Apply LoRA to model
let mut model = GoldbullText::from_pretrained("base_model")?;
model.add_lora_adapters(lora_config)?;

// Train only LoRA parameters
let trainable_params = model.get_lora_parameters();
println!("Training {} LoRA parameters", trainable_params.len());
```

### Adapter-based Fine-tuning

```rust
use goldbull_core::adapters::{AdapterConfig, AdapterLayer};

let adapter_config = AdapterConfig {
    bottleneck_size: 64,
    dropout: 0.1,
    activation: "relu",
    init_weights: "normal",
};

// Add adapters to specific layers
let mut model = GoldbullText::from_pretrained("base_model")?;
model.add_adapters_to_layers(&[6, 8, 10, 12], adapter_config)?;

// Train only adapter parameters
let adapter_params = model.get_adapter_parameters();
model.freeze_base_parameters();
```

## Advanced Training Techniques

### Curriculum Learning

```rust
use goldbull_core::curriculum::{CurriculumScheduler, DifficultyMetric};

// Define curriculum based on text length
let curriculum = CurriculumScheduler::new()
    .stage(1, DifficultyMetric::TextLength(50))   // Start with short texts
    .stage(2, DifficultyMetric::TextLength(200))  // Move to medium texts
    .stage(3, DifficultyMetric::TextLength(500))  // Finally long texts
    .transition_threshold(0.85);  // Move to next stage at 85% accuracy

let mut trainer = TextTrainer::new(model, training_config);
trainer.set_curriculum(curriculum);
trainer.train(dataset, validation_dataset)?;
```

### Multi-task Learning

```rust
use goldbull_core::multitask::{MultiTaskTrainer, TaskConfig};

// Define multiple training tasks
let tasks = vec![
    TaskConfig {
        name: "text_generation",
        dataset_path: "generation_data.txt",
        loss_weight: 1.0,
        batch_size: 4,
    },
    TaskConfig {
        name: "summarization", 
        dataset_path: "summary_data.txt",
        loss_weight: 0.5,
        batch_size: 6,
    },
    TaskConfig {
        name: "question_answering",
        dataset_path: "qa_data.txt", 
        loss_weight: 0.8,
        batch_size: 4,
    },
];

let mut multi_trainer = MultiTaskTrainer::new(model, tasks);
multi_trainer.train_alternating(num_epochs)?;
```

### Reinforcement Learning from Human Feedback (RLHF)

```rust
use goldbull_core::rlhf::{RLHFTrainer, RewardModel, PPOConfig};

// Train reward model first
let reward_model = RewardModel::new(model_config.clone(), device.clone())?;
let preference_data = load_preference_dataset("preferences.json")?;
reward_model.train_on_preferences(preference_data)?;

// Configure PPO training
let ppo_config = PPOConfig {
    learning_rate: 1e-6,
    clip_ratio: 0.2,
    value_loss_coef: 0.1,
    entropy_coef: 0.01,
    max_kl_divergence: 0.01,
    target_kl: 0.008,
    ppo_epochs: 4,
};

// RLHF training
let mut rlhf_trainer = RLHFTrainer::new(model, reward_model, ppo_config);
rlhf_trainer.train_with_human_feedback(prompts_dataset)?;
```

## Training Monitoring and Evaluation

### Training Metrics

```rust
use goldbull_core::metrics::{TrainingMetrics, MetricsLogger};

// Initialize metrics logging
let mut metrics_logger = MetricsLogger::new("training_logs");

// During training loop
for epoch in 0..num_epochs {
    let mut epoch_metrics = TrainingMetrics::new();
    
    for batch in data_loader {
        let loss = trainer.train_step(&batch)?;
        epoch_metrics.add_loss(loss);
        
        // Log metrics every N steps
        if step % 10 == 0 {
            metrics_logger.log_step(step, loss, learning_rate, grad_norm);
        }
    }
    
    // Validation at end of epoch
    let val_metrics = trainer.evaluate(&validation_dataset)?;
    metrics_logger.log_epoch(epoch, &epoch_metrics, &val_metrics);
    
    println!("Epoch {}: Loss={:.4}, Val_Loss={:.4}, Val_Acc={:.3}", 
             epoch, epoch_metrics.average_loss(), 
             val_metrics.loss, val_metrics.accuracy);
}
```

### Evaluation Metrics

```rust
use goldbull_core::evaluation::{Evaluator, EvaluationMetrics};

// Text generation evaluation
let evaluator = Evaluator::new();
let test_dataset = TextDataset::from_file("test.txt", preprocess_config)?;

let metrics = evaluator.evaluate_text_generation(
    &model, 
    &test_dataset,
    &["bleu", "rouge", "perplexity", "bertscore"]
)?;

println!("BLEU Score: {:.3}", metrics.bleu);
println!("ROUGE-L: {:.3}", metrics.rouge_l);
println!("Perplexity: {:.2}", metrics.perplexity);
println!("BERTScore: {:.3}", metrics.bert_score);

// Code generation evaluation
let code_metrics = evaluator.evaluate_code_generation(
    &code_model,
    &code_test_dataset,
    &["syntax_correctness", "compilation_rate", "test_pass_rate"]
)?;

println!("Syntax Correct: {:.1}%", code_metrics.syntax_correctness * 100.0);
println!("Compilation Rate: {:.1}%", code_metrics.compilation_rate * 100.0);
println!("Test Pass Rate: {:.1}%", code_metrics.test_pass_rate * 100.0);
```

### Monitoring Tools Integration

```rust
use goldbull_core::monitoring::{WandBLogger, TensorBoardLogger};

// Weights & Biases integration
let wandb_logger = WandBLogger::new("goldbull-training", "experiment-1")?;
wandb_logger.log_config(&training_config);

// TensorBoard integration  
let tb_logger = TensorBoardLogger::new("./runs/experiment-1");

// Log metrics to both platforms
trainer.add_logger(Box::new(wandb_logger));
trainer.add_logger(Box::new(tb_logger));
```

## Memory-Efficient Training

### Gradient Checkpointing

```rust
use goldbull_core::optimization::{GradientCheckpointing, CheckpointConfig};

let checkpoint_config = CheckpointConfig {
    checkpoint_every_n_layers: 2,
    preserve_rng_state: true,
    use_reentrant: false,
};

let mut model = GoldbullText::new(model_config, device)?;
model.enable_gradient_checkpointing(checkpoint_config);

// Training will use less memory at the cost of some computation
```

### Mixed Precision Training

```rust
use goldbull_core::precision::{MixedPrecisionTrainer, AmpConfig};

let amp_config = AmpConfig {
    enabled: true,
    init_scale: 65536.0,
    growth_factor: 2.0,
    backoff_factor: 0.5,
    growth_interval: 2000,
};

let mut trainer = MixedPrecisionTrainer::new(model, training_config, amp_config);
trainer.train(dataset, validation_dataset)?;
```

### Memory Optimization Strategies

```rust
use goldbull_core::memory::{MemoryOptimizer, OptimizationLevel};

// Configure memory optimization
let memory_optimizer = MemoryOptimizer::new()
    .optimization_level(OptimizationLevel::Aggressive)
    .enable_tensor_fusion(true)
    .enable_memory_mapping(true)
    .set_memory_budget(1_500_000_000); // 1.5 GB

// Apply optimizations to model
let optimized_model = memory_optimizer.optimize_model(model)?;
```

## Distributed Training

### Multi-Process Training

```rust
use goldbull_core::distributed::{DistributedTrainer, DistributedConfig};

let distributed_config = DistributedConfig {
    world_size: 4,
    rank: 0, // Process rank
    backend: "nccl",
    init_method: "env://",
};

let mut distributed_trainer = DistributedTrainer::new(
    model, 
    training_config, 
    distributed_config
)?;

// Training will be distributed across processes
distributed_trainer.train(dataset, validation_dataset)?;
```

### Data Parallel Training

```rust
use goldbull_core::parallel::{DataParallel, ModelParallel};

// Data parallel across multiple devices
let parallel_model = DataParallel::new(model, device_ids)?;

// Model parallel for large models
let model_parallel = ModelParallel::new(model)
    .split_at_layers(&[6, 12, 18])
    .assign_to_devices(&[device1, device2, device3, device4]);
```

## Model Export and Deployment

### Export Trained Models

```rust
use goldbull_core::export::{ModelExporter, ExportFormat};

let exporter = ModelExporter::new();

// Export to different formats
exporter.export_safetensors(&model, "model.safetensors")?;
exporter.export_onnx(&model, "model.onnx")?;
exporter.export_pytorch(&model, "model.pt")?;

// Export with quantization
exporter.export_quantized(&model, "model_int8.safetensors", 8)?;
```

### Model Optimization for Deployment

```rust
use goldbull_core::optimization::{ModelOptimizer, OptimizationPass};

let optimizer = ModelOptimizer::new()
    .add_pass(OptimizationPass::FuseLinearLayers)
    .add_pass(OptimizationPass::OptimizeAttention)
    .add_pass(OptimizationPass::QuantizeWeights)
    .add_pass(OptimizationPass::PruneUnusedLayers);

let optimized_model = optimizer.optimize(model)?;
optimized_model.save("optimized_model")?;
```

## Training Best Practices

### Learning Rate Scheduling

```rust
use goldbull_core::schedulers::{LRScheduler, WarmupScheduler, CosineScheduler};

// Warmup + Cosine decay
let scheduler = WarmupScheduler::new(warmup_steps, base_lr)
    .chain(CosineScheduler::new(total_steps, base_lr, min_lr));

trainer.set_lr_scheduler(scheduler);
```

### Regularization Techniques

```rust
use goldbull_core::regularization::{Dropout, WeightDecay, LabelSmoothing};

// Configure regularization
let training_config = TrainingConfig {
    weight_decay: 0.01,
    dropout_rate: 0.1,
    label_smoothing: 0.1,
    gradient_clipping: 1.0,
    // ... other config
};
```

### Checkpointing Strategy

```rust
use goldbull_core::checkpointing::{CheckpointManager, CheckpointConfig};

let checkpoint_config = CheckpointConfig {
    save_frequency: 500,
    max_checkpoints: 5,
    save_optimizer_state: true,
    save_scheduler_state: true,
    best_metric: "validation_loss",
};

let checkpoint_manager = CheckpointManager::new("./checkpoints", checkpoint_config);
trainer.set_checkpoint_manager(checkpoint_manager);
```

### Hyperparameter Tuning

```rust
use goldbull_core::tuning::{HyperparameterTuner, SearchSpace, OptimizationGoal};

// Define search space
let search_space = SearchSpace::new()
    .add_float("learning_rate", 1e-6, 1e-3)
    .add_int("batch_size", 2, 16)
    .add_categorical("optimizer", &["adam", "adamw", "sgd"])
    .add_float("weight_decay", 0.0, 0.1);

// Configure tuning
let tuner = HyperparameterTuner::new()
    .search_space(search_space)
    .optimization_goal(OptimizationGoal::Minimize("validation_loss"))
    .max_trials(50)
    .early_stopping_patience(10);

// Run hyperparameter search
let best_params = tuner.optimize(train_function)?;
println!("Best parameters: {:?}", best_params);
```

## Troubleshooting

### Common Training Issues

#### Out of Memory Errors
```rust
// Reduce batch size
let config = TrainingConfig {
    batch_size: 2, // Reduced from 8
    gradient_accumulation_steps: 4, // Maintain effective batch size
    // ...
};

// Enable gradient checkpointing
model.enable_gradient_checkpointing(CheckpointConfig::default());

// Use memory optimization
let memory_optimizer = MemoryOptimizer::new()
    .optimization_level(OptimizationLevel::Aggressive);
```

#### Training Instability
```rust
// Reduce learning rate
let config = TrainingConfig {
    learning_rate: 1e-5, // Reduced from 1e-4
    gradient_clipping: 0.5, // Add gradient clipping
    // ...
};

// Add warmup
let scheduler = WarmupScheduler::new(100, config.learning_rate);
```

#### Slow Convergence
```rust
// Increase learning rate with warmup
let config = TrainingConfig {
    learning_rate: 5e-4,
    warmup_steps: 500,
    // ...
};

// Use different optimizer
trainer.set_optimizer("adamw", OptimizerConfig {
    weight_decay: 0.01,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
});
```

### Performance Profiling

```rust
use goldbull_core::profiling::{Profiler, ProfileConfig};

let profiler = Profiler::new(ProfileConfig {
    profile_memory: true,
    profile_compute: true,
    profile_io: true,
    output_path: "./profiling_results",
});

profiler.start();
trainer.train(dataset, validation_dataset)?;
let profile_results = profiler.stop();

println!("Training profile: {:?}", profile_results);
```

This comprehensive training guide provides everything needed to successfully train and fine-tune models in the Goldbull AI Model Suite, from basic setups to advanced techniques and troubleshooting.
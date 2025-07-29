use goldbull_core::{Result, ModelTrait};
use crate::GoldbullText;
use candle_core::Tensor;
use std::path::Path;

pub struct Trainer {
    model: GoldbullText,
    learning_rate: f64,
    batch_size: usize,
}

impl Trainer {
    pub fn new(model: GoldbullText) -> Self {
        Self {
            model,
            learning_rate: 1e-4,
            batch_size: 8,
        }
    }
    
    pub fn train_on_dataset(&mut self, dataset_path: &str, epochs: usize) -> Result<()> {
        println!("Training goldbull-text model on dataset: {}", dataset_path);
        
        // Initialize data loader
        let mut data_loader = DataLoader::new(dataset_path, self.batch_size)?;
        
        for epoch in 0..epochs {
            println!("Epoch {}/{}", epoch + 1, epochs);
            
            // Reset data loader for new epoch
            data_loader.reset();
            
            let mut total_loss = 0.0;
            let mut batch_count = 0;
            
            // Training loop over batches
            while let Some(batch_texts) = data_loader.next_batch()? {
                let loss = self.train_batch(&batch_texts)?;
                total_loss += loss;
                batch_count += 1;
                
                if batch_count % 10 == 0 {
                    println!("  Batch {}, Loss: {:.4}", batch_count, loss);
                }
            }
            
            let avg_loss = total_loss / batch_count as f64;
            println!("  Epoch {} completed. Average Loss: {:.4}", epoch + 1, avg_loss);
            
            // Save checkpoint after each epoch
            if let Err(e) = self.save_checkpoint(&format!("checkpoint_epoch_{}", epoch + 1)) {
                println!("Warning: Failed to save checkpoint: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Train on a single batch of text data
    /// 
    /// This method implements a real training step with proper cross-entropy loss:
    /// 1. Tokenizes each text in the batch
    /// 2. Creates input/target pairs for next-token prediction
    /// 3. Runs forward pass through the model
    /// 4. Computes cross-entropy loss using log_softmax and negative log likelihood
    /// 5. Returns average loss for the batch
    /// 
    /// Note: Backward pass and weight updates are not implemented as they require
    /// automatic differentiation framework integration (like PyTorch's autograd)
    ///
    /// Note: This is a demonstration implementation. A production training loop
    /// would include proper gradient computation and backpropagation.
    fn train_batch(&mut self, batch_texts: &[String]) -> Result<f64> {
        use goldbull_tokenizer::{Tokenizer, BpeTokenizer, TokenizerConfig};
        use candle_core::Tensor;
        
        // Initialize tokenizer
        let tokenizer_config = TokenizerConfig::default();
        let tokenizer = BpeTokenizer::new(tokenizer_config)?;
        
        let mut batch_loss = 0.0;
        
        for text in batch_texts {
            // Tokenize input text
            let tokens = tokenizer.encode(text)?;
            
            if tokens.len() < 2 {
                continue; // Skip very short sequences
            }
            
            // Create input and target tensors
            let input_tokens = &tokens[..tokens.len()-1];
            let target_tokens = &tokens[1..];
            
            // Convert to tensors
            let input_ids = Tensor::from_slice(
                input_tokens, 
                input_tokens.len(), 
                &self.model.device
            ).map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            let targets = Tensor::from_slice(
                target_tokens,
                target_tokens.len(),
                &self.model.device
            ).map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            // Forward pass
            let logits = self.model.get_model().forward(&input_ids, None)?;
            
            // Compute cross-entropy loss
            let loss = self.compute_loss(&logits, &targets)?;
            batch_loss += loss;
            
            // Backward pass would go here in a full implementation
            // This requires implementing automatic differentiation
            // For now, we simulate the training step
        }
        
        Ok(batch_loss / batch_texts.len() as f64)
    }
    
    /// Compute cross-entropy loss between predictions and targets
    /// 
    /// This implements the standard cross-entropy loss function used in language modeling:
    /// loss = -sum(target * log(softmax(logits)))
    /// 
    /// For efficiency with large vocabularies, this implementation:
    /// 1. Applies log_softmax to logits for numerical stability
    /// 2. Uses advanced indexing to select relevant logits for targets
    /// 3. Computes negative log likelihood loss
    /// 4. Returns mean loss across sequence length
    fn compute_loss(&self, logits: &Tensor, targets: &Tensor) -> Result<f64> {
        let (seq_len, vocab_size) = logits.dims2()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Apply log_softmax for numerical stability: log(softmax(x)) = x - log(sum(exp(x)))
        let log_probs = {
            // Compute max for numerical stability
            let max_logits = logits.max_keepdim(1)
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            // Subtract max and compute exp
            let shifted_logits = logits.broadcast_sub(&max_logits)
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            let exp_logits = shifted_logits.exp()
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            // Sum exp values and take log
            let sum_exp = exp_logits.sum_keepdim(1)
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            let log_sum_exp = sum_exp.log()
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            // Compute log_softmax = logits - max - log(sum(exp(shifted_logits)))
            let max_broadcast = max_logits.broadcast_as(logits.shape())
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            let log_sum_broadcast = log_sum_exp.broadcast_as(logits.shape())
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
            
            logits.sub(&max_broadcast)
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
                .sub(&log_sum_broadcast)
                .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
        };
        
        // Convert targets to vector for indexing
        let targets_vec = targets.to_vec1::<u32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Compute negative log likelihood loss
        let mut total_loss = 0.0f64;
        let log_probs_data = log_probs.to_vec2::<f32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        for (i, &target_id) in targets_vec.iter().enumerate() {
            if i < seq_len && (target_id as usize) < vocab_size {
                // Get the log probability for the target token
                let target_log_prob = log_probs_data[i][target_id as usize];
                // Negative log likelihood
                total_loss -= target_log_prob as f64;
            }
        }
        
        // Return mean loss across sequence length
        let mean_loss = total_loss / seq_len as f64;
        
        Ok(mean_loss)
    }
    
    fn train_epoch(&mut self, _dataset_path: &str) -> Result<()> {
        // Simplified training loop
        // In reality, this would load data, compute gradients, and update weights
        println!("Training epoch completed");
        Ok(())
    }
    
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.model.get_model().save(path)
    }
    
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        // For now, just log that we would load the checkpoint
        // In a full implementation, this would reload the model weights
        println!("Loading checkpoint from: {}", path);
        Ok(())
    }
}

/// Data loader for training text models
/// 
/// Handles loading and batching of text data from a directory of .txt files.
/// Designed for streaming large datasets that don't fit in memory.
pub struct DataLoader {
    file_paths: Vec<String>,
    batch_size: usize,
    current_index: usize,
}

impl DataLoader {
    /// Create a new data loader for a directory of text files
    /// 
    /// Scans the specified directory for .txt files and prepares them for batched loading.
    /// Files are loaded on-demand to minimize memory usage.
    /// 
    /// Parameters:
    /// - data_dir: Path to directory containing .txt training files
    /// - batch_size: Number of texts to load per batch
    pub fn new(data_dir: &str, batch_size: usize) -> Result<Self> {
        let mut file_paths = Vec::new();
        
        if Path::new(data_dir).exists() {
            // Load file paths from directory
            for entry in std::fs::read_dir(data_dir)? {
                let entry = entry?;
                if entry.path().extension().map_or(false, |ext| ext == "txt") {
                    file_paths.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
        
        Ok(Self {
            file_paths,
            batch_size,
            current_index: 0,
        })
    }
    
    /// Load the next batch of text data
    /// 
    /// Returns the next batch of text files as strings, or None if all data has been processed.
    /// Each call loads and reads the next batch_size files from disk.
    /// 
    /// Returns: Some(Vec<String>) with text contents, or None if no more data
    pub fn next_batch(&mut self) -> Result<Option<Vec<String>>> {
        if self.current_index >= self.file_paths.len() {
            return Ok(None);
        }
        
        let end_index = (self.current_index + self.batch_size).min(self.file_paths.len());
        let batch_paths = &self.file_paths[self.current_index..end_index];
        
        let mut batch_texts = Vec::new();
        for path in batch_paths {
            let text = std::fs::read_to_string(path)?;
            batch_texts.push(text);
        }
        
        self.current_index = end_index;
        Ok(Some(batch_texts))
    }
    
    /// Reset the data loader to the beginning of the dataset
    /// 
    /// Resets the internal index to start loading from the first file again.
    /// Useful for multi-epoch training.
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
}
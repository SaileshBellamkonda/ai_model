use goldbull_core::{Result, ModelTrait};
use crate::GoldbullTextModel;
use candle_core::{Tensor, IndexOp};
use rand::prelude::*;

pub struct TextGenerator<'a> {
    model: &'a GoldbullTextModel,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

impl<'a> TextGenerator<'a> {
    pub fn new(model: GoldbullTextModel) -> TextGenerator<'static> {
        // This method is kept for backward compatibility but leaks memory
        // Use new_with_ref instead
        let model = Box::leak(Box::new(model));
        TextGenerator {
            model,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
        }
    }
    
    pub fn new_with_ref(model: &'a GoldbullTextModel) -> Self {
        Self {
            model,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
        }
    }
    
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }
    
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }
    
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }
    
    pub fn generate_with_sampling(
        &self,
        input_ids: &Tensor,
        max_length: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = Vec::new();
        let mut current_input = input_ids.clone();
        
        // Extract initial tokens
        let initial_tokens = input_ids.to_vec1::<u32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        tokens.extend(initial_tokens);
        
        for _ in tokens.len()..max_length {
            let logits = self.model.forward(&current_input, None)?;
            
            // Get last token logits
            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            
            // Apply temperature scaling
            let scaled_logits = if self.temperature != 1.0 {
                (last_logits / self.temperature as f64)?
            } else {
                last_logits
            };
            
            // Apply top-k filtering
            let filtered_logits = self.apply_top_k(&scaled_logits)?;
            
            // Apply top-p (nucleus) filtering
            let final_logits = self.apply_top_p(&filtered_logits)?;
            
            // Sample from the distribution
            let next_token_id = self.sample_token(&final_logits)?;
            tokens.push(next_token_id);
            
            // Check for EOS token to stop generation early
            // Assuming EOS token ID is typically 2 (common in many models)
            // This could be made configurable
            if next_token_id == 2 {  // EOS token
                break;
            }
            
            // Update input for next iteration
            current_input = Tensor::new(&[next_token_id], current_input.device())?
                .unsqueeze(0)?;
        }
        
        Ok(tokens)
    }
    
    fn apply_top_k(&self, logits: &Tensor) -> Result<Tensor> {
        if self.top_k == 0 {
            return Ok(logits.clone());
        }
        
        let device = logits.device();
        let dims = logits.dims();
        let vocab_size = dims[dims.len() - 1];
        
        if self.top_k >= vocab_size {
            return Ok(logits.clone());
        }
        
        // Get the values and sort them to find the k-th largest
        let logits_vec = logits.flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let mut sorted_logits = logits_vec.clone();
        sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find the threshold (k-th largest value)
        let threshold = sorted_logits.get(self.top_k - 1).copied().unwrap_or(f32::NEG_INFINITY);
        
        // Create a mask where only top-k elements are kept, others set to -infinity
        let masked_logits: Vec<f32> = logits_vec.iter().map(|&val| {
            if val >= threshold {
                val
            } else {
                f32::NEG_INFINITY
            }
        }).collect();
        
        let result = Tensor::from_vec(masked_logits, dims, device)?;
        Ok(result)
    }
    
    fn apply_top_p(&self, logits: &Tensor) -> Result<Tensor> {
        if self.top_p >= 1.0 {
            return Ok(logits.clone());
        }
        
        let device = logits.device();
        let dims = logits.dims();
        
        // Convert logits to probabilities
        let probs = candle_nn::ops::softmax(logits, logits.dims().len() - 1)?;
        let probs_vec = probs.flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Create index-probability pairs and sort by probability (descending)
        let mut indexed_probs: Vec<(usize, f32)> = probs_vec.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate cumulative probabilities and find cutoff
        let mut cumulative_prob = 0.0;
        let mut cutoff_index = indexed_probs.len();
        
        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative_prob += prob;
            if cumulative_prob >= self.top_p {
                cutoff_index = i + 1;
                break;
            }
        }
        
        // Create mask: keep probabilities in top-p, set others to -infinity
        let logits_vec = logits.flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let kept_indices: std::collections::HashSet<usize> = indexed_probs[..cutoff_index]
            .iter()
            .map(|(idx, _)| *idx)
            .collect();
        
        let masked_logits: Vec<f32> = logits_vec.iter().enumerate().map(|(i, &val)| {
            if kept_indices.contains(&i) {
                val
            } else {
                f32::NEG_INFINITY
            }
        }).collect();
        
        let result = Tensor::from_vec(masked_logits, dims, device)?;
        Ok(result)
    }
    
    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        // Convert logits to probabilities
        let probs = candle_nn::ops::softmax(logits, logits.dims().len() - 1)?;
        let probs_vec = probs.flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Filter out infinite or invalid probabilities
        let valid_probs: Vec<(usize, f32)> = probs_vec.iter()
            .enumerate()
            .filter(|(_, &p)| p.is_finite() && p > 0.0)
            .map(|(i, &p)| (i, p))
            .collect();
        
        if valid_probs.is_empty() {
            // Fallback to argmax if no valid probabilities
            let token_id = probs.argmax(probs.dims().len() - 1)?
                .to_scalar::<u32>()?;
            return Ok(token_id);
        }
        
        // Weighted random sampling
        let total_prob: f32 = valid_probs.iter().map(|(_, p)| p).sum();
        let mut rng = thread_rng();
        let mut random_value = rng.gen::<f32>() * total_prob;
        
        for &(index, prob) in &valid_probs {
            random_value -= prob;
            if random_value <= 0.0 {
                return Ok(index as u32);
            }
        }
        
        // Fallback: return the last valid index
        Ok(valid_probs.last().unwrap().0 as u32)
    }
}
use crate::{Result, ModelConfig};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait ModelTrait {
    fn config(&self) -> &ModelConfig;
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;
    fn generate(&self, input_ids: &Tensor, max_length: usize) -> Result<Vec<u32>>;
    fn save(&self, path: &str) -> Result<()>;
    fn load(path: &str, device: &Device) -> Result<Self> where Self: Sized;
}

/// Memory pool for efficient tensor reuse
#[derive(Debug)]
pub struct TensorPool {
    tensors: Arc<Mutex<Vec<Tensor>>>,
    max_size: usize,
}

impl TensorPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            tensors: Arc::new(Mutex::new(Vec::new())),
            max_size,
        }
    }
    
    pub fn get_or_create(&self, shape: &[usize], dtype: candle_core::DType, device: &Device) -> Result<Tensor> {
        let mut tensors = self.tensors.lock().unwrap();
        
        // Try to find a compatible tensor
        for (i, tensor) in tensors.iter().enumerate() {
            if tensor.shape().dims() == shape && 
               tensor.dtype() == dtype && 
               tensor.device().same_device(device) {
                return Ok(tensors.remove(i));
            }
        }
        
        // Create new tensor if none found
        Tensor::zeros(shape, dtype, device)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))
    }
    
    pub fn return_tensor(&self, tensor: Tensor) {
        let mut tensors = self.tensors.lock().unwrap();
        if tensors.len() < self.max_size {
            tensors.push(tensor);
        }
    }
    
    pub fn clear(&self) {
        let mut tensors = self.tensors.lock().unwrap();
        tensors.clear();
    }
}

#[derive(Debug)]
pub struct Model {
    pub config: ModelConfig,
    pub device: Device,
    pub weights: HashMap<String, Tensor>,
    pub tensor_pool: TensorPool,
    pub memory_usage: Arc<Mutex<usize>>,
}

impl Model {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let pool_size = if config.memory_optimization.tensor_pooling { 50 } else { 0 };
        
        Ok(Self {
            config,
            device,
            weights: HashMap::new(),
            tensor_pool: TensorPool::new(pool_size),
            memory_usage: Arc::new(Mutex::new(0)),
        })
    }
    
    pub fn get_memory_usage(&self) -> usize {
        let weights_memory = self.weights.values()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum::<usize>();
        
        let cached_memory = *self.memory_usage.lock().unwrap();
        weights_memory + cached_memory
    }
    
    pub fn is_within_memory_limit(&self) -> bool {
        let max_memory_bytes = self.config.memory_optimization.max_memory_mb * 1024 * 1024;
        self.get_memory_usage() < max_memory_bytes
    }
    
    /// Clean up memory when usage exceeds threshold
    pub fn cleanup_memory_if_needed(&self) -> Result<()> {
        let current_usage = self.get_memory_usage();
        let max_memory = self.config.memory_optimization.max_memory_mb * 1024 * 1024;
        let threshold = (max_memory as f32 * self.config.memory_optimization.memory_cleanup_threshold) as usize;
        
        if current_usage > threshold {
            // Clear tensor pool
            self.tensor_pool.clear();
            
            // Update memory tracking
            *self.memory_usage.lock().unwrap() = 0;
            
            // Force garbage collection hint to the allocator
            // This is platform-specific and optional
            #[cfg(target_os = "linux")]
            {
                // On Linux, we can try to release memory back to the OS
                unsafe {
                    libc::malloc_trim(0);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get optimal batch size based on available memory
    pub fn get_optimal_batch_size(&self, sequence_length: usize) -> usize {
        let available_memory = self.config.memory_optimization.max_memory_mb * 1024 * 1024;
        let current_usage = self.get_memory_usage();
        let free_memory = available_memory.saturating_sub(current_usage);
        
        // Estimate memory per sample (rough calculation)
        let hidden_size = self.config.hidden_size;
        let bytes_per_token = match self.config.quantization.activation_dtype {
            crate::config::ActivationDType::F32 => 4,
            crate::config::ActivationDType::F16 | crate::config::ActivationDType::BF16 => 2,
        };
        
        let memory_per_sample = sequence_length * hidden_size * bytes_per_token * 4; // Rough overhead factor
        
        if memory_per_sample == 0 {
            return 1;
        }
        
        let max_batch_size = free_memory / memory_per_sample;
        std::cmp::max(1, std::cmp::min(max_batch_size, 32)) // Cap at 32 for safety
    }
    
    /// Calculate current CPU utilization optimizations
    pub fn get_cpu_optimizations(&self) -> CpuOptimizationInfo {
        let num_cores = num_cpus::get();
        let optimal_threads = self.config.cpu_optimization.num_threads
            .unwrap_or_else(|| std::cmp::min(num_cores, 8)); // Cap at 8 threads
        
        CpuOptimizationInfo {
            num_threads: optimal_threads,
            use_simd: self.config.cpu_optimization.use_simd,
            cache_friendly: self.config.cpu_optimization.cache_friendly_layout,
            progressive_loading: self.config.cpu_optimization.progressive_loading,
            total_cpu_cores: num_cores,
        }
    }
}

#[derive(Debug)]
pub struct CpuOptimizationInfo {
    pub num_threads: usize,
    pub use_simd: bool,
    pub cache_friendly: bool,
    pub progressive_loading: bool,
    pub total_cpu_cores: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 1024,
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            do_sample: true,
            pad_token_id: None,
            eos_token_id: None,
        }
    }
}
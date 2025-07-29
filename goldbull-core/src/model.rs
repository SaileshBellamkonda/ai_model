use crate::{Result, ModelConfig};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub trait ModelTrait {
    fn config(&self) -> &ModelConfig;
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor>;
    fn generate(&self, input_ids: &Tensor, max_length: usize) -> Result<Vec<u32>>;
    fn save(&self, path: &str) -> Result<()>;
    fn load(path: &str, device: &Device) -> Result<Self> where Self: Sized;
}

#[derive(Debug)]
pub struct Model {
    pub config: ModelConfig,
    pub device: Device,
    pub weights: HashMap<String, Tensor>,
}

impl Model {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        Ok(Self {
            config,
            device,
            weights: HashMap::new(),
        })
    }
    
    pub fn get_memory_usage(&self) -> usize {
        self.weights.values()
            .map(|tensor| tensor.elem_count() * tensor.dtype().size_in_bytes())
            .sum()
    }
    
    pub fn is_within_memory_limit(&self) -> bool {
        const MAX_MEMORY_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2GB
        self.get_memory_usage() < MAX_MEMORY_BYTES
    }
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
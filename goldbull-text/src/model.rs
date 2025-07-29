use goldbull_core::{Result, ModelConfig, ModelTrait, Model};
use candle_core::{Device, Tensor, Module};
use candle_nn::{Linear, Embedding, VarBuilder};

pub struct GoldbullTextModel {
    pub base: Model,
    pub embeddings: Embedding,
    pub layers: Vec<super::transformer::TransformerLayer>,
    pub lm_head: Linear,
    pub vocab_size: usize,
}

impl GoldbullTextModel {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        
        // Initialize VarBuilder for weight loading and initialization
        let vs = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&vs, candle_core::DType::F32, &device);
        
        // Initialize model components with proper weight initialization
        let embeddings = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embeddings"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = super::transformer::TransformerLayer::new(
                &config,
                vb.pp(&format!("layers.{}", i)),
            )?;
            layers.push(layer);
        }
        
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let base = Model::new(config, device)?;
        
        Ok(Self {
            base,
            embeddings,
            layers,
            lm_head,
            vocab_size,
        })
    }
    
    /// Create model from pre-trained weights file (safetensors format)
    /// 
    /// This method loads a model from pre-trained weights stored in safetensors format.
    /// Safetensors is a safe, fast serialization format for ML tensors.
    /// 
    /// Process:
    /// 1. Validates that weights file exists
    /// 2. Loads tensors from safetensors file using candle
    /// 3. Creates VarBuilder from loaded tensors
    /// 4. Initializes model components with loaded weights
    /// 
    /// Parameters:
    /// - weights_path: Path to .safetensors file containing model weights
    /// - config: Model configuration (architecture, sizes, etc.)
    /// - device: Compute device (CPU/GPU) for model execution
    pub fn from_pretrained(weights_path: &str, config: ModelConfig, device: Device) -> Result<Self> {
        use std::path::Path;
        
        if !Path::new(weights_path).exists() {
            return Err(goldbull_core::GoldbullError::Model(
                format!("Weights file not found: {}", weights_path)
            ));
        }
        
        // Load weights from safetensors file
        let weights = candle_core::safetensors::load(weights_path, &device)
            .map_err(|e| goldbull_core::GoldbullError::Model(format!("Failed to load weights: {}", e)))?;
        
        let vb = VarBuilder::from_tensors(weights, candle_core::DType::F32, &device);
        
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        
        // Load model components from pre-trained weights
        let embeddings = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embeddings"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let layer = super::transformer::TransformerLayer::new(
                &config,
                vb.pp(&format!("layers.{}", i)),
            )?;
            layers.push(layer);
        }
        
        let lm_head = candle_nn::linear(hidden_size, vocab_size, vb.pp("lm_head"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let base = Model::new(config, device)?;
        
        Ok(Self {
            base,
            embeddings,
            layers,
            lm_head,
            vocab_size,
        })
    }
    
    pub fn get_memory_usage(&self) -> usize {
        self.base.get_memory_usage()
    }
    
    pub fn is_within_memory_limit(&self) -> bool {
        self.base.is_within_memory_limit()
    }
}

impl ModelTrait for GoldbullTextModel {
    fn config(&self) -> &ModelConfig {
        &self.base.config
    }
    
    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Embedding layer
        let mut hidden_states = self.embeddings.forward(input_ids)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        
        // Language modeling head
        let logits = self.lm_head.forward(&hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(logits)
    }
    
    fn generate(&self, input_ids: &Tensor, max_length: usize) -> Result<Vec<u32>> {
        // Use the advanced TextGenerator for proper sampling
        let generator = crate::generation::TextGenerator::new_with_ref(self)
            .with_temperature(0.8)
            .with_top_p(0.9)
            .with_top_k(50);
        
        generator.generate_with_sampling(input_ids, max_length)
    }
    
    fn save(&self, path: &str) -> Result<()> {
        // Save model weights - simplified for now
        goldbull_core::utils::ensure_dir_exists(path)?;
        std::fs::write(format!("{}/config.json", path), serde_json::to_string_pretty(&self.config())?)?;
        Ok(())
    }
    
    fn load(path: &str, device: &Device) -> Result<Self> {
        let config_path = format!("{}/config.json", path);
        let weights_path = format!("{}/model.safetensors", path);
        
        // Load configuration
        if !std::path::Path::new(&config_path).exists() {
            return Err(goldbull_core::GoldbullError::Model(
                format!("Config file not found: {}", config_path)
            ));
        }
        
        let config_content = std::fs::read_to_string(config_path)?;
        let config: ModelConfig = serde_json::from_str(&config_content)?;
        
        // Try to load from pre-trained weights, fall back to new model if not found
        if std::path::Path::new(&weights_path).exists() {
            Self::from_pretrained(&weights_path, config, device.clone())
        } else {
            // No weights found, create new model with random initialization
            println!("Warning: No pre-trained weights found at {}, using random initialization", weights_path);
            Self::new(config, device.clone())
        }
    }
}
pub mod model;
pub mod transformer;
pub mod training;
pub mod generation;

use goldbull_core::{Result, ModelConfig, ModelTrait};
use goldbull_tokenizer::{Tokenizer, TikTokenizer};
use candle_core::{Device, Tensor};

pub use model::GoldbullTextModel;
pub use transformer::TransformerLayer;
pub use generation::TextGenerator;

/// Entry point for the goldbull-text model
pub struct GoldbullText {
    model: GoldbullTextModel,
    tokenizer: TikTokenizer,
    device: Device,
}

impl GoldbullText {
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let model = GoldbullTextModel::new(config, device.clone())?;
        let tokenizer_config = goldbull_tokenizer::TokenizerConfig::default();
        let tokenizer = TikTokenizer::new(tokenizer_config)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
    
    pub fn from_pretrained(model_path: &str) -> Result<Self> {
        let device = Device::Cpu;
        let model = GoldbullTextModel::load(model_path, &device)?;
        let tokenizer_config = goldbull_tokenizer::TokenizerConfig::default();
        let tokenizer = TikTokenizer::new(tokenizer_config)?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
    
    pub fn generate_text(&self, prompt: &str, max_length: usize) -> Result<String> {
        let input_tokens = self.tokenizer.encode(prompt)?;
        let input_tensor = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| goldbull_core::GoldbullError::Tensor(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| goldbull_core::GoldbullError::Tensor(e.to_string()))?;
        
        let output_tokens = self.model.generate(&input_tensor, max_length)?;
        let generated_text = self.tokenizer.decode(&output_tokens)?;
        
        Ok(generated_text)
    }
    
    pub fn get_memory_usage(&self) -> usize {
        self.model.get_memory_usage()
    }
    
    pub fn is_within_memory_limit(&self) -> bool {
        self.model.is_within_memory_limit()
    }
}
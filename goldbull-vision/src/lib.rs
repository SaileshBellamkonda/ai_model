pub mod model;
pub mod training;
pub mod vision;

pub use model::GoldbullVision;
pub use training::Trainer;
pub use vision::{VisionRequest, VisionResponse, VisionTask};

use anyhow::Result;
use goldbull_core::ModelConfig;
use candle_core::Device;

/// Create a new computer vision model with default configuration
pub fn new_vision_model(device: Device) -> Result<GoldbullVision> {
    let config = ModelConfig::vision();
    GoldbullVision::new(config, device)
}

/// Process image with vision model
pub async fn process_image(
    model: &GoldbullVision,
    request: VisionRequest,
) -> Result<VisionResponse> {
    model.process(request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use goldbull_core::config::ModelType;

    #[test]
    fn test_vision_model_creation() {
        let config = ModelConfig::vision();
        assert_eq!(config.model_name, "goldbull-vision");
        assert_eq!(config.model_type, ModelType::Vision);
        assert_eq!(config.hidden_size, 512);
    }

    #[test]
    fn test_vision_request_creation() {
        let request = VisionRequest {
            image_data: vec![255u8; 224 * 224 * 3],
            task: VisionTask::Classification,
            max_results: 10,
            ..Default::default()
        };
        
        assert_eq!(request.image_data.len(), 224 * 224 * 3);
        assert_eq!(request.task, VisionTask::Classification);
        assert_eq!(request.max_results, 10);
    }
    
    #[test]
    fn test_model_config_vision() {
        let config = ModelConfig::vision();
        assert!(config.hidden_size > 0);
        assert!(config.num_layers > 0);
        assert_eq!(config.vocab_size, 1000); // ImageNet classes
    }
}
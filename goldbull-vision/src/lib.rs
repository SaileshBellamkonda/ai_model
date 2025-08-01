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

    #[tokio::test]
    async fn test_vision_processing() {
        let model = new_vision_model(Device::Cpu).unwrap();
        
        let request = VisionRequest {
            image_data: vec![0u8; 224 * 224 * 3], // Dummy RGB image data
            task: VisionTask::Classification,
            max_results: 5,
            ..Default::default()
        };
        
        let response = process_image(&model, request).await.unwrap();
        assert!(!response.results.is_empty());
        assert!(response.confidence > 0.0);
    }
}
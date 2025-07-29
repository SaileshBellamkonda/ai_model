pub mod model;
pub mod training;
pub mod multimodal;

pub use model::GoldbullMultimodel;
pub use training::Trainer;
pub use multimodal::{MultimodalRequest, MultimodalResponse, ModalityType};

use anyhow::Result;
use goldbull_core::ModelConfig;
use candle_core::Device;

/// Create a new multimodal AI model with default configuration
pub fn new_multimodal_model(device: Device) -> Result<GoldbullMultimodel> {
    let config = ModelConfig::multimodal();
    GoldbullMultimodel::new(config, device)
}

/// Process multimodal input
pub async fn process_multimodal(
    model: &GoldbullMultimodel,
    request: MultimodalRequest,
) -> Result<MultimodalResponse> {
    model.process(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multimodal_processing() {
        let model = new_multimodal_model(Device::Cpu).unwrap();
        
        let request = MultimodalRequest {
            text: Some("Describe this image".to_string()),
            image_data: Some(vec![0u8; 224 * 224 * 3]),
            audio_data: None,
            modalities: vec![ModalityType::Text, ModalityType::Vision],
            ..Default::default()
        };
        
        let response = process_multimodal(&model, request).await.unwrap();
        assert!(!response.text_output.unwrap_or_default().is_empty());
        assert!(response.confidence > 0.0);
    }
}
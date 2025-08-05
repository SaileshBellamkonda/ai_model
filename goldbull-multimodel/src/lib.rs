pub mod model;
pub mod training;
pub mod multimodal;

pub use model::GoldbullMultimodel;
pub use training::Trainer;
pub use multimodal::{MultimodalRequest, MultimodalResponse, ModalityType, ModalityInput, InputModality};

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
    use std::collections::HashMap;

    #[test]
    fn test_multimodal_request_creation() {
        let request = MultimodalRequest {
            inputs: vec![
                ModalityInput {
                    modality: InputModality::Text { content: "Describe this image".to_string() },
                    metadata: HashMap::new(),
                },
                ModalityInput {
                    modality: InputModality::Image { data: vec![0u8; 224 * 224 * 3] },
                    metadata: HashMap::new(),
                },
            ],
            output_modalities: vec![ModalityType::Text, ModalityType::Vision],
            options: HashMap::new(),
        };
        
        assert_eq!(request.inputs.len(), 2);
        assert_eq!(request.output_modalities.len(), 2);
        assert!(matches!(request.inputs[0].modality, InputModality::Text { .. }));
        assert!(matches!(request.inputs[1].modality, InputModality::Image { .. }));
    }

    #[test]
    fn test_modality_types() {
        let text_type = ModalityType::Text;
        let vision_type = ModalityType::Vision;
        let audio_type = ModalityType::Audio;
        
        assert!(matches!(text_type, ModalityType::Text));
        assert!(matches!(vision_type, ModalityType::Vision));
        assert!(matches!(audio_type, ModalityType::Audio));
    }

    #[test]
    fn test_model_config_multimodal() {
        let config = ModelConfig::multimodal();
        assert!(config.model_name.contains("multimodal") || config.model_name.contains("multi"));
    }
}
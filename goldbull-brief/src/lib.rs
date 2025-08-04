pub mod model;
pub mod training;
pub mod summarization;

pub use model::GoldbullBrief;
pub use training::Trainer;
pub use summarization::{SummarizationRequest, SummarizationResponse, SummaryType, SummaryStyle};

use anyhow::Result;
use goldbull_core::ModelConfig;
use candle_core::Device;

/// Create a new text summarization model with default configuration
pub fn new_summarization_model(device: Device) -> Result<GoldbullBrief> {
    let config = ModelConfig::summarization();
    GoldbullBrief::new(config, device)
}

/// Summarize text given input
pub async fn summarize_text(
    model: &GoldbullBrief,
    request: SummarizationRequest,
) -> Result<SummarizationResponse> {
    model.summarize(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarization_request_creation() {
        let request = SummarizationRequest {
            text: "Test text".to_string(),
            summary_type: SummaryType::Abstractive,
            summary_style: SummaryStyle::Concise,
            max_length: 50,
            min_length: 10,
            ..Default::default()
        };
        
        assert_eq!(request.text, "Test text");
        assert_eq!(request.summary_type, SummaryType::Abstractive);
        assert_eq!(request.summary_style, SummaryStyle::Concise);
        assert_eq!(request.max_length, 50);
        assert_eq!(request.min_length, 10);
    }

    #[test]
    fn test_model_config() {
        let config = ModelConfig::summarization();
        assert_eq!(config.model_name, "goldbull-brief");
        assert!(config.vocab_size > 0);
        assert!(config.hidden_size > 0);
    }
}
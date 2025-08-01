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

    #[tokio::test]
    async fn test_text_summarization() {
        let model = new_summarization_model(Device::Cpu).unwrap();
        
        let request = SummarizationRequest {
            text: "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. Colloquially, the term 'artificial intelligence' is often used to describe machines (or computers) that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving'.".to_string(),
            summary_type: SummaryType::Abstractive,
            summary_style: SummaryStyle::Concise,
            max_length: 50,
            min_length: 10,
            ..Default::default()
        };
        
        let response = summarize_text(&model, request).await.unwrap();
        assert!(!response.summary.is_empty());
        assert!(response.confidence > 0.0);
    }
}
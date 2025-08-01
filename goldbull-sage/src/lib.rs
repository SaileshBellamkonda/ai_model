pub mod model;
pub mod training;
pub mod qa;

pub use model::GoldbullSage;
pub use training::Trainer;
pub use qa::{QARequest, QAResponse, QuestionType};

use anyhow::Result;
use goldbull_core::ModelConfig;
use candle_core::Device;

/// Create a new question answering model with default configuration
pub fn new_qa_model(device: Device) -> Result<GoldbullSage> {
    let config = ModelConfig::question_answering();
    GoldbullSage::new(config, device)
}

/// Answer a question given context
pub async fn answer_question(
    model: &GoldbullSage,
    request: QARequest,
) -> Result<QAResponse> {
    model.answer(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_question_answering() {
        let model = new_qa_model(Device::Cpu).unwrap();
        
        let request = QARequest {
            question: "What is the capital of France?".to_string(),
            context: Some("Paris is the capital and most populous city of France.".to_string()),
            question_type: QuestionType::Factual,
            max_answer_length: 50,
            ..Default::default()
        };
        
        let response = answer_question(&model, request).await.unwrap();
        assert!(!response.answer.is_empty());
        assert!(response.confidence > 0.0);
    }
}
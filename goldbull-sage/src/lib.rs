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
    use goldbull_core::config::ModelType;

    #[test]
    fn test_qa_request_creation() {
        let request = QARequest {
            question: "What is the capital of France?".to_string(),
            context: Some("Paris is the capital and most populous city of France.".to_string()),
            question_type: QuestionType::Factual,
            max_answer_length: 50,
            temperature: 0.1,
            use_context: true,
            ..Default::default()
        };
        
        assert_eq!(request.question, "What is the capital of France?");
        assert_eq!(request.context, Some("Paris is the capital and most populous city of France.".to_string()));
        assert_eq!(request.question_type, QuestionType::Factual);
        assert_eq!(request.max_answer_length, 50);
        assert_eq!(request.temperature, 0.1);
        assert!(request.use_context);
    }

    #[test]
    fn test_question_answering_config() {
        let config = ModelConfig::question_answering();
        assert_eq!(config.model_name, "goldbull-sage");
        assert_eq!(config.model_type, ModelType::QuestionAnswering);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.max_sequence_length, 1024);
    }
    
    #[test]
    fn test_question_type_variants() {
        let types = [
            QuestionType::Factual,
            QuestionType::Analytical,
            QuestionType::YesNo,
            QuestionType::MultipleChoice,
            QuestionType::OpenEnded,
            QuestionType::Definition,
            QuestionType::Procedural,
            QuestionType::Summarization,
        ];
        
        for question_type in types {
            let request = QARequest {
                question: "Test question".to_string(),
                question_type,
                ..Default::default()
            };
            assert_eq!(request.question_type, question_type);
        }
    }
}
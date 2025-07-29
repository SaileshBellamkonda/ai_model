pub mod model;
pub mod training;
pub mod generation;
pub mod completion;
pub mod syntax;

pub use model::GoldbullCode;
pub use training::Trainer;
pub use generation::CodeGenerator;
pub use completion::{CompletionRequest, CompletionResponse, CodeContext};
pub use syntax::{SyntaxAnalyzer, LanguageType, CodeFeatures};

use anyhow::Result;
use goldbull_core::ModelConfig;
use goldbull_tokenizer::BpeTokenizer;
use candle_core::Device;

/// Create a new code completion model with default configuration
pub fn new_code_model(device: Device) -> Result<GoldbullCode> {
    let config = ModelConfig::code_completion();
    GoldbullCode::new(config, device)
}

/// Analyze code syntax and extract features for completion
pub fn analyze_code(code: &str, language: LanguageType) -> Result<CodeFeatures> {
    let mut analyzer = SyntaxAnalyzer::new(language)?;
    analyzer.analyze(code)
}

/// Complete code given a prefix and context
pub async fn complete_code(
    model: &GoldbullCode,
    request: CompletionRequest,
) -> Result<CompletionResponse> {
    let generator = CodeGenerator::new(model.clone())?;
    generator.complete(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_code_completion() {
        let model = new_code_model(Device::Cpu).unwrap();
        
        let request = CompletionRequest {
            prefix: "fn fibonacci(n: u32) -> u32 {".to_string(),
            suffix: Some("}".to_string()),
            language: LanguageType::Rust,
            max_tokens: 100,
            temperature: 0.2,
            ..Default::default()
        };
        
        let response = complete_code(&model, request).await.unwrap();
        assert!(!response.completion.is_empty());
        assert!(response.confidence > 0.0);
    }

    #[test]
    fn test_syntax_analysis() {
        let code = r#"
            fn main() {
                println!("Hello, world!");
            }
        "#;
        
        let features = analyze_code(code, LanguageType::Rust).unwrap();
        assert!(features.functions.len() > 0);
        assert_eq!(features.language, LanguageType::Rust);
    }
}
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
    let mut generator = CodeGenerator::new(model)?;
    generator.complete(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completion_request_creation() {
        let request = CompletionRequest {
            prefix: "fn fibonacci(n: u32) -> u32 {".to_string(),
            suffix: Some("}".to_string()),
            language: LanguageType::Rust,
            max_tokens: 100,
            temperature: 0.2,
            ..Default::default()
        };
        
        assert_eq!(request.prefix, "fn fibonacci(n: u32) -> u32 {");
        assert_eq!(request.suffix, Some("}".to_string()));
        assert!(matches!(request.language, LanguageType::Rust));
        assert_eq!(request.max_tokens, 100);
        assert_eq!(request.temperature, 0.2);
    }

    #[test]
    fn test_language_types() {
        let rust_lang = LanguageType::Rust;
        let python_lang = LanguageType::Python;
        let javascript_lang = LanguageType::JavaScript;
        
        assert!(matches!(rust_lang, LanguageType::Rust));
        assert!(matches!(python_lang, LanguageType::Python));
        assert!(matches!(javascript_lang, LanguageType::JavaScript));
    }

    #[test]
    fn test_model_config_code_completion() {
        let config = ModelConfig::code_completion();
        assert!(config.model_name.contains("code") || config.model_name.contains("completion"));
    }
}
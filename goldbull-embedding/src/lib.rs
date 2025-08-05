pub mod model;
pub mod training;
pub mod embeddings;

pub use model::GoldbullEmbedding;
pub use training::Trainer;
pub use embeddings::{EmbeddingRequest, EmbeddingResponse, SimilarityRequest, SimilarityResponse};

use anyhow::Result;
use goldbull_core::ModelConfig;
use candle_core::Device;

/// Create a new text embedding model with default configuration
pub fn new_embedding_model(device: Device) -> Result<GoldbullEmbedding> {
    let config = ModelConfig::embedding();
    GoldbullEmbedding::new(config, device)
}

/// Generate embeddings for text
pub async fn generate_embeddings(
    model: &GoldbullEmbedding,
    request: EmbeddingRequest,
) -> Result<EmbeddingResponse> {
    model.embed(request).await
}

/// Calculate similarity between texts
pub async fn calculate_similarity(
    model: &GoldbullEmbedding,
    request: SimilarityRequest,
) -> Result<SimilarityResponse> {
    model.similarity(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_creation() {
        let request = EmbeddingRequest {
            texts: vec!["Hello world".to_string(), "Natural language processing".to_string()],
            normalize: true,
            ..Default::default()
        };
        
        assert_eq!(request.texts.len(), 2);
        assert_eq!(request.texts[0], "Hello world");
        assert_eq!(request.texts[1], "Natural language processing");
        assert!(request.normalize);
    }
    
    #[test]
    fn test_similarity_request_creation() {
        let request = SimilarityRequest {
            text1: "Machine learning is amazing".to_string(),
            text2: "AI and ML are fascinating".to_string(),
            metric: embeddings::SimilarityMetric::Cosine,
        };
        
        assert_eq!(request.text1, "Machine learning is amazing");
        assert_eq!(request.text2, "AI and ML are fascinating");
        assert!(matches!(request.metric, embeddings::SimilarityMetric::Cosine));
    }

    #[test]
    fn test_model_config_embedding() {
        let config = ModelConfig::embedding();
        assert!(config.model_name.contains("embedding") || config.model_name.contains("embed"));
    }
}
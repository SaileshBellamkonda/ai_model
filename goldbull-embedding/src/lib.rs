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

    #[tokio::test]
    async fn test_text_embedding() {
        let model = new_embedding_model(Device::Cpu).unwrap();
        
        let request = EmbeddingRequest {
            texts: vec!["Hello world".to_string(), "Natural language processing".to_string()],
            normalize: true,
            ..Default::default()
        };
        
        let response = generate_embeddings(&model, request).await.unwrap();
        assert_eq!(response.embeddings.len(), 2);
        assert!(!response.embeddings[0].is_empty());
    }
    
    #[tokio::test]
    async fn test_similarity_calculation() {
        let model = new_embedding_model(Device::Cpu).unwrap();
        
        let request = SimilarityRequest {
            text1: "Machine learning is amazing".to_string(),
            text2: "AI and ML are fascinating".to_string(),
            similarity_metric: embeddings::SimilarityMetric::Cosine,
        };
        
        let response = calculate_similarity(&model, request).await.unwrap();
        assert!(response.similarity >= 0.0 && response.similarity <= 1.0);
    }
}
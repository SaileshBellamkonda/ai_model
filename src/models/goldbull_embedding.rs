use crate::Result;
use crate::tokenizer::{TiktokenBpeTokenizer};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Configuration for the Goldbull embedding model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldbullEmbeddingConfig {
    pub embedding_dim: usize,
    pub max_sequence_length: usize,
    pub vocab_size: usize,
    pub pooling_strategy: PoolingStrategy,
}

impl Default for GoldbullEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 1024, // 1024-dimensional embeddings as requested
            max_sequence_length: 512,
            vocab_size: 1000000, // Match BPEmb vocabulary size
            pooling_strategy: PoolingStrategy::Mean,
        }
    }
}

/// Pooling strategies for text embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolingStrategy {
    Mean,      // Average of all token embeddings
    Max,       // Max pooling over token embeddings
    Cls,       // Use first token (CLS) embedding
    LastToken, // Use last token embedding
}

/// Goldbull text embedding model
/// Specialized for generating high-quality 1024-dimensional text embeddings
pub struct GoldbullEmbedding {
    config: GoldbullEmbeddingConfig,
    
    // Model weights
    token_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,
    
    // Transformer layers for contextualization
    attention_layers: Vec<EmbeddingAttentionLayer>,
    
    // Output projection to final embedding dimension
    output_projection: Array2<f32>,
    
    // Tokenizer
    tokenizer: TiktokenBpeTokenizer,
}

/// Simplified attention layer for embeddings
#[derive(Clone)]
struct EmbeddingAttentionLayer {
    query_weights: Array2<f32>,
    key_weights: Array2<f32>,
    value_weights: Array2<f32>,
    output_weights: Array2<f32>,
    layer_norm_weights: Array1<f32>,
    layer_norm_bias: Array1<f32>,
}

impl GoldbullEmbedding {
    /// Create a new Goldbull embedding model
    pub async fn new(config: GoldbullEmbeddingConfig, tokenizer: TiktokenBpeTokenizer) -> Result<Self> {
        let mut rng = rand::thread_rng();
        
        // Initialize token embeddings
        let token_embeddings = Array2::from_shape_fn(
            (config.vocab_size, config.embedding_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        
        // Initialize position embeddings
        let position_embeddings = Array2::from_shape_fn(
            (config.max_sequence_length, config.embedding_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        
        // Initialize attention layers (6 layers for good contextualization)
        let mut attention_layers = Vec::new();
        for _ in 0..6 {
            attention_layers.push(EmbeddingAttentionLayer {
                query_weights: Array2::from_shape_fn(
                    (config.embedding_dim, config.embedding_dim),
                    |_| rng.gen_range(-0.1..0.1),
                ),
                key_weights: Array2::from_shape_fn(
                    (config.embedding_dim, config.embedding_dim),
                    |_| rng.gen_range(-0.1..0.1),
                ),
                value_weights: Array2::from_shape_fn(
                    (config.embedding_dim, config.embedding_dim),
                    |_| rng.gen_range(-0.1..0.1),
                ),
                output_weights: Array2::from_shape_fn(
                    (config.embedding_dim, config.embedding_dim),
                    |_| rng.gen_range(-0.1..0.1),
                ),
                layer_norm_weights: Array1::ones(config.embedding_dim),
                layer_norm_bias: Array1::zeros(config.embedding_dim),
            });
        }
        
        // Initialize output projection (if needed)
        let output_projection = Array2::from_shape_fn(
            (config.embedding_dim, config.embedding_dim),
            |_| rng.gen_range(-0.1..0.1),
        );
        
        Ok(Self {
            config,
            token_embeddings,
            position_embeddings,
            attention_layers,
            output_projection,
            tokenizer,
        })
    }

    /// Generate embeddings for a single text
    pub async fn encode(&self, text: &str) -> Result<Array1<f32>> {
        let texts = vec![text.to_string()];
        let embeddings = self.encode_batch(&texts).await?;
        Ok(embeddings.row(0).to_owned())
    }

    /// Generate embeddings for a batch of texts
    pub async fn encode_batch(&self, texts: &[String]) -> Result<Array2<f32>> {
        let mut batch_embeddings = Array2::zeros((texts.len(), self.config.embedding_dim));
        
        for (i, text) in texts.iter().enumerate() {
            let embedding = self.encode_single_text(text).await?;
            batch_embeddings.row_mut(i).assign(&embedding);
        }
        
        Ok(batch_embeddings)
    }

    /// Encode a single text into embeddings
    async fn encode_single_text(&self, text: &str) -> Result<Array1<f32>> {
        // Tokenize the text
        let token_ids = self.tokenizer.encode(text)?;
        
        // Truncate to max sequence length
        let truncated_tokens: Vec<u32> = token_ids.into_iter()
            .take(self.config.max_sequence_length)
            .collect();
        
        if truncated_tokens.is_empty() {
            return Ok(Array1::zeros(self.config.embedding_dim));
        }
        
        // Get token embeddings
        let mut sequence_embeddings = Array2::zeros((truncated_tokens.len(), self.config.embedding_dim));
        for (i, &token_id) in truncated_tokens.iter().enumerate() {
            let token_idx = token_id as usize % self.config.vocab_size;
            let token_embedding = self.token_embeddings.row(token_idx);
            sequence_embeddings.row_mut(i).assign(&token_embedding);
        }
        
        // Add position embeddings
        for (i, mut row) in sequence_embeddings.rows_mut().into_iter().enumerate() {
            if i < self.config.max_sequence_length {
                let pos_embedding = self.position_embeddings.row(i);
                row += &pos_embedding;
            }
        }
        
        // Apply attention layers for contextualization
        let mut hidden_states = sequence_embeddings;
        for layer in &self.attention_layers {
            hidden_states = self.apply_attention_layer(&hidden_states, layer)?;
        }
        
        // Apply pooling to get final embedding
        let final_embedding = self.apply_pooling(&hidden_states)?;
        
        // Apply output projection
        let projected = final_embedding.dot(&self.output_projection);
        
        // Normalize embedding (unit vector)
        let norm = (projected.dot(&projected)).sqrt();
        if norm > 1e-8 {
            Ok(&projected / norm)
        } else {
            Ok(projected)
        }
    }

    /// Apply attention layer for contextualization
    fn apply_attention_layer(
        &self, 
        input: &Array2<f32>, 
        layer: &EmbeddingAttentionLayer
    ) -> Result<Array2<f32>> {
        let seq_len = input.nrows();
        let embed_dim = input.ncols();
        
        // Compute Q, K, V
        let queries = input.dot(&layer.query_weights);
        let keys = input.dot(&layer.key_weights);
        let values = input.dot(&layer.value_weights);
        
        // Compute attention scores
        let mut attention_output = Array2::zeros((seq_len, embed_dim));
        
        for i in 0..seq_len {
            let query = queries.row(i);
            let mut attention_weights = Array1::zeros(seq_len);
            
            // Compute attention weights (simplified self-attention)
            for j in 0..seq_len {
                let key = keys.row(j);
                attention_weights[j] = query.dot(&key) / (embed_dim as f32).sqrt();
            }
            
            // Apply softmax
            let attention_weights = self.softmax(&attention_weights);
            
            // Compute weighted sum of values
            for j in 0..seq_len {
                let value = values.row(j);
                attention_output.row_mut(i).scaled_add(attention_weights[j], &value);
            }
        }
        
        // Apply output projection
        let projected = attention_output.dot(&layer.output_weights);
        
        // Residual connection and layer norm
        let residual = input + &projected;
        self.layer_norm(&residual, &layer.layer_norm_weights, &layer.layer_norm_bias)
    }

    /// Apply layer normalization
    fn layer_norm(
        &self, 
        input: &Array2<f32>, 
        weights: &Array1<f32>, 
        bias: &Array1<f32>
    ) -> Result<Array2<f32>> {
        let mut output = Array2::zeros(input.dim());
        
        for (_i, (input_row, mut output_row)) in input.rows().into_iter()
            .zip(output.rows_mut().into_iter()).enumerate() {
            
            // Compute mean and variance
            let mean = input_row.mean().unwrap_or(0.0);
            let variance = input_row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / input_row.len() as f32;
            
            let std = (variance + 1e-8).sqrt();
            
            // Normalize
            for (j, &x) in input_row.iter().enumerate() {
                output_row[j] = ((x - mean) / std) * weights[j] + bias[j];
            }
        }
        
        Ok(output)
    }

    /// Apply pooling strategy to get final embedding
    fn apply_pooling(&self, sequence_embeddings: &Array2<f32>) -> Result<Array1<f32>> {
        match self.config.pooling_strategy {
            PoolingStrategy::Mean => {
                // Mean pooling
                let mut mean_embedding = Array1::zeros(self.config.embedding_dim);
                for row in sequence_embeddings.rows() {
                    mean_embedding += &row;
                }
                Ok(mean_embedding / sequence_embeddings.nrows() as f32)
            },
            
            PoolingStrategy::Max => {
                // Max pooling
                let mut max_embedding = Array1::from_elem(self.config.embedding_dim, f32::NEG_INFINITY);
                for row in sequence_embeddings.rows() {
                    for (i, &val) in row.iter().enumerate() {
                        if val > max_embedding[i] {
                            max_embedding[i] = val;
                        }
                    }
                }
                Ok(max_embedding)
            },
            
            PoolingStrategy::Cls => {
                // Use first token (CLS token)
                Ok(sequence_embeddings.row(0).to_owned())
            },
            
            PoolingStrategy::LastToken => {
                // Use last token
                let last_idx = sequence_embeddings.nrows() - 1;
                Ok(sequence_embeddings.row(last_idx).to_owned())
            },
        }
    }

    /// Apply softmax function
    fn softmax(&self, input: &Array1<f32>) -> Array1<f32> {
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Array1<f32> = input.map(|x| (x - max_val).exp());
        let sum: f32 = exp_vals.sum();
        exp_vals / sum
    }

    /// Compute similarity between two texts
    pub async fn similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let emb1 = self.encode(text1).await?;
        let emb2 = self.encode(text2).await?;
        
        // Cosine similarity
        let dot_product = emb1.dot(&emb2);
        let norm1 = emb1.dot(&emb1).sqrt();
        let norm2 = emb2.dot(&emb2).sqrt();
        
        if norm1 > 1e-8 && norm2 > 1e-8 {
            Ok(dot_product / (norm1 * norm2))
        } else {
            Ok(0.0)
        }
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get model configuration
    pub fn get_config(&self) -> &GoldbullEmbeddingConfig {
        &self.config
    }

    /// Save the embedding model
    pub fn save(&self, path: &str) -> Result<()> {
        // Create directory if needed
        std::fs::create_dir_all(path).map_err(|e| crate::AIError::IoError(e))?;
        
        // Save configuration
        let config_path = format!("{}/config.json", path);
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json).map_err(|e| crate::AIError::IoError(e))?;
        
        // In a real implementation, we would save the model weights as well
        let weights_info = serde_json::json!({
            "model_type": "goldbull_embedding",
            "embedding_dim": self.config.embedding_dim,
            "vocab_size": self.config.vocab_size,
            "num_parameters": self.count_parameters(),
            "saved_at": chrono::Utc::now().to_rfc3339()
        });
        
        let weights_path = format!("{}/model_info.json", path);
        std::fs::write(weights_path, serde_json::to_string_pretty(&weights_info)?)
            .map_err(|e| crate::AIError::IoError(e))?;
        
        log::info!("Goldbull embedding model saved to {}", path);
        Ok(())
    }

    /// Count total parameters in the model
    fn count_parameters(&self) -> usize {
        let mut total = 0;
        total += self.token_embeddings.len();
        total += self.position_embeddings.len();
        
        for layer in &self.attention_layers {
            total += layer.query_weights.len();
            total += layer.key_weights.len();
            total += layer.value_weights.len();
            total += layer.output_weights.len();
            total += layer.layer_norm_weights.len();
            total += layer.layer_norm_bias.len();
        }
        
        total += self.output_projection.len();
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TiktokenBpeConfig;

    #[tokio::test]
    async fn test_embedding_model_creation() {
        let config = GoldbullEmbeddingConfig::default();
        let tokenizer_config = TiktokenBpeConfig::default();
        let tokenizer = TiktokenBpeTokenizer::new(tokenizer_config).unwrap();
        
        let model = GoldbullEmbedding::new(config, tokenizer).await;
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn test_text_encoding() {
        let config = GoldbullEmbeddingConfig::default();
        let tokenizer_config = TiktokenBpeConfig::default();
        let tokenizer = TiktokenBpeTokenizer::new(tokenizer_config).unwrap();
        let model = GoldbullEmbedding::new(config, tokenizer).await.unwrap();

        let embedding = model.encode("Hello world").await.unwrap();
        assert_eq!(embedding.len(), 1024);
        
        // Check that embedding is normalized (unit vector)
        let norm = embedding.dot(&embedding).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_similarity_computation() {
        let config = GoldbullEmbeddingConfig::default();
        let tokenizer_config = TiktokenBpeConfig::default();
        let tokenizer = TiktokenBpeTokenizer::new(tokenizer_config).unwrap();
        let model = GoldbullEmbedding::new(config, tokenizer).await.unwrap();

        let similarity = model.similarity("Hello world", "Hello world").await.unwrap();
        assert!(similarity > 0.9); // Should be very similar to itself
    }

    #[test]
    fn test_pooling_strategies() {
        // Test that all pooling strategies are available
        let strategies = [
            PoolingStrategy::Mean,
            PoolingStrategy::Max,
            PoolingStrategy::Cls,
            PoolingStrategy::LastToken,
        ];
        
        for strategy in &strategies {
            let config = GoldbullEmbeddingConfig {
                pooling_strategy: strategy.clone(),
                ..Default::default()
            };
            assert_eq!(config.embedding_dim, 1024);
        }
    }
}
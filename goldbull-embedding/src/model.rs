use anyhow::Result;
use candle_core::{Device, Tensor, Module};
use candle_nn::{embedding, linear, layer_norm, VarBuilder, VarMap};
use goldbull_core::ModelConfig;
use goldbull_tokenizer::{BpeTokenizer, Tokenizer};
use crate::embeddings::{EmbeddingRequest, EmbeddingResponse, SimilarityRequest, SimilarityResponse, SimilarityMetric};

/// Text embedding transformer model
/// Specialized for generating dense vector representations of text
pub struct GoldbullEmbedding {
    /// Model configuration parameters
    config: ModelConfig,
    /// Computational device (CPU/GPU)
    device: Device,
    /// Token embeddings layer
    token_embeddings: candle_nn::Embedding,
    /// Positional embeddings for sequence modeling
    position_embeddings: candle_nn::Embedding,
    /// Transformer encoder layers
    encoder_layers: Vec<TransformerLayer>,
    /// Layer normalization
    norm: candle_nn::LayerNorm,
    /// Pooling layer for sentence-level embeddings
    pooler: EmbeddingPooler,
    /// Output projection to final embedding dimension
    output_projection: candle_nn::Linear,
    /// Tokenizer for text processing
    tokenizer: BpeTokenizer,
    /// Variable map for weight management
    var_map: VarMap,
}

impl std::fmt::Debug for GoldbullEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoldbullEmbedding")
            .field("config", &self.config)
            .field("device", &self.device)
            .finish()
    }
}

impl Clone for GoldbullEmbedding {
    fn clone(&self) -> Self {
        Self::new(self.config.clone(), self.device.clone()).unwrap()
    }
}

/// Transformer encoder layer for text understanding
#[derive(Debug)]
pub struct TransformerLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
    dropout: f32,
}

/// Multi-head attention mechanism
#[derive(Debug)]
pub struct MultiHeadAttention {
    query_proj: candle_nn::Linear,
    key_proj: candle_nn::Linear,
    value_proj: candle_nn::Linear,
    output_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    dropout: f32,
}

/// Feed-forward network
#[derive(Debug)]
pub struct FeedForward {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
    dropout: f32,
}

/// Pooling strategy for sentence embeddings
#[derive(Debug)]
pub struct EmbeddingPooler {
    pooling_strategy: PoolingStrategy,
    projection: Option<candle_nn::Linear>,
}

#[derive(Debug, Clone)]
pub enum PoolingStrategy {
    Mean,
    Max,
    CLS,
    MeanMax,
}

impl GoldbullEmbedding {
    /// Create a new text embedding model
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, config.dtype, &device);
        
        // Token embeddings
        let token_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            var_builder.pp("token_embeddings"),
        )?;
        
        // Positional embeddings
        let position_embeddings = embedding(
            config.max_sequence_length,
            config.hidden_size,
            var_builder.pp("position_embeddings"),
        )?;
        
        // Transformer encoder layers
        let mut encoder_layers = Vec::new();
        for i in 0..config.num_layers {
            encoder_layers.push(TransformerLayer::new(&config, var_builder.pp(&format!("encoder_layer_{}", i)))?);
        }
        
        // Layer normalization
        let norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm"))?;
        
        // Pooling layer
        let pooler = EmbeddingPooler::new(&config, var_builder.pp("pooler"))?;
        
        // Output projection (to standard embedding dimension like 384 or 768)
        let embedding_dim = config.hidden_size; // Can be different from hidden_size
        let output_projection = linear(
            config.hidden_size,
            embedding_dim,
            var_builder.pp("output_projection"),
        )?;
        
        let tokenizer = BpeTokenizer::from_pretrained()?;
        
        Ok(Self {
            config,
            device,
            token_embeddings,
            position_embeddings,
            encoder_layers,
            norm,
            pooler,
            output_projection,
            tokenizer,
            var_map,
        })
    }
    
    /// Generate embeddings for input texts
    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let start_time = std::time::Instant::now();
        let mut embeddings = Vec::new();
        
        for text in &request.texts {
            let embedding = self.encode_text(text).await?;
            embeddings.push(embedding);
        }
        
        let processing_time = start_time.elapsed().as_millis();
        
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("processing_time_ms".to_string(), processing_time.to_string());
        metadata.insert("num_texts".to_string(), request.texts.len().to_string());
        metadata.insert("embedding_dimension".to_string(), self.config.hidden_size.to_string());
        metadata.insert("model_type".to_string(), "sentence_transformer".to_string());
        
        Ok(EmbeddingResponse {
            embeddings,
            metadata,
        })
    }
    
    /// Calculate similarity between two texts
    pub async fn similarity(&self, request: SimilarityRequest) -> Result<SimilarityResponse> {
        let start_time = std::time::Instant::now();
        
        // Generate embeddings for both texts
        let embedding1 = self.encode_text(&request.text1).await?;
        let embedding2 = self.encode_text(&request.text2).await?;
        
        // Calculate similarity based on requested metric
        let similarity = match request.metric {
            SimilarityMetric::Cosine => self.cosine_similarity(&embedding1, &embedding2)?,
            SimilarityMetric::Euclidean => self.euclidean_similarity(&embedding1, &embedding2)?,
            SimilarityMetric::DotProduct => self.dot_product_similarity(&embedding1, &embedding2)?,
        };
        
        let processing_time = start_time.elapsed().as_millis();
        
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("processing_time_ms".to_string(), processing_time.to_string());
        metadata.insert("similarity_metric".to_string(), format!("{:?}", request.metric));
        metadata.insert("embedding_dimension".to_string(), self.config.hidden_size.to_string());
        
        Ok(SimilarityResponse {
            similarity,
            metadata,
        })
    }
    
    /// Encode a single text into an embedding vector
    async fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // Tokenize the input text
        let tokens = self.tokenizer.encode(text)?;
        
        // Truncate if necessary
        let max_len = self.config.max_sequence_length - 2; // Reserve space for special tokens
        let tokens = if tokens.len() > max_len {
            tokens[..max_len].to_vec()
        } else {
            tokens
        };
        
        // Add special tokens (CLS and SEP)
        let mut input_tokens = vec![self.tokenizer.bos_token_id()];
        input_tokens.extend(tokens);
        input_tokens.push(self.tokenizer.eos_token_id());
        
        // Create tensors
        let input_tensor = Tensor::from_vec(
            input_tokens.clone(),
            (1, input_tokens.len()),
            &self.device,
        )?;
        
        // Create position IDs
        let position_ids: Vec<u32> = (0..input_tokens.len() as u32).collect();
        let position_tensor = Tensor::from_vec(
            position_ids,
            (1, input_tokens.len()),
            &self.device,
        )?;
        
        // Forward pass through the model
        let sequence_output = self.forward(&input_tensor, &position_tensor)?;
        
        // Pool the sequence output to get sentence embedding
        let pooled_output = self.pooler.forward(&sequence_output, &self.create_attention_mask(&input_tokens)?)?;
        
        // Apply output projection
        let final_embedding = self.output_projection.forward(&pooled_output)?;
        
        // Normalize the embedding (common practice for embedding models)
        let normalized_embedding = self.normalize_embedding(&final_embedding)?;
        
        // Convert to Vec<f32>
        let embedding_vec: Vec<f32> = normalized_embedding.to_vec1()?;
        
        Ok(embedding_vec)
    }
    
    /// Forward pass through the transformer encoder
    fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        // Get token embeddings
        let token_embeddings = self.token_embeddings.forward(input_ids)?;
        
        // Get position embeddings
        let position_embeddings = self.position_embeddings.forward(position_ids)?;
        
        // Combine embeddings
        let mut hidden_states = token_embeddings.add(&position_embeddings)?;
        
        // Pass through encoder layers
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        // Final layer normalization
        self.norm.forward(&hidden_states)
    }
    
    /// Create attention mask for padding tokens
    fn create_attention_mask(&self, tokens: &[u32]) -> Result<Tensor> {
        // For simplicity, assume no padding (all tokens are valid)
        let mask: Vec<f32> = vec![1.0; tokens.len()];
        Ok(Tensor::from_vec(mask, (1, tokens.len()), &self.device)?)
    }
    
    /// Normalize embedding vector to unit length
    fn normalize_embedding(&self, embedding: &Tensor) -> Result<Tensor> {
        let norm = embedding.sqr()?.sum_keepdim(1)?.sqrt()?;
        embedding.div(&norm)
    }
    
    /// Calculate cosine similarity between two embeddings
    fn cosine_similarity(&self, emb1: &[f32], emb2: &[f32]) -> Result<f64> {
        if emb1.len() != emb2.len() {
            return Err(anyhow::anyhow!("Embedding dimensions must match"));
        }
        
        let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok((dot_product / (norm1 * norm2)) as f64)
    }
    
    /// Calculate Euclidean similarity (inverse of distance)
    fn euclidean_similarity(&self, emb1: &[f32], emb2: &[f32]) -> Result<f64> {
        if emb1.len() != emb2.len() {
            return Err(anyhow::anyhow!("Embedding dimensions must match"));
        }
        
        let distance: f32 = emb1.iter()
            .zip(emb2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        // Convert distance to similarity (closer = more similar)
        Ok(1.0 / (1.0 + distance as f64))
    }
    
    /// Calculate dot product similarity
    fn dot_product_similarity(&self, emb1: &[f32], emb2: &[f32]) -> Result<f64> {
        if emb1.len() != emb2.len() {
            return Err(anyhow::anyhow!("Embedding dimensions must match"));
        }
        
        let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        Ok(dot_product as f64)
    }
    
    /// Get model tokenizer
    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }
    
    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TransformerLayer {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(config, var_builder.pp("self_attention"))?;
        let feed_forward = FeedForward::new(config, var_builder.pp("feed_forward"))?;
        let norm1 = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm1"))?;
        let norm2 = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm2"))?;
        
        Ok(Self {
            self_attention,
            feed_forward,
            norm1,
            norm2,
            dropout: 0.1,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection
        let normed_input = self.norm1.forward(input)?;
        let attn_output = self.self_attention.forward(&normed_input)?;
        let attn_residual = input.add(&attn_output)?;
        
        // Feed-forward with residual connection
        let normed_attn = self.norm2.forward(&attn_residual)?;
        let ff_output = self.feed_forward.forward(&normed_attn)?;
        let final_output = attn_residual.add(&ff_output)?;
        
        Ok(final_output)
    }
}

impl MultiHeadAttention {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads;
        
        let query_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("query"))?;
        let key_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("key"))?;
        let value_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("value"))?;
        let output_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("output"))?;
        
        Ok(Self {
            query_proj,
            key_proj,
            value_proj,
            output_proj,
            num_heads,
            head_dim,
            dropout: 0.1,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = input.dims3()?;
        
        // Project to Q, K, V
        let q = self.query_proj.forward(input)?;
        let k = self.key_proj.forward(input)?;
        let v = self.value_proj.forward(input)?;
        
        // Reshape for multi-head attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        
        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scaled_scores = scores.mul(&Tensor::from_slice(&[scale], (), input.device())?)?;
        
        let attention_weights = candle_nn::ops::softmax(&scaled_scores, 3)?;
        let attention_output = attention_weights.matmul(&v)?;
        
        // Reshape and concatenate heads
        let attention_output = attention_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, hidden_size))?;
        
        // Final output projection
        self.output_proj.forward(&attention_output)
    }
}

impl FeedForward {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let intermediate_size = config.hidden_size * 4; // Standard transformer ratio
        let linear1 = linear(config.hidden_size, intermediate_size, var_builder.pp("linear1"))?;
        let linear2 = linear(intermediate_size, config.hidden_size, var_builder.pp("linear2"))?;
        
        Ok(Self {
            linear1,
            linear2,
            dropout: 0.1,
        })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let activated = hidden.gelu()?; // GELU activation
        self.linear2.forward(&activated)
    }
}

impl EmbeddingPooler {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let pooling_strategy = PoolingStrategy::Mean; // Default to mean pooling
        let projection = Some(linear(
            config.hidden_size,
            config.hidden_size,
            var_builder.pp("projection"),
        )?);
        
        Ok(Self {
            pooling_strategy,
            projection,
        })
    }
    
    fn forward(&self, sequence_output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let pooled = match self.pooling_strategy {
            PoolingStrategy::Mean => self.mean_pooling(sequence_output, attention_mask)?,
            PoolingStrategy::Max => self.max_pooling(sequence_output)?,
            PoolingStrategy::CLS => self.cls_pooling(sequence_output)?,
            PoolingStrategy::MeanMax => {
                let mean_pooled = self.mean_pooling(sequence_output, attention_mask)?;
                let max_pooled = self.max_pooling(sequence_output)?;
                mean_pooled.add(&max_pooled)?.div(&Tensor::from_slice(&[2.0], (), sequence_output.device())?)?
            }
        };
        
        // Apply projection if available
        if let Some(ref projection) = self.projection {
            projection.forward(&pooled)
        } else {
            Ok(pooled)
        }
    }
    
    fn mean_pooling(&self, sequence_output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match sequence_output dimensions
        let mask_expanded = attention_mask.unsqueeze(2)?
            .expand(sequence_output.shape())?;
        
        // Apply mask and sum
        let masked_output = sequence_output.mul(&mask_expanded)?;
        let sum_embeddings = masked_output.sum(1)?;
        
        // Calculate the number of non-masked tokens
        let mask_sum = attention_mask.sum(1)?.unsqueeze(1)?;
        
        // Avoid division by zero
        let mask_sum_clamped = mask_sum.clamp_min(&Tensor::from_slice(&[1e-9], (), sequence_output.device())?)?;
        
        // Mean pooling
        sum_embeddings.div(&mask_sum_clamped)
    }
    
    fn max_pooling(&self, sequence_output: &Tensor) -> Result<Tensor> {
        sequence_output.max(1)
    }
    
    fn cls_pooling(&self, sequence_output: &Tensor) -> Result<Tensor> {
        // Take the first token (CLS token) representation
        sequence_output.i((.., 0, ..))
    }
}
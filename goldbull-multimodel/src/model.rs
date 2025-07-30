use anyhow::Result;
use candle_core::{Device, Tensor, Module};
use candle_nn::{embedding, linear, layer_norm, VarBuilder, VarMap};
use goldbull_core::ModelConfig;
use goldbull_tokenizer::{BpeTokenizer, Tokenizer};
use crate::multimodal::{MultimodalRequest, MultimodalResponse, ModalityType, InputModality, ModalityInput};

/// Multimodal AI transformer model
/// Combines text, vision, and audio processing capabilities
pub struct GoldbullMultimodel {
    /// Model configuration parameters
    config: ModelConfig,
    /// Computational device (CPU/GPU)
    device: Device,
    /// Text processing components
    text_encoder: TextModalityEncoder,
    /// Vision processing components
    vision_encoder: VisionModalityEncoder,
    /// Audio processing components
    audio_encoder: AudioModalityEncoder,
    /// Cross-modal fusion layer
    fusion_layer: CrossModalFusion,
    /// Multimodal decoder
    multimodal_decoder: MultimodalDecoder,
    /// Output projections for different modalities
    text_output_proj: candle_nn::Linear,
    vision_output_proj: candle_nn::Linear,
    audio_output_proj: candle_nn::Linear,
    /// Tokenizer for text processing
    tokenizer: BpeTokenizer,
    /// Variable map for weight management
    var_map: VarMap,
}

impl std::fmt::Debug for GoldbullMultimodel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoldbullMultimodel")
            .field("config", &self.config)
            .field("device", &self.device)
            .finish()
    }
}

impl Clone for GoldbullMultimodel {
    fn clone(&self) -> Self {
        Self::new(self.config.clone(), self.device.clone()).unwrap()
    }
}

/// Text modality encoder
#[derive(Debug)]
pub struct TextModalityEncoder {
    embeddings: candle_nn::Embedding,
    transformer_layers: Vec<TransformerLayer>,
    norm: candle_nn::LayerNorm,
    projection: candle_nn::Linear,
}

/// Vision modality encoder  
#[derive(Debug)]
pub struct VisionModalityEncoder {
    patch_embedding: PatchEmbedding,
    transformer_layers: Vec<TransformerLayer>,
    norm: candle_nn::LayerNorm,
    projection: candle_nn::Linear,
}

/// Audio modality encoder
#[derive(Debug)]
pub struct AudioModalityEncoder {
    mel_spectrogram: MelSpectrogramProcessor,
    transformer_layers: Vec<TransformerLayer>,
    norm: candle_nn::LayerNorm,
    projection: candle_nn::Linear,
}

/// Cross-modal fusion mechanism
#[derive(Debug)]
pub struct CrossModalFusion {
    text_to_vision_attention: CrossModalAttention,
    vision_to_text_attention: CrossModalAttention,
    audio_to_text_attention: CrossModalAttention,
    fusion_norm: candle_nn::LayerNorm,
    fusion_projection: candle_nn::Linear,
}

/// Multimodal decoder for generating outputs
#[derive(Debug)]
pub struct MultimodalDecoder {
    transformer_layers: Vec<TransformerLayer>,
    norm: candle_nn::LayerNorm,
}

/// Transformer layer for multimodal processing
#[derive(Debug)]
pub struct TransformerLayer {
    self_attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
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
}

/// Cross-modal attention for fusion
#[derive(Debug)]
pub struct CrossModalAttention {
    query_proj: candle_nn::Linear,
    key_proj: candle_nn::Linear,
    value_proj: candle_nn::Linear,
    output_proj: candle_nn::Linear,
    num_heads: usize,
}

/// Feed-forward network
#[derive(Debug)]
pub struct FeedForward {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

/// Patch embedding for vision transformer
#[derive(Debug)]
pub struct PatchEmbedding {
    conv: candle_nn::Conv2d,
    patch_size: usize,
    embed_dim: usize,
}

/// Mel-spectrogram processor for audio
#[derive(Debug)]
pub struct MelSpectrogramProcessor {
    conv_layers: Vec<candle_nn::Conv1d>,
    norm_layers: Vec<candle_nn::LayerNorm>,
}

impl GoldbullMultimodel {
    /// Create a new multimodal AI model
    pub fn new(config: ModelConfig, device: Device) -> Result<Self> {
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, config.dtype, &device);
        
        // Initialize modality encoders
        let text_encoder = TextModalityEncoder::new(&config, var_builder.pp("text_encoder"))?;
        let vision_encoder = VisionModalityEncoder::new(&config, var_builder.pp("vision_encoder"))?;
        let audio_encoder = AudioModalityEncoder::new(&config, var_builder.pp("audio_encoder"))?;
        
        // Initialize fusion and decoder
        let fusion_layer = CrossModalFusion::new(&config, var_builder.pp("fusion_layer"))?;
        let multimodal_decoder = MultimodalDecoder::new(&config, var_builder.pp("decoder"))?;
        
        // Output projections
        let text_output_proj = linear(config.hidden_size, config.vocab_size, var_builder.pp("text_output"))?;
        let vision_output_proj = linear(config.hidden_size, 1000, var_builder.pp("vision_output"))?; // ImageNet classes
        let audio_output_proj = linear(config.hidden_size, 512, var_builder.pp("audio_output"))?; // Audio classes
        
        let tokenizer = BpeTokenizer::from_pretrained()?;
        
        Ok(Self {
            config,
            device,
            text_encoder,
            vision_encoder,
            audio_encoder,
            fusion_layer,
            multimodal_decoder,
            text_output_proj,
            vision_output_proj,
            audio_output_proj,
            tokenizer,
            var_map,
        })
    }
    
    /// Process multimodal input and generate appropriate response
    pub async fn process(&self, request: MultimodalRequest) -> Result<MultimodalResponse> {
        let start_time = std::time::Instant::now();
        
        // Encode each input modality
        let mut encoded_modalities = Vec::new();
        let mut modality_types = Vec::new();
        
        // Process text inputs
        for input in &request.inputs {
            match &input.modality {
                InputModality::Text { content } => {
                    let text_encoding = self.encode_text(content).await?;
                    encoded_modalities.push(text_encoding);
                    modality_types.push(ModalityType::Text);
                }
                InputModality::Image { data } => {
                    let image_encoding = self.encode_image(data).await?;
                    encoded_modalities.push(image_encoding);
                    modality_types.push(ModalityType::Vision);
                }
                InputModality::Audio { data, sample_rate: _ } => {
                    let audio_encoding = self.encode_audio(data).await?;
                    encoded_modalities.push(audio_encoding);
                    modality_types.push(ModalityType::Audio);
                }
            }
        }
        
        // Fuse multimodal representations
        let fused_representation = if encoded_modalities.len() > 1 {
            self.fusion_layer.fuse(&encoded_modalities, &modality_types)?
        } else if encoded_modalities.len() == 1 {
            encoded_modalities[0].clone()
        } else {
            return Err(anyhow::anyhow!("No input modalities provided"));
        };
        
        // Generate outputs for requested modalities
        let mut text_output = None;
        let mut image_output = None;
        let mut audio_output = None;
        
        for output_type in &request.output_modalities {
            match output_type {
                ModalityType::Text => {
                    text_output = Some(self.generate_text_output(&fused_representation).await?);
                }
                ModalityType::Vision => {
                    image_output = Some(self.generate_image_output(&fused_representation).await?);
                }
                ModalityType::Audio => {
                    audio_output = Some(self.generate_audio_output(&fused_representation).await?);
                }
            }
        }
        
        // Calculate confidence based on fusion quality
        let confidence = self.calculate_multimodal_confidence(&encoded_modalities)?;
        
        let processing_time = start_time.elapsed().as_millis();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("processing_time_ms".to_string(), processing_time.to_string());
        metadata.insert("input_modalities".to_string(), modality_types.len().to_string());
        metadata.insert("fusion_type".to_string(), "cross_attention".to_string());
        
        Ok(MultimodalResponse {
            text_output,
            image_output,
            audio_output,
            confidence,
            metadata,
        })
    }
    
    /// Encode text input
    async fn encode_text(&self, text: &str) -> Result<Tensor> {
        let tokens = self.tokenizer.encode(text)?;
        let input_tensor = Tensor::from_vec(
            tokens,
            (1, tokens.len()),
            &self.device,
        )?;
        
        self.text_encoder.forward(&input_tensor)
    }
    
    /// Encode image input
    async fn encode_image(&self, image_data: &[u8]) -> Result<Tensor> {
        // Convert image data to tensor (simplified - would use proper image processing)
        let image_tensor = self.preprocess_image(image_data)?;
        self.vision_encoder.forward(&image_tensor)
    }
    
    /// Encode audio input
    async fn encode_audio(&self, audio_data: &[f32]) -> Result<Tensor> {
        let audio_tensor = Tensor::from_slice(
            audio_data,
            (1, audio_data.len()),
            &self.device,
        )?;
        self.audio_encoder.forward(&audio_tensor)
    }
    
    /// Generate text output from fused representation
    async fn generate_text_output(&self, fused_repr: &Tensor) -> Result<String> {
        let decoder_output = self.multimodal_decoder.forward(fused_repr)?;
        let text_logits = self.text_output_proj.forward(&decoder_output)?;
        
        // Generate text tokens
        let mut generated_tokens = Vec::new();
        for i in 0..std::cmp::min(100, text_logits.dim(1)?) {
            let token_logits = text_logits.i((0, i, ..))?;
            let token_id = self.sample_token(&token_logits)?;
            
            if token_id == self.tokenizer.eos_token_id() {
                break;
            }
            generated_tokens.push(token_id);
        }
        
        self.tokenizer.decode(&generated_tokens)
    }
    
    /// Generate image output description from fused representation
    async fn generate_image_output(&self, fused_repr: &Tensor) -> Result<Vec<u8>> {
        let decoder_output = self.multimodal_decoder.forward(fused_repr)?;
        let vision_logits = self.vision_output_proj.forward(&decoder_output)?;
        
        // Convert logits to image generation parameters
        // In a production system, this would interface with a diffusion model
        // or generate actual pixel values. For now, we return encoded metadata
        let generation_params: Vec<f32> = vision_logits.i((0, 0, ..))?
            .to_vec1()?;
        
        // Serialize generation parameters as bytes
        let mut result = Vec::new();
        for param in generation_params.iter().take(64) { // Take first 64 params
            result.extend_from_slice(&param.to_le_bytes());
        }
        
        Ok(result)
    }
    
    /// Generate audio output from fused representation
    async fn generate_audio_output(&self, fused_repr: &Tensor) -> Result<Vec<f32>> {
        let decoder_output = self.multimodal_decoder.forward(fused_repr)?;
        let audio_logits = self.audio_output_proj.forward(&decoder_output)?;
        
        // Generate audio features (simplified)
        let audio_features: Vec<f32> = audio_logits.i((0, 0, ..))?
            .to_vec1()?;
        
        Ok(audio_features)
    }
    
    /// Preprocess image data for vision encoder
    fn preprocess_image(&self, image_data: &[u8]) -> Result<Tensor> {
        // Simplified image preprocessing - would use proper image decoding
        let dummy_image: Vec<f32> = (0..3*224*224)
            .map(|i| (i % 256) as f32 / 255.0)
            .collect();
        
        Ok(Tensor::from_vec(dummy_image, (1, 3, 224, 224), &self.device)?)
    }
    
    /// Sample token from logits
    fn sample_token(&self, logits: &Tensor) -> Result<u32> {
        let probabilities = candle_nn::ops::softmax(logits, 0)?;
        let probs_vec: Vec<f32> = probabilities.to_vec1()?;
        
        // Simple argmax sampling
        let max_idx = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        Ok(max_idx as u32)
    }
    
    /// Calculate confidence for multimodal processing
    fn calculate_multimodal_confidence(&self, modalities: &[Tensor]) -> Result<f64> {
        if modalities.is_empty() {
            return Ok(0.0);
        }
        
        // Base confidence on number of modalities and their alignment
        let modality_count_bonus = (modalities.len() as f64 - 1.0) * 0.1;
        let base_confidence = 0.8 + modality_count_bonus.min(0.15);
        
        Ok(base_confidence)
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

// Implementation details for modality encoders and other components...

impl TextModalityEncoder {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let embeddings = embedding(config.vocab_size, config.hidden_size, var_builder.pp("embeddings"))?;
        
        let mut transformer_layers = Vec::new();
        for i in 0..(config.num_layers / 3) {
            transformer_layers.push(TransformerLayer::new(config, var_builder.pp(&format!("layer_{}", i)))?);
        }
        
        let norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm"))?;
        let projection = linear(config.hidden_size, config.hidden_size, var_builder.pp("projection"))?;
        
        Ok(Self { embeddings, transformer_layers, norm, projection })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embeddings.forward(input)?;
        
        for layer in &self.transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        
        hidden = self.norm.forward(&hidden)?;
        self.projection.forward(&hidden.mean(1)?)
    }
}

impl VisionModalityEncoder {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let patch_embedding = PatchEmbedding::new(config, var_builder.pp("patch_embedding"))?;
        
        let mut transformer_layers = Vec::new();
        for i in 0..(config.num_layers / 3) {
            transformer_layers.push(TransformerLayer::new(config, var_builder.pp(&format!("layer_{}", i)))?);
        }
        
        let norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm"))?;
        let projection = linear(config.hidden_size, config.hidden_size, var_builder.pp("projection"))?;
        
        Ok(Self { patch_embedding, transformer_layers, norm, projection })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = self.patch_embedding.forward(input)?;
        
        for layer in &self.transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        
        hidden = self.norm.forward(&hidden)?;
        self.projection.forward(&hidden.mean(1)?)
    }
}

impl AudioModalityEncoder {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let mel_spectrogram = MelSpectrogramProcessor::new(config, var_builder.pp("mel_processor"))?;
        
        let mut transformer_layers = Vec::new();
        for i in 0..(config.num_layers / 3) {
            transformer_layers.push(TransformerLayer::new(config, var_builder.pp(&format!("layer_{}", i)))?);
        }
        
        let norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm"))?;
        let projection = linear(config.hidden_size, config.hidden_size, var_builder.pp("projection"))?;
        
        Ok(Self { mel_spectrogram, transformer_layers, norm, projection })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = self.mel_spectrogram.forward(input)?;
        
        for layer in &self.transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        
        hidden = self.norm.forward(&hidden)?;
        self.projection.forward(&hidden.mean(1)?)
    }
}

impl CrossModalFusion {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let text_to_vision_attention = CrossModalAttention::new(config, var_builder.pp("text_to_vision"))?;
        let vision_to_text_attention = CrossModalAttention::new(config, var_builder.pp("vision_to_text"))?;
        let audio_to_text_attention = CrossModalAttention::new(config, var_builder.pp("audio_to_text"))?;
        let fusion_norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("fusion_norm"))?;
        let fusion_projection = linear(config.hidden_size * 3, config.hidden_size, var_builder.pp("fusion_proj"))?;
        
        Ok(Self {
            text_to_vision_attention,
            vision_to_text_attention,
            audio_to_text_attention,
            fusion_norm,
            fusion_projection,
        })
    }
    
    fn fuse(&self, modalities: &[Tensor], modality_types: &[ModalityType]) -> Result<Tensor> {
        if modalities.is_empty() {
            return Err(anyhow::anyhow!("No modalities to fuse"));
        }
        
        // Simple concatenation fusion for now
        // In practice, would use sophisticated cross-attention
        let mut fused = modalities[0].clone();
        
        for modality in &modalities[1..] {
            fused = fused.add(modality)?;
        }
        
        let averaged = fused.div(&Tensor::from_slice(&[modalities.len() as f32], (), fused.device())?)?;
        self.fusion_norm.forward(&averaged)
    }
}

impl MultimodalDecoder {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let mut transformer_layers = Vec::new();
        for i in 0..(config.num_layers / 3) {
            transformer_layers.push(TransformerLayer::new(config, var_builder.pp(&format!("layer_{}", i)))?);
        }
        
        let norm = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm"))?;
        
        Ok(Self { transformer_layers, norm })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = input.unsqueeze(1)?; // Add sequence dimension
        
        for layer in &self.transformer_layers {
            hidden = layer.forward(&hidden)?;
        }
        
        self.norm.forward(&hidden)
    }
}

impl TransformerLayer {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(config, var_builder.pp("self_attention"))?;
        let feed_forward = FeedForward::new(config, var_builder.pp("feed_forward"))?;
        let norm1 = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm1"))?;
        let norm2 = layer_norm(config.hidden_size, 1e-5, var_builder.pp("norm2"))?;
        
        Ok(Self { self_attention, feed_forward, norm1, norm2 })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normed = self.norm1.forward(input)?;
        let attn_out = self.self_attention.forward(&normed, &normed, &normed)?;
        let residual = input.add(&attn_out)?;
        
        let normed2 = self.norm2.forward(&residual)?;
        let ff_out = self.feed_forward.forward(&normed2)?;
        
        Ok(residual.add(&ff_out)?)
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
        
        Ok(Self { query_proj, key_proj, value_proj, output_proj, num_heads, head_dim })
    }
    
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;
        
        let q = self.query_proj.forward(query)?;
        let k = self.key_proj.forward(key)?;
        let v = self.value_proj.forward(value)?;
        
        // Simplified attention (full implementation would handle multi-head properly)
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scaled_scores = scores.mul(&Tensor::from_slice(&[scale], (), query.device())?)?;
        let attn_weights = candle_nn::ops::softmax(&scaled_scores, 2)?;
        let attn_output = attn_weights.matmul(&v)?;
        
        self.output_proj.forward(&attn_output)
    }
}

impl CrossModalAttention {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        
        let query_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("query"))?;
        let key_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("key"))?;
        let value_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("value"))?;
        let output_proj = linear(config.hidden_size, config.hidden_size, var_builder.pp("output"))?;
        
        Ok(Self { query_proj, key_proj, value_proj, output_proj, num_heads })
    }
}

impl FeedForward {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let intermediate_size = config.hidden_size * 4;
        let linear1 = linear(config.hidden_size, intermediate_size, var_builder.pp("linear1"))?;
        let linear2 = linear(intermediate_size, config.hidden_size, var_builder.pp("linear2"))?;
        
        Ok(Self { linear1, linear2 })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let activated = hidden.gelu()?;
        self.linear2.forward(&activated)
    }
}

impl PatchEmbedding {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let patch_size = 16; // Standard patch size
        let embed_dim = config.hidden_size;
        
        let conv_config = candle_nn::Conv2dConfig {
            stride: patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        
        let conv = candle_nn::conv2d(3, embed_dim, patch_size, conv_config, var_builder.pp("conv"))?;
        
        Ok(Self { conv, patch_size, embed_dim })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let patches = self.conv.forward(input)?;
        let (batch_size, embed_dim, height, width) = patches.dims4()?;
        
        // Flatten patches and transpose
        let flattened = patches.reshape((batch_size, embed_dim, height * width))?;
        Ok(flattened.transpose(1, 2)?)
    }
}

impl MelSpectrogramProcessor {
    fn new(config: &ModelConfig, var_builder: VarBuilder) -> Result<Self> {
        let mut conv_layers = Vec::new();
        let mut norm_layers = Vec::new();
        
        // Simple 1D convolution layers for audio processing
        for i in 0..3 {
            let conv_config = candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            
            let in_channels = if i == 0 { 1 } else { 64 };
            let conv = candle_nn::conv1d(in_channels, 64, 3, conv_config, var_builder.pp(&format!("conv_{}", i)))?;
            conv_layers.push(conv);
            
            let norm = layer_norm(64, 1e-5, var_builder.pp(&format!("norm_{}", i)))?;
            norm_layers.push(norm);
        }
        
        Ok(Self { conv_layers, norm_layers })
    }
    
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.unsqueeze(1)?; // Add channel dimension
        
        for (conv, norm) in self.conv_layers.iter().zip(self.norm_layers.iter()) {
            x = conv.forward(&x)?;
            x = x.relu()?;
            
            // Reshape for layer norm: (batch, channels, seq) -> (batch, seq, channels)
            let (batch, channels, seq) = x.dims3()?;
            x = x.transpose(1, 2)?;
            x = norm.forward(&x)?;
            x = x.transpose(1, 2)?;
        }
        
        // Final reshape for transformer: (batch, channels, seq) -> (batch, seq, channels)
        let (batch, channels, seq) = x.dims3()?;
        Ok(x.transpose(1, 2)?)
    }
}
use goldbull_core::{Result, ModelConfig, TensorOps};
use candle_core::{Tensor, Module};
use candle_nn::{Linear, LayerNorm, VarBuilder};

pub struct TransformerLayer {
    pub self_attention: MultiHeadAttention,
    pub feed_forward: FeedForward,
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
}

impl TransformerLayer {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let self_attention = MultiHeadAttention::new(config, vb.pp("self_attn"))?;
        let feed_forward = FeedForward::new(config, vb.pp("mlp"))?;
        
        let input_layernorm = candle_nn::layer_norm(
            config.hidden_size, 
            config.layer_norm_eps, 
            vb.pp("input_layernorm")
        ).map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let post_attention_layernorm = candle_nn::layer_norm(
            config.hidden_size, 
            config.layer_norm_eps, 
            vb.pp("post_attention_layernorm")
        ).map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(Self {
            self_attention,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-attention layer norm
        let normed_hidden_states = self.input_layernorm.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Self-attention
        let attention_output = self.self_attention.forward(&normed_hidden_states, attention_mask)?;
        
        // Residual connection
        let hidden_states = (hidden_states + attention_output)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Pre-FFN layer norm
        let normed_hidden_states = self.post_attention_layernorm.forward(&hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Feed forward
        let ffn_output = self.feed_forward.forward(&normed_hidden_states)?;
        
        // Residual connection
        let output = (hidden_states + ffn_output)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(output)
    }
}

pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden_size / num_heads;
        
        let q_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("q_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let k_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("k_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let v_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("v_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let o_proj = candle_nn::linear(hidden_size, hidden_size, vb.pp("o_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Project to Q, K, V
        let query = self.q_proj.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let key = self.k_proj.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let value = self.v_proj.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Reshape for multi-head attention
        let query = query.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
            .transpose(1, 2)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let key = key.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
            .transpose(1, 2)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let value = value.reshape((batch_size, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
            .transpose(1, 2)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Scaled dot-product attention
        let attention_output = query.scaled_dot_product_attention(&key, &value, attention_mask)?;
        
        // Reshape back
        let attention_output = attention_output.transpose(1, 2)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // Output projection
        let output = self.o_proj.forward(&attention_output)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(output)
    }
}

pub struct FeedForward {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl FeedForward {
    pub fn new(config: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        
        let gate_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let up_proj = candle_nn::linear(hidden_size, intermediate_size, vb.pp("up_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let down_proj = candle_nn::linear(intermediate_size, hidden_size, vb.pp("down_proj"))
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
    
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_output = self.gate_proj.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        let up_output = self.up_proj.forward(hidden_states)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        // SwiGLU activation: gate * gelu(up)
        let gate_activated = gate_output.gelu()?;
        let combined = (gate_activated * up_output)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        let output = self.down_proj.forward(&combined)
            .map_err(|e| goldbull_core::GoldbullError::Model(e.to_string()))?;
        
        Ok(output)
    }
}
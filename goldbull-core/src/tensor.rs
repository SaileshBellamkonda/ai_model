use crate::Result;
use candle_core::Tensor;

pub trait TensorOps {
    fn cpu_optimized_matmul(&self, other: &Tensor) -> Result<Tensor>;
    fn apply_rotary_embeddings(&self, position_ids: &Tensor) -> Result<Tensor>;
    fn layer_norm(&self, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Result<Tensor>;
    fn gelu(&self) -> Result<Tensor>;
    fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

impl TensorOps for Tensor {
    fn cpu_optimized_matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Optimized matrix multiplication for CPU execution
        // Using efficient BLAS operations when available
        self.matmul(other).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))
    }
    
    fn apply_rotary_embeddings(&self, _position_ids: &Tensor) -> Result<Tensor> {
        // Implement RoPE (Rotary Position Embeddings) for better position encoding
        // This is crucial for the 32K sequence length support
        let (_batch_size, _seq_len, _hidden_size) = self.dims3()
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        
        // For now, return self - full RoPE implementation would go here
        Ok(self.clone())
    }
    
    fn layer_norm(&self, weight: &Tensor, bias: Option<&Tensor>, eps: f64) -> Result<Tensor> {
        let mean = self.mean_keepdim(self.dims().len() - 1)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        let variance = self.broadcast_sub(&mean)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            .powf(2.0)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            .mean_keepdim(self.dims().len() - 1)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        
        let normalized = self.broadcast_sub(&mean)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            .broadcast_div(&(variance + eps).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?.sqrt().map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        
        let result = normalized.broadcast_mul(weight)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        
        if let Some(bias) = bias {
            result.broadcast_add(bias)
                .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))
        } else {
            Ok(result)
        }
    }
    
    fn gelu(&self) -> Result<Tensor> {
        // GELU activation function: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let x_cubed = self.powf(3.0)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        let inner = (self + (x_cubed * 0.044715).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            * (2.0 / std::f64::consts::PI).sqrt();
        let tanh_result = inner.map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?.tanh()
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        let one_plus_tanh = (tanh_result + 1.0)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        (self * 0.5).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            .broadcast_mul(&one_plus_tanh)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))
    }
    
    fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let scale = (self.dim(self.dims().len() - 1).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))? as f64).sqrt();
        
        let scores = (self.matmul(&key.transpose(key.dims().len() - 2, key.dims().len() - 1)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
            / scale).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
        
        let attention_weights = if let Some(mask) = attention_mask {
            let masked_scores = (scores + mask).map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?;
            candle_nn::ops::softmax(&masked_scores, masked_scores.dims().len() - 1)
                .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
        } else {
            candle_nn::ops::softmax(&scores, scores.dims().len() - 1)
                .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))?
        };
        
        attention_weights.matmul(value)
            .map_err(|e| crate::GoldbullError::Tensor(e.to_string()))
    }
}
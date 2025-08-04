pub mod model;
pub mod tensor;
pub mod error;
pub mod config;
pub mod utils;

pub use error::{GoldbullError, Result};
pub use config::ModelConfig;
pub use model::{Model, ModelTrait};
pub use tensor::TensorOps;

#[cfg(test)]
mod tests {
    #[test]
    fn test_clamp_equivalence() {
        // Test that clamp produces the same results as the original pattern
        let test_values = [0, 1, 16, 32, 50, 100];
        
        for &value in &test_values {
            let original_result = std::cmp::max(1, std::cmp::min(value, 32));
            let clamp_result = value.clamp(1, 32);
            assert_eq!(original_result, clamp_result, "Clamp mismatch for value {}", value);
        }
    }
}
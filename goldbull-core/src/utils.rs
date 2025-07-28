use std::path::Path;
use crate::Result;

pub fn ensure_dir_exists(path: &str) -> Result<()> {
    let dir = Path::new(path);
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub fn get_available_memory() -> usize {
    // Get available system memory (simplified)
    // In a real implementation, you'd use system APIs
    8 * 1024 * 1024 * 1024 // Assume 8GB available
}

pub fn is_cpu_optimized() -> bool {
    // Check if running on optimized CPU
    true // For this implementation, assume CPU optimization
}

pub fn get_optimal_batch_size(model_size: usize, sequence_length: usize) -> usize {
    // Calculate optimal batch size based on model size and sequence length
    // to stay within memory constraints
    const TARGET_MEMORY: usize = 1024 * 1024 * 1024; // 1GB target
    let per_sample_memory = model_size * sequence_length * 4; // Assume f32
    std::cmp::max(1, TARGET_MEMORY / per_sample_memory)
}
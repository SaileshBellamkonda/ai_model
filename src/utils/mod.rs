use crate::Result;
use std::time::{Duration, Instant};

/// Performance measurement utilities
pub struct PerformanceTimer {
    start_time: Instant,
    name: String,
}

impl PerformanceTimer {
    pub fn new(name: &str) -> Self {
        log::debug!("Starting timer: {}", name);
        Self {
            start_time: Instant::now(),
            name: name.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_millis() as f64
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed_ms();
        log::debug!("Timer '{}' completed in {:.2}ms", self.name, elapsed);
    }
}

/// Text processing utilities
pub fn tokenize_text(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .collect()
}

pub fn normalize_text(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Calculate text similarity (simple implementation)
pub fn text_similarity(text1: &str, text2: &str) -> f32 {
    let tokens1 = tokenize_text(text1);
    let tokens2 = tokenize_text(text2);
    
    if tokens1.is_empty() && tokens2.is_empty() {
        return 1.0;
    }
    
    if tokens1.is_empty() || tokens2.is_empty() {
        return 0.0;
    }
    
    let set1: std::collections::HashSet<_> = tokens1.iter().collect();
    let set2: std::collections::HashSet<_> = tokens2.iter().collect();
    
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    
    intersection as f32 / union as f32
}

/// Memory size utilities
pub fn bytes_to_mb(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn mb_to_bytes(mb: f64) -> usize {
    (mb * 1024.0 * 1024.0) as usize
}

/// Estimate memory usage of a string
pub fn estimate_string_memory(s: &str) -> usize {
    s.len() * std::mem::size_of::<u8>() + std::mem::size_of::<String>()
}

/// Configuration loading utilities
pub fn load_config_from_env() -> crate::core::ModelConfig {
    let mut config = crate::core::ModelConfig::default();
    
    if let Ok(max_memory) = std::env::var("AI_MODEL_MAX_MEMORY_MB") {
        if let Ok(value) = max_memory.parse::<usize>() {
            config.max_memory_mb = value;
        }
    }
    
    if let Ok(threads) = std::env::var("AI_MODEL_CPU_THREADS") {
        if let Ok(value) = threads.parse::<usize>() {
            config.cpu_threads = value;
        }
    }
    
    if let Ok(hidden_size) = std::env::var("AI_MODEL_HIDDEN_SIZE") {
        if let Ok(value) = hidden_size.parse::<usize>() {
            config.hidden_size = value;
        }
    }
    
    if let Ok(num_layers) = std::env::var("AI_MODEL_NUM_LAYERS") {
        if let Ok(value) = num_layers.parse::<usize>() {
            config.num_layers = value;
        }
    }
    
    config
}

/// Logging setup utility
pub fn setup_logging() -> Result<()> {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();
    
    log::info!("AI Model logging initialized");
    Ok(())
}

/// System information utilities
pub fn get_system_info() -> SystemInfo {
    SystemInfo {
        cpu_count: num_cpus::get(),
        available_memory_mb: get_available_memory_mb(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SystemInfo {
    pub cpu_count: usize,
    pub available_memory_mb: usize,
    pub os: String,
    pub arch: String,
}

fn get_available_memory_mb() -> usize {
    // This is a simplified implementation
    // In practice, you'd use platform-specific APIs
    match std::env::consts::OS {
        "linux" => {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb / 1024; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        },
        _ => {
            // Default assumption for other platforms
            return 8192; // 8GB default
        }
    }
    
    4096 // 4GB fallback
}

/// Validation utilities
pub fn validate_input_length(input: &str, max_length: usize) -> Result<()> {
    if input.len() > max_length {
        return Err(crate::AIError::InferenceError(
            format!("Input length {} exceeds maximum allowed length {}", input.len(), max_length)
        ));
    }
    Ok(())
}

pub fn validate_temperature(temperature: f32) -> Result<()> {
    if temperature < 0.0 || temperature > 2.0 {
        return Err(crate::AIError::InferenceError(
            format!("Temperature {} must be between 0.0 and 2.0", temperature)
        ));
    }
    Ok(())
}

/// URL validation for external API calls
pub fn validate_url(url: &str) -> Result<()> {
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(crate::AIError::ApiError(
            "URL must start with http:// or https://".to_string()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_similarity() {
        assert!((text_similarity("hello world", "hello world") - 1.0).abs() < f32::EPSILON);
        assert!((text_similarity("hello", "world") - 0.0).abs() < f32::EPSILON);
        assert!(text_similarity("hello world", "hello") > 0.0);
    }

    #[test]
    fn test_memory_conversion() {
        assert_eq!(mb_to_bytes(1.0), 1024 * 1024);
        assert!((bytes_to_mb(1024 * 1024) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_input_validation() {
        assert!(validate_input_length("hello", 10).is_ok());
        assert!(validate_input_length("hello world", 5).is_err());
        
        assert!(validate_temperature(0.5).is_ok());
        assert!(validate_temperature(-0.1).is_err());
        assert!(validate_temperature(2.1).is_err());
    }
}
use std::path::Path;
use crate::Result;

pub fn ensure_dir_exists(path: &str) -> Result<()> {
    let dir = Path::new(path);
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

/// System resource information for optimization
#[derive(Debug, Clone)]
pub struct SystemResources {
    pub total_memory_bytes: usize,
    pub available_memory_bytes: usize,
    pub cpu_cores: usize,
    pub cache_size_bytes: Option<usize>,
    pub is_low_memory_system: bool,
}

impl SystemResources {
    pub fn detect() -> Self {
        let available_memory = get_available_memory();
        let total_memory = get_total_memory();
        let cpu_cores = num_cpus::get();
        let cache_size = get_cpu_cache_size();
        
        Self {
            total_memory_bytes: total_memory,
            available_memory_bytes: available_memory,
            cpu_cores,
            cache_size_bytes: cache_size,
            is_low_memory_system: total_memory < 2 * 1024 * 1024 * 1024, // < 2GB
        }
    }
    
    /// Recommend model configuration based on system resources
    pub fn recommend_model_config(&self) -> crate::config::ModelConfig {
        if self.is_low_memory_system || self.available_memory_bytes < 512 * 1024 * 1024 {
            // Very low memory system
            crate::config::ModelConfig::lightweight()
        } else if self.available_memory_bytes < 1024 * 1024 * 1024 {
            // Low memory system
            let mut config = crate::config::ModelConfig::default();
            config.memory_optimization.max_memory_mb = 256;
            config.quantization.weight_dtype = crate::config::WeightDType::I8;
            config
        } else if self.available_memory_bytes < 2048 * 1024 * 1024 {
            // Medium memory system
            let mut config = crate::config::ModelConfig::default();
            config.memory_optimization.max_memory_mb = 512;
            config
        } else {
            // High memory system
            let mut config = crate::config::ModelConfig::default();
            config.memory_optimization.max_memory_mb = 1024;
            config.quantization.weight_dtype = crate::config::WeightDType::F16;
            config
        }
    }
}

/// Get total system memory
pub fn get_total_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(mem_str) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_str.parse::<usize>() {
                            return mem_kb * 1024;
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .args(&["-n", "hw.memsize"])
            .output()
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(total_memory) = mem_str.trim().parse::<usize>() {
                    return total_memory;
                }
            }
        }
    }
    
    // Platform-specific implementations with fallbacks
    #[cfg(target_os = "windows")]
    {
        // Windows implementation would go here
        return 8 * 1024 * 1024 * 1024; // 8GB fallback
    }
    
    // Fallback for non-Windows platforms
    4 * 1024 * 1024 * 1024 // 4GB fallback
}

/// Get available system memory using platform-specific APIs
/// 
/// This function determines the actual available memory on the system
/// using platform-specific approaches:
/// 
/// - Linux: Reads /proc/meminfo to get MemAvailable
/// - macOS: Uses sysctl to get total memory, returns 75% as available estimate
/// - Windows: Uses GlobalMemoryStatusEx API to get actual available physical memory
/// - Fallback: Returns 2GB conservative estimate for unknown platforms
/// 
/// Returns: Available memory in bytes
pub fn get_available_memory() -> usize {
    // Get available system memory using platform-specific APIs
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/meminfo on Linux
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    if let Some(mem_str) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_str.parse::<usize>() {
                            return mem_kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        // Use sysctl on macOS
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .args(&["-n", "hw.memsize"])
            .output()
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(total_memory) = mem_str.trim().parse::<usize>() {
                    // Return 75% of total memory as available (conservative estimate)
                    return (total_memory as f64 * 0.75) as usize;
                }
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // Use Windows API to get actual memory information
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        
        let mut mem_status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        
        unsafe {
            if GlobalMemoryStatusEx(&mut mem_status).into() {
                // Return available physical memory in bytes
                // ullAvailPhys contains available physical memory
                return mem_status.ullAvailPhys as usize;
            }
        }
        
        // Fallback if Windows API call fails
        return 4 * 1024 * 1024 * 1024; // 4GB fallback
    }
    
    // Fallback for non-Windows platforms or if system calls fail
    2 * 1024 * 1024 * 1024 // 2GB conservative fallback
}

/// Get CPU cache size information
pub fn get_cpu_cache_size() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // Try to read L3 cache size from /sys/devices/system/cpu
        if let Ok(cache_info) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            let size_str = cache_info.trim();
            if let Some(num_str) = size_str.strip_suffix("K") {
                if let Ok(size_kb) = num_str.parse::<usize>() {
                    return Some(size_kb * 1024);
                }
            }
        }
        
        // Fallback: read from /proc/cpuinfo
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("cache size") {
                    if let Some(size_part) = line.split(':').nth(1) {
                        let size_str = size_part.trim();
                        if let Some(num_str) = size_str.strip_suffix(" KB") {
                            if let Ok(size_kb) = num_str.parse::<usize>() {
                                return Some(size_kb * 1024);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Default fallback cache size estimates
    None
}

pub fn is_cpu_optimized() -> bool {
    // Check if running on optimized CPU with vectorization support
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX2 support
        if is_x86_feature_detected!("avx2") {
            return true;
        }
        
        // Check for SSE4.1 support
        if is_x86_feature_detected!("sse4.1") {
            return true;
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is typically available on AArch64
        return true;
    }
    
    // Fallback
    false
}

pub fn get_optimal_batch_size(model_size: usize, sequence_length: usize) -> usize {
    let system_resources = SystemResources::detect();
    
    // Calculate based on available memory
    let available_memory = system_resources.available_memory_bytes;
    let safety_factor = 0.7; // Use 70% of available memory
    let usable_memory = (available_memory as f64 * safety_factor) as usize;
    
    // Estimate memory per sample (rough calculation)
    // Model weights + activations + gradients
    let bytes_per_token = 4; // F32
    let memory_per_sample = model_size + (sequence_length * 512 * bytes_per_token); // Hidden size estimate
    
    if memory_per_sample == 0 {
        return 1;
    }
    
    let max_batch_size = usable_memory / memory_per_sample;
    
    // Consider CPU cores for parallel processing
    let cpu_optimal = std::cmp::min(system_resources.cpu_cores * 2, 16);
    
    std::cmp::max(1, std::cmp::min(max_batch_size, cpu_optimal))
}

/// Get optimal thread count for CPU operations
pub fn get_optimal_thread_count() -> usize {
    let cpu_cores = num_cpus::get();
    
    // For CPU-bound tasks, use fewer threads than cores to avoid contention
    if cpu_cores <= 4 {
        cpu_cores
    } else if cpu_cores <= 8 {
        cpu_cores - 1
    } else {
        // For many-core systems, use 75% of cores
        std::cmp::max(4, (cpu_cores as f64 * 0.75) as usize)
    }
}

/// Check if system is suitable for AI model inference
pub fn validate_system_requirements() -> Result<SystemResources> {
    let resources = SystemResources::detect();
    
    // Minimum requirements: 256MB available memory, 1 CPU core
    if resources.available_memory_bytes < 256 * 1024 * 1024 {
        return Err(crate::GoldbullError::Config(
            "Insufficient memory: minimum 256MB required".to_string()
        ));
    }
    
    if resources.cpu_cores < 1 {
        return Err(crate::GoldbullError::Config(
            "No CPU cores detected".to_string()
        ));
    }
    
    Ok(resources)
}
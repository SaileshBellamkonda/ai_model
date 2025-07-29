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
        // Use Windows API through std (simplified approach)
        // In production, you'd use the Windows API directly
        // For now, return a conservative default
        return 4 * 1024 * 1024 * 1024; // 4GB default for Windows
    }
    
    // Fallback for unsupported platforms or if system calls fail
    2 * 1024 * 1024 * 1024 // 2GB conservative fallback
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
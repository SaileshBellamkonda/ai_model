use crate::{AIError, Result};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Memory manager to ensure the model stays within the specified memory limits
pub struct MemoryManager {
    max_memory_mb: usize,
    current_usage_bytes: AtomicUsize,
    peak_usage_bytes: AtomicUsize,
}

impl MemoryManager {
    /// Create a new memory manager with the specified limit
    pub fn new(max_memory_mb: usize) -> Self {
        Self {
            max_memory_mb,
            current_usage_bytes: AtomicUsize::new(0),
            peak_usage_bytes: AtomicUsize::new(0),
        }
    }
    
    /// Check if we're within memory limits
    pub fn check_memory_limit(&self) -> Result<()> {
        let current_mb = self.current_usage_mb();
        if current_mb > self.max_memory_mb as f64 {
            return Err(AIError::MemoryLimitExceeded {
                current: current_mb as usize,
                limit: self.max_memory_mb,
            });
        }
        Ok(())
    }
    
    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
    
    /// Get peak memory usage in MB
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage_bytes.load(Ordering::Relaxed) as f64 / (1024.0 * 1024.0)
    }
    
    /// Track memory allocation
    pub fn allocate(&self, bytes: usize) -> Result<()> {
        let new_usage = self.current_usage_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        
        // Update peak usage
        let current_peak = self.peak_usage_bytes.load(Ordering::Relaxed);
        if new_usage > current_peak {
            self.peak_usage_bytes.store(new_usage, Ordering::Relaxed);
        }
        
        // Check if we exceeded the limit
        let current_mb = new_usage as f64 / (1024.0 * 1024.0);
        if current_mb > self.max_memory_mb as f64 {
            // Rollback the allocation
            self.current_usage_bytes.fetch_sub(bytes, Ordering::Relaxed);
            return Err(AIError::MemoryLimitExceeded {
                current: current_mb as usize,
                limit: self.max_memory_mb,
            });
        }
        
        Ok(())
    }
    
    /// Track memory deallocation
    pub fn deallocate(&self, bytes: usize) {
        let current = self.current_usage_bytes.load(Ordering::Relaxed);
        let new_usage = current.saturating_sub(bytes);
        self.current_usage_bytes.store(new_usage, Ordering::Relaxed);
    }
    
    /// Reset memory tracking
    pub fn reset(&self) {
        self.current_usage_bytes.store(0, Ordering::Relaxed);
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage_mb: self.current_usage_mb(),
            peak_usage_mb: self.peak_usage_mb(),
            limit_mb: self.max_memory_mb as f64,
            usage_percentage: (self.current_usage_mb() / self.max_memory_mb as f64) * 100.0,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryStats {
    pub current_usage_mb: f64,
    pub peak_usage_mb: f64,
    pub limit_mb: f64,
    pub usage_percentage: f64,
}

/// RAII wrapper for automatic memory tracking
pub struct MemoryAllocation<'a> {
    memory_manager: &'a MemoryManager,
    size_bytes: usize,
}

impl<'a> MemoryAllocation<'a> {
    /// Create a new memory allocation tracker
    pub fn new(memory_manager: &'a MemoryManager, size_bytes: usize) -> Result<Self> {
        memory_manager.allocate(size_bytes)?;
        Ok(Self {
            memory_manager,
            size_bytes,
        })
    }
}

impl<'a> Drop for MemoryAllocation<'a> {
    fn drop(&mut self) {
        self.memory_manager.deallocate(self.size_bytes);
    }
}
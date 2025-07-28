using System.ComponentModel.DataAnnotations;

namespace AIModel.Core.Models;

/// <summary>
/// Configuration settings for the AI model
/// </summary>
public class ModelConfiguration
{
    /// <summary>
    /// Path to the ONNX model file
    /// </summary>
    [Required]
    public string ModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Maximum sequence length for input tokens
    /// </summary>
    public int MaxSequenceLength { get; set; } = 512;

    /// <summary>
    /// Temperature for text generation (0.0 to 1.0)
    /// </summary>
    public float Temperature { get; set; } = 0.7f;

    /// <summary>
    /// Top-k sampling parameter
    /// </summary>
    public int TopK { get; set; } = 50;

    /// <summary>
    /// Top-p (nucleus) sampling parameter
    /// </summary>
    public float TopP { get; set; } = 0.9f;

    /// <summary>
    /// Maximum number of tokens to generate
    /// </summary>
    public int MaxTokens { get; set; } = 150;

    /// <summary>
    /// Memory optimization settings
    /// </summary>
    public MemorySettings Memory { get; set; } = new();

    /// <summary>
    /// Whether to enable CPU optimization
    /// </summary>
    public bool EnableCpuOptimization { get; set; } = true;

    /// <summary>
    /// Number of CPU threads to use (-1 for auto)
    /// </summary>
    public int CpuThreads { get; set; } = -1;
}

/// <summary>
/// Memory optimization settings
/// </summary>
public class MemorySettings
{
    /// <summary>
    /// Target memory limit in MB (default: 2048 MB = 2GB)
    /// </summary>
    public int MemoryLimitMB { get; set; } = 2048;

    /// <summary>
    /// Enable memory mapping for model files
    /// </summary>
    public bool EnableMemoryMapping { get; set; } = true;

    /// <summary>
    /// Enable gradient checkpointing to reduce memory usage
    /// </summary>
    public bool EnableGradientCheckpointing { get; set; } = true;
}
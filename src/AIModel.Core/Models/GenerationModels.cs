namespace AIModel.Core.Models;

/// <summary>
/// Request for text generation
/// </summary>
public class GenerationRequest
{
    /// <summary>
    /// Input prompt for generation
    /// </summary>
    public string Prompt { get; set; } = string.Empty;

    /// <summary>
    /// System message for context (optional)
    /// </summary>
    public string? SystemMessage { get; set; }

    /// <summary>
    /// Override temperature for this request
    /// </summary>
    public float? Temperature { get; set; }

    /// <summary>
    /// Override max tokens for this request
    /// </summary>
    public int? MaxTokens { get; set; }

    /// <summary>
    /// Whether to stop at the first complete response
    /// </summary>
    public bool StopAtComplete { get; set; } = true;

    /// <summary>
    /// Custom stop sequences
    /// </summary>
    public List<string> StopSequences { get; set; } = new();

    /// <summary>
    /// Whether to stream the response
    /// </summary>
    public bool Stream { get; set; } = false;
}

/// <summary>
/// Response from text generation
/// </summary>
public class GenerationResponse
{
    /// <summary>
    /// Generated text
    /// </summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>
    /// Tokens used in generation
    /// </summary>
    public int TokensUsed { get; set; }

    /// <summary>
    /// Time taken for generation in milliseconds
    /// </summary>
    public long GenerationTimeMs { get; set; }

    /// <summary>
    /// Whether the generation was completed (not truncated)
    /// </summary>
    public bool IsComplete { get; set; }

    /// <summary>
    /// Reason for stopping generation
    /// </summary>
    public string StopReason { get; set; } = string.Empty;

    /// <summary>
    /// Model usage statistics
    /// </summary>
    public ModelUsage Usage { get; set; } = new();
}

/// <summary>
/// Model usage statistics
/// </summary>
public class ModelUsage
{
    /// <summary>
    /// Input tokens processed
    /// </summary>
    public int InputTokens { get; set; }

    /// <summary>
    /// Output tokens generated
    /// </summary>
    public int OutputTokens { get; set; }

    /// <summary>
    /// Total tokens (input + output)
    /// </summary>
    public int TotalTokens => InputTokens + OutputTokens;

    /// <summary>
    /// Memory used in MB
    /// </summary>
    public long MemoryUsedMB { get; set; }

    /// <summary>
    /// CPU usage percentage during generation
    /// </summary>
    public double CpuUsagePercent { get; set; }
}
using AIModel.Core.Models;

namespace AIModel.Core.Interfaces;

/// <summary>
/// Core interface for AI model operations
/// </summary>
public interface IAIModel : IDisposable
{
    /// <summary>
    /// Initialize the model with configuration
    /// </summary>
    Task InitializeAsync(ModelConfiguration configuration, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generate text based on prompt
    /// </summary>
    Task<GenerationResponse> GenerateTextAsync(GenerationRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generate text with streaming support
    /// </summary>
    IAsyncEnumerable<string> GenerateTextStreamAsync(GenerationRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Generate code based on prompt
    /// </summary>
    Task<GenerationResponse> GenerateCodeAsync(GenerationRequest request, CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if the model is ready for inference
    /// </summary>
    bool IsReady { get; }

    /// <summary>
    /// Get current model usage statistics
    /// </summary>
    ModelUsage GetCurrentUsage();

    /// <summary>
    /// Get model configuration
    /// </summary>
    ModelConfiguration Configuration { get; }
}
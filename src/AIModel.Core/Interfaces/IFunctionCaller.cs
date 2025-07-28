using AIModel.Core.Models;
using System.Text.Json;

namespace AIModel.Core.Interfaces;

/// <summary>
/// Interface for function calling capabilities
/// </summary>
public interface IFunctionCaller
{
    /// <summary>
    /// Register a function that can be called by the AI model
    /// </summary>
    void RegisterFunction(FunctionDefinition function, Func<JsonElement, Task<JsonElement>> handler);

    /// <summary>
    /// Register a function with typed parameters and return value
    /// </summary>
    void RegisterFunction<TInput, TOutput>(
        string name, 
        string description, 
        Func<TInput, Task<TOutput>> handler,
        bool requiresConfirmation = false);

    /// <summary>
    /// Execute a function call
    /// </summary>
    Task<FunctionCallResponse> ExecuteFunctionAsync(FunctionCall call, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get all available functions
    /// </summary>
    IReadOnlyList<FunctionDefinition> GetAvailableFunctions();

    /// <summary>
    /// Check if a function is registered
    /// </summary>
    bool IsFunctionRegistered(string name);

    /// <summary>
    /// Remove a registered function
    /// </summary>
    bool RemoveFunction(string name);
}
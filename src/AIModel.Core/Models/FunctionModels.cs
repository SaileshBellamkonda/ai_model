using System.Text.Json;

namespace AIModel.Core.Models;

/// <summary>
/// Represents a function that can be called by the AI model
/// </summary>
public class FunctionDefinition
{
    /// <summary>
    /// Function name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Function description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Function parameters schema
    /// </summary>
    public JsonElement Parameters { get; set; }

    /// <summary>
    /// Whether this function requires confirmation before execution
    /// </summary>
    public bool RequiresConfirmation { get; set; } = false;
}

/// <summary>
/// Request to call a function
/// </summary>
public class FunctionCall
{
    /// <summary>
    /// Name of the function to call
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Arguments to pass to the function
    /// </summary>
    public JsonElement Arguments { get; set; }

    /// <summary>
    /// Unique identifier for this function call
    /// </summary>
    public string CallId { get; set; } = Guid.NewGuid().ToString();
}

/// <summary>
/// Response from a function call
/// </summary>
public class FunctionCallResponse
{
    /// <summary>
    /// Call ID that this response corresponds to
    /// </summary>
    public string CallId { get; set; } = string.Empty;

    /// <summary>
    /// Whether the function call was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Result of the function call
    /// </summary>
    public JsonElement? Result { get; set; }

    /// <summary>
    /// Error message if the call failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Time taken to execute the function in milliseconds
    /// </summary>
    public long ExecutionTimeMs { get; set; }
}

/// <summary>
/// Context for tool/function calling
/// </summary>
public class ToolContext
{
    /// <summary>
    /// Available functions
    /// </summary>
    public List<FunctionDefinition> AvailableFunctions { get; set; } = new();

    /// <summary>
    /// Previous function calls in this conversation
    /// </summary>
    public List<FunctionCallResponse> CallHistory { get; set; } = new();

    /// <summary>
    /// Whether to automatically execute functions without confirmation
    /// </summary>
    public bool AutoExecute { get; set; } = false;

    /// <summary>
    /// Maximum number of function calls allowed in a single request
    /// </summary>
    public int MaxFunctionCalls { get; set; } = 5;
}
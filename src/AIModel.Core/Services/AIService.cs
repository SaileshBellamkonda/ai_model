using AIModel.Core.Interfaces;
using AIModel.Core.Models;
using AIModel.Core.MCP;
using System.Runtime.CompilerServices;
using System.Text.Json;

namespace AIModel.Core.Services;

/// <summary>
/// Comprehensive AI service that combines model inference, function calling, and MCP support
/// </summary>
public class AIService : IDisposable
{
    private readonly IAIModel _model;
    private readonly IFunctionCaller _functionCaller;
    private readonly IMCPClient _mcpClient;
    private bool _isDisposed;

    public IAIModel Model => _model;
    public IFunctionCaller FunctionCaller => _functionCaller;
    public IMCPClient MCPClient => _mcpClient;

    public AIService(IAIModel? model = null, IFunctionCaller? functionCaller = null, IMCPClient? mcpClient = null)
    {
        _model = model ?? new CpuAIModel();
        _functionCaller = functionCaller ?? new FunctionCaller();
        _mcpClient = mcpClient ?? new MCPClient();
        
        RegisterBuiltInFunctions();
    }

    /// <summary>
    /// Initialize the AI service with configuration
    /// </summary>
    public async Task InitializeAsync(ModelConfiguration configuration, CancellationToken cancellationToken = default)
    {
        await _model.InitializeAsync(configuration, cancellationToken);
    }

    /// <summary>
    /// Connect to an MCP server
    /// </summary>
    public async Task ConnectToMCPServerAsync(string serverUrl, CancellationToken cancellationToken = default)
    {
        await _mcpClient.ConnectAsync(serverUrl, cancellationToken);
        
        // Load available tools from MCP server
        if (_mcpClient.IsConnected)
        {
            var mcpTools = await _mcpClient.GetAvailableToolsAsync(cancellationToken);
            foreach (var tool in mcpTools)
            {
                RegisterMCPTool(tool);
            }
        }
    }

    /// <summary>
    /// Generate text with optional function calling support
    /// </summary>
    public async Task<GenerationResponse> GenerateWithToolsAsync(
        GenerationRequest request, 
        ToolContext? toolContext = null,
        CancellationToken cancellationToken = default)
    {
        if (!_model.IsReady)
            throw new InvalidOperationException("Model is not ready");

        // Enhance prompt with available functions if tool context is provided
        if (toolContext != null && toolContext.AvailableFunctions.Any())
        {
            request = EnhancePromptWithFunctions(request, toolContext);
        }

        var response = await _model.GenerateTextAsync(request, cancellationToken);

        // Check if the response contains function calls
        if (toolContext != null && ContainsFunctionCalls(response.Text))
        {
            var functionCalls = ExtractFunctionCalls(response.Text);
            var functionResults = new List<FunctionCallResponse>();

            foreach (var call in functionCalls)
            {
                if (toolContext.MaxFunctionCalls > 0 && functionResults.Count >= toolContext.MaxFunctionCalls)
                    break;

                var result = await ExecuteFunctionCallAsync(call, toolContext.AutoExecute, cancellationToken);
                functionResults.Add(result);
                toolContext.CallHistory.Add(result);
            }

            // If functions were called, generate a follow-up response
            if (functionResults.Any())
            {
                var followUpPrompt = CreateFollowUpPrompt(request.Prompt, functionResults);
                var followUpRequest = new GenerationRequest
                {
                    Prompt = followUpPrompt,
                    SystemMessage = request.SystemMessage,
                    Temperature = request.Temperature,
                    MaxTokens = request.MaxTokens,
                    StopSequences = request.StopSequences
                };

                var followUpResponse = await _model.GenerateTextAsync(followUpRequest, cancellationToken);
                
                // Combine responses
                response.Text = $"{response.Text}\n\n{followUpResponse.Text}";
                response.TokensUsed += followUpResponse.TokensUsed;
                response.GenerationTimeMs += followUpResponse.GenerationTimeMs;
            }
        }

        return response;
    }

    /// <summary>
    /// Generate code with enhanced prompting for code generation
    /// </summary>
    public async Task<GenerationResponse> GenerateCodeAsync(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        return await _model.GenerateCodeAsync(request, cancellationToken);
    }

    /// <summary>
    /// Stream text generation with optional function calling
    /// </summary>
    public async IAsyncEnumerable<string> GenerateStreamAsync(
        GenerationRequest request, 
        ToolContext? toolContext = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (toolContext != null && toolContext.AvailableFunctions.Any())
        {
            request = EnhancePromptWithFunctions(request, toolContext);
        }

        await foreach (var chunk in _model.GenerateTextStreamAsync(request, cancellationToken))
        {
            if (cancellationToken.IsCancellationRequested)
                yield break;

            yield return chunk;
        }
    }

    private void RegisterBuiltInFunctions()
    {
        // Register built-in utility functions
        _functionCaller.RegisterFunction<string, DateTime>(
            "get_current_time",
            "Get the current date and time",
            async (input) =>
            {
                await Task.Delay(1); // Simulate async operation
                return DateTime.Now;
            });

        _functionCaller.RegisterFunction<MathOperation, double>(
            "calculate",
            "Perform basic mathematical operations",
            async (operation) =>
            {
                await Task.Delay(1);
                return operation.Operator switch
                {
                    "+" => operation.A + operation.B,
                    "-" => operation.A - operation.B,
                    "*" => operation.A * operation.B,
                    "/" => operation.B != 0 ? operation.A / operation.B : throw new DivideByZeroException(),
                    _ => throw new ArgumentException($"Unsupported operator: {operation.Operator}")
                };
            });

        _functionCaller.RegisterFunction<WebSearchRequest, string>(
            "web_search",
            "Search the web for information (simulated)",
            async (request) =>
            {
                await Task.Delay(100); // Simulate network delay
                return $"Search results for '{request.Query}': This is a simulated search result. " +
                       $"In a real implementation, this would perform an actual web search.";
            });
    }

    private void RegisterMCPTool(FunctionDefinition tool)
    {
        _functionCaller.RegisterFunction(tool, async (args) =>
        {
            var result = await _mcpClient.CallToolAsync(tool.Name, args);
            if (result.Success && result.Result.HasValue)
            {
                return result.Result.Value;
            }
            throw new InvalidOperationException(result.ErrorMessage ?? "MCP tool call failed");
        });
    }

    private GenerationRequest EnhancePromptWithFunctions(GenerationRequest request, ToolContext toolContext)
    {
        var functionsJson = JsonSerializer.Serialize(toolContext.AvailableFunctions.Select(f => new
        {
            name = f.Name,
            description = f.Description,
            parameters = f.Parameters
        }), new JsonSerializerOptions { WriteIndented = true });

        var enhancedPrompt = $@"{request.Prompt}

Available functions:
{functionsJson}

You can call functions by including them in your response in the format:
FUNCTION_CALL: {{""name"": ""function_name"", ""arguments"": {{""param"": ""value""}}}}

";

        return new GenerationRequest
        {
            Prompt = enhancedPrompt,
            SystemMessage = request.SystemMessage,
            Temperature = request.Temperature,
            MaxTokens = request.MaxTokens,
            StopSequences = request.StopSequences,
            Stream = request.Stream
        };
    }

    private bool ContainsFunctionCalls(string text)
    {
        return text.Contains("FUNCTION_CALL:");
    }

    private List<FunctionCall> ExtractFunctionCalls(string text)
    {
        var calls = new List<FunctionCall>();
        var lines = text.Split('\n');

        foreach (var line in lines)
        {
            if (line.Trim().StartsWith("FUNCTION_CALL:"))
            {
                try
                {
                    var jsonStr = line.Substring(line.IndexOf('{'));
                    using var doc = JsonDocument.Parse(jsonStr);
                    var name = doc.RootElement.GetProperty("name").GetString();
                    var arguments = doc.RootElement.GetProperty("arguments");

                    if (!string.IsNullOrEmpty(name))
                    {
                        calls.Add(new FunctionCall
                        {
                            Name = name,
                            Arguments = arguments,
                            CallId = Guid.NewGuid().ToString()
                        });
                    }
                }
                catch
                {
                    // Ignore malformed function calls
                }
            }
        }

        return calls;
    }

    private async Task<FunctionCallResponse> ExecuteFunctionCallAsync(FunctionCall call, bool autoExecute, CancellationToken cancellationToken)
    {
        if (!autoExecute && _functionCaller.GetAvailableFunctions().Any(f => f.Name == call.Name && f.RequiresConfirmation))
        {
            // In a real implementation, this would prompt the user for confirmation
            // For demo purposes, we'll auto-approve
        }

        return await _functionCaller.ExecuteFunctionAsync(call, cancellationToken);
    }

    private string CreateFollowUpPrompt(string originalPrompt, List<FunctionCallResponse> functionResults)
    {
        var resultsText = string.Join("\n", functionResults.Select(r =>
            $"Function {r.CallId}: {(r.Success ? $"Result: {r.Result}" : $"Error: {r.ErrorMessage}")}"));

        return $@"Original request: {originalPrompt}

Function execution results:
{resultsText}

Based on these results, provide a comprehensive response to the original request:";
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            _model?.Dispose();
            _mcpClient?.Dispose();
            _isDisposed = true;
        }
    }
}

// Supporting models for built-in functions
public class MathOperation
{
    public double A { get; set; }
    public double B { get; set; }
    public string Operator { get; set; } = string.Empty;
}

public class WebSearchRequest
{
    public string Query { get; set; } = string.Empty;
    public int MaxResults { get; set; } = 5;
}
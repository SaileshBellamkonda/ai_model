using AIModel.Core.Models;
using AIModel.Core.Services;
using System.Text.Json;

namespace AIModel.Tests;

public class AIModelTests
{
    [Fact]
    public async Task AIService_Initialize_ShouldSucceed()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration
        {
            ModelPath = "demo-model",
            Temperature = 0.7f,
            MaxTokens = 50,
            EnableCpuOptimization = true
        };

        // Act & Assert
        await aiService.InitializeAsync(config);
        Assert.True(aiService.Model.IsReady);
    }

    [Fact]
    public async Task GenerateText_WithValidPrompt_ShouldReturnResponse()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration { ModelPath = "demo-model" };
        await aiService.InitializeAsync(config);

        var request = new GenerationRequest
        {
            Prompt = "Hello world",
            MaxTokens = 50
        };

        // Act
        var response = await aiService.Model.GenerateTextAsync(request);

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
        Assert.True(response.TokensUsed > 0);
        Assert.True(response.GenerationTimeMs > 0);
    }

    [Fact]
    public async Task FunctionCaller_RegisterAndExecute_ShouldWork()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration { ModelPath = "demo-model" };
        await aiService.InitializeAsync(config);

        // Act - Test built-in function
        var call = new FunctionCall
        {
            Name = "get_current_time",
            Arguments = JsonSerializer.SerializeToElement(""),
            CallId = "test-call"
        };

        var response = await aiService.FunctionCaller.ExecuteFunctionAsync(call);

        // Assert
        Assert.True(response.Success);
        Assert.NotNull(response.Result);
        Assert.True(response.ExecutionTimeMs >= 0);
    }

    [Fact]
    public async Task FunctionCaller_MathOperation_ShouldCalculateCorrectly()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration { ModelPath = "demo-model" };
        await aiService.InitializeAsync(config);

        var mathOperation = new { A = 10.0, B = 5.0, Operator = "+" };
        var call = new FunctionCall
        {
            Name = "calculate",
            Arguments = JsonSerializer.SerializeToElement(mathOperation),
            CallId = "math-test"
        };

        // Act
        var response = await aiService.FunctionCaller.ExecuteFunctionAsync(call);

        // Assert
        Assert.True(response.Success);
        Assert.NotNull(response.Result);
        
        var result = response.Result.Value.GetDouble();
        Assert.Equal(15.0, result);
    }

    [Fact]
    public async Task GenerateCode_WithCodePrompt_ShouldReturnCodeResponse()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration { ModelPath = "demo-model" };
        await aiService.InitializeAsync(config);

        var request = new GenerationRequest
        {
            Prompt = "Create a hello world program",
            MaxTokens = 100
        };

        // Act
        var response = await aiService.GenerateCodeAsync(request);

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
        Assert.True(response.TokensUsed > 0);
    }

    [Fact]
    public void ModelConfiguration_DefaultValues_ShouldBeValid()
    {
        // Arrange & Act
        var config = new ModelConfiguration();

        // Assert
        Assert.Equal(512, config.MaxSequenceLength);
        Assert.Equal(0.7f, config.Temperature);
        Assert.Equal(50, config.TopK);
        Assert.Equal(0.9f, config.TopP);
        Assert.Equal(150, config.MaxTokens);
        Assert.True(config.EnableCpuOptimization);
        Assert.Equal(-1, config.CpuThreads);
        Assert.NotNull(config.Memory);
        Assert.Equal(2048, config.Memory.MemoryLimitMB);
    }

    [Fact]
    public async Task GenerateWithTools_WithToolContext_ShouldHandleFunctions()
    {
        // Arrange
        using var aiService = new AIService();
        var config = new ModelConfiguration { ModelPath = "demo-model" };
        await aiService.InitializeAsync(config);

        var toolContext = new ToolContext
        {
            AvailableFunctions = aiService.FunctionCaller.GetAvailableFunctions().ToList(),
            AutoExecute = true,
            MaxFunctionCalls = 2
        };

        var request = new GenerationRequest
        {
            Prompt = "What time is it?",
            MaxTokens = 100
        };

        // Act
        var response = await aiService.GenerateWithToolsAsync(request, toolContext);

        // Assert
        Assert.NotNull(response);
        Assert.NotEmpty(response.Text);
        Assert.True(response.TokensUsed > 0);
    }

    [Fact]
    public void FunctionCaller_GetAvailableFunctions_ShouldReturnBuiltInFunctions()
    {
        // Arrange
        using var aiService = new AIService();

        // Act
        var functions = aiService.FunctionCaller.GetAvailableFunctions();

        // Assert
        Assert.NotEmpty(functions);
        Assert.Contains(functions, f => f.Name == "get_current_time");
        Assert.Contains(functions, f => f.Name == "calculate");
        Assert.Contains(functions, f => f.Name == "web_search");
    }

    [Fact]
    public void ModelUsage_TotalTokens_ShouldCalculateCorrectly()
    {
        // Arrange
        var usage = new ModelUsage
        {
            InputTokens = 10,
            OutputTokens = 15
        };

        // Act & Assert
        Assert.Equal(25, usage.TotalTokens);
    }
}
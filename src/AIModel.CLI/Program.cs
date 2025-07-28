using AIModel.Core.Models;
using AIModel.Core.Services;
using System.CommandLine;
using System.Text.Json;

namespace AIModel.CLI;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("AI Model - CPU-optimized AI with function calling and MCP support");

        // Chat command
        var chatCommand = new Command("chat", "Interactive chat mode");
        var modelPathOption = new Option<string>("--model", "Path to ONNX model file") { IsRequired = false };
        var temperatureOption = new Option<float>("--temperature", () => 0.7f, "Temperature for generation (0.0-1.0)");
        var maxTokensOption = new Option<int>("--max-tokens", () => 150, "Maximum tokens to generate");
        var mcpServerOption = new Option<string>("--mcp-server", "MCP server URL to connect to");

        chatCommand.AddOption(modelPathOption);
        chatCommand.AddOption(temperatureOption);
        chatCommand.AddOption(maxTokensOption);
        chatCommand.AddOption(mcpServerOption);

        chatCommand.SetHandler(async (modelPath, temperature, maxTokens, mcpServer) =>
        {
            await RunChatMode(modelPath, temperature, maxTokens, mcpServer);
        }, modelPathOption, temperatureOption, maxTokensOption, mcpServerOption);

        // Generate command
        var generateCommand = new Command("generate", "Generate text from a prompt");
        var promptOption = new Option<string>("--prompt", "Text prompt for generation") { IsRequired = true };
        var codeOption = new Option<bool>("--code", "Generate code instead of text");

        generateCommand.AddOption(promptOption);
        generateCommand.AddOption(modelPathOption);
        generateCommand.AddOption(temperatureOption);
        generateCommand.AddOption(maxTokensOption);
        generateCommand.AddOption(codeOption);

        generateCommand.SetHandler(async (prompt, modelPath, temperature, maxTokens, isCode) =>
        {
            await RunGeneration(prompt, modelPath, temperature, maxTokens, isCode);
        }, promptOption, modelPathOption, temperatureOption, maxTokensOption, codeOption);

        // Function test command
        var testCommand = new Command("test", "Test function calling capabilities");
        testCommand.SetHandler(async () =>
        {
            await RunFunctionTest();
        });

        rootCommand.AddCommand(chatCommand);
        rootCommand.AddCommand(generateCommand);
        rootCommand.AddCommand(testCommand);

        return await rootCommand.InvokeAsync(args);
    }

    static async Task RunChatMode(string? modelPath, float temperature, int maxTokens, string? mcpServer)
    {
        Console.WriteLine("🤖 AI Model Chat Mode");
        Console.WriteLine("Type 'exit' to quit, 'help' for commands\n");

        using var aiService = new AIService();

        try
        {
            // Initialize model
            var config = new ModelConfiguration
            {
                ModelPath = modelPath ?? "demo-model", // Demo mode
                Temperature = temperature,
                MaxTokens = maxTokens,
                EnableCpuOptimization = true,
                Memory = new MemorySettings { MemoryLimitMB = 2048 }
            };

            Console.WriteLine("Initializing AI model...");
            await aiService.InitializeAsync(config);
            Console.WriteLine("✅ Model initialized successfully");

            // Connect to MCP server if specified
            if (!string.IsNullOrEmpty(mcpServer))
            {
                Console.WriteLine($"Connecting to MCP server: {mcpServer}");
                try
                {
                    await aiService.ConnectToMCPServerAsync(mcpServer);
                    Console.WriteLine("✅ Connected to MCP server");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"⚠️  Failed to connect to MCP server: {ex.Message}");
                }
            }

            // Setup tool context
            var toolContext = new ToolContext
            {
                AvailableFunctions = aiService.FunctionCaller.GetAvailableFunctions().ToList(),
                AutoExecute = true,
                MaxFunctionCalls = 3
            };

            Console.WriteLine($"Available functions: {string.Join(", ", toolContext.AvailableFunctions.Select(f => f.Name))}");
            Console.WriteLine();

            while (true)
            {
                Console.Write("You: ");
                var input = Console.ReadLine();

                if (string.IsNullOrEmpty(input))
                    continue;

                if (input.ToLower() == "exit")
                    break;

                if (input.ToLower() == "help")
                {
                    ShowHelp();
                    continue;
                }

                if (input.ToLower() == "status")
                {
                    ShowStatus(aiService);
                    continue;
                }

                try
                {
                    Console.Write("AI: ");
                    var request = new GenerationRequest
                    {
                        Prompt = input,
                        Temperature = temperature,
                        MaxTokens = maxTokens
                    };

                    var response = await aiService.GenerateWithToolsAsync(request, toolContext);
                    Console.WriteLine(response.Text);
                    Console.WriteLine($"[Tokens: {response.TokensUsed}, Time: {response.GenerationTimeMs}ms]");
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error: {ex.Message}");
                    Console.WriteLine();
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Failed to initialize: {ex.Message}");
        }
    }

    static async Task RunGeneration(string prompt, string? modelPath, float temperature, int maxTokens, bool isCode)
    {
        Console.WriteLine($"🚀 Generating {(isCode ? "code" : "text")} for prompt: \"{prompt}\"");
        Console.WriteLine();

        using var aiService = new AIService();

        try
        {
            var config = new ModelConfiguration
            {
                ModelPath = modelPath ?? "demo-model",
                Temperature = temperature,
                MaxTokens = maxTokens,
                EnableCpuOptimization = true
            };

            await aiService.InitializeAsync(config);

            var request = new GenerationRequest
            {
                Prompt = prompt,
                Temperature = temperature,
                MaxTokens = maxTokens
            };

            var response = isCode 
                ? await aiService.GenerateCodeAsync(request)
                : await aiService.Model.GenerateTextAsync(request);

            Console.WriteLine("Generated Output:");
            Console.WriteLine("================");
            Console.WriteLine(response.Text);
            Console.WriteLine();
            Console.WriteLine($"Statistics:");
            Console.WriteLine($"  Tokens used: {response.TokensUsed}");
            Console.WriteLine($"  Generation time: {response.GenerationTimeMs}ms");
            Console.WriteLine($"  Memory used: {response.Usage.MemoryUsedMB}MB");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
        }
    }

    static async Task RunFunctionTest()
    {
        Console.WriteLine("🧪 Testing Function Calling Capabilities");
        Console.WriteLine();

        using var aiService = new AIService();

        try
        {
            var config = new ModelConfiguration
            {
                ModelPath = "demo-model",
                EnableCpuOptimization = true
            };

            await aiService.InitializeAsync(config);

            // Test built-in functions
            Console.WriteLine("Testing built-in functions:");

            // Test get_current_time
            var timeCall = new FunctionCall
            {
                Name = "get_current_time",
                Arguments = JsonSerializer.SerializeToElement(""),
                CallId = "test-time"
            };

            var timeResult = await aiService.FunctionCaller.ExecuteFunctionAsync(timeCall);
            Console.WriteLine($"  get_current_time: {(timeResult.Success ? timeResult.Result : timeResult.ErrorMessage)}");

            // Test calculate
            var mathCall = new FunctionCall
            {
                Name = "calculate",
                Arguments = JsonSerializer.SerializeToElement(new { A = 10.0, B = 5.0, Operator = "+" }),
                CallId = "test-math"
            };

            var mathResult = await aiService.FunctionCaller.ExecuteFunctionAsync(mathCall);
            Console.WriteLine($"  calculate(10 + 5): {(mathResult.Success ? mathResult.Result : mathResult.ErrorMessage)}");

            // Test web_search
            var searchCall = new FunctionCall
            {
                Name = "web_search",
                Arguments = JsonSerializer.SerializeToElement(new { Query = "artificial intelligence", MaxResults = 3 }),
                CallId = "test-search"
            };

            var searchResult = await aiService.FunctionCaller.ExecuteFunctionAsync(searchCall);
            Console.WriteLine($"  web_search: {(searchResult.Success ? "✅ Success" : $"❌ {searchResult.ErrorMessage}")}");

            Console.WriteLine();
            Console.WriteLine("✅ Function calling tests completed");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
        }
    }

    static void ShowHelp()
    {
        Console.WriteLine("Available commands:");
        Console.WriteLine("  help   - Show this help message");
        Console.WriteLine("  status - Show model status and memory usage");
        Console.WriteLine("  exit   - Exit the chat");
        Console.WriteLine();
        Console.WriteLine("Function calling:");
        Console.WriteLine("  You can ask the AI to use functions like:");
        Console.WriteLine("  - 'What time is it?' (uses get_current_time)");
        Console.WriteLine("  - 'Calculate 15 * 8' (uses calculate)");
        Console.WriteLine("  - 'Search for machine learning' (uses web_search)");
        Console.WriteLine();
    }

    static void ShowStatus(AIService aiService)
    {
        var usage = aiService.Model.GetCurrentUsage();
        Console.WriteLine("Model Status:");
        Console.WriteLine($"  Ready: {aiService.Model.IsReady}");
        Console.WriteLine($"  Memory used: {usage.MemoryUsedMB}MB");
        Console.WriteLine($"  CPU usage: {usage.CpuUsagePercent:F1}%");
        Console.WriteLine($"  MCP connected: {aiService.MCPClient.IsConnected}");
        Console.WriteLine($"  Available functions: {aiService.FunctionCaller.GetAvailableFunctions().Count}");
        Console.WriteLine();
    }
}

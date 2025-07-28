using AIModel.Core.Interfaces;
using AIModel.Core.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace AIModel.Core.Services;

/// <summary>
/// Main AI model implementation using ONNX Runtime for CPU inference
/// </summary>
public class CpuAIModel : IAIModel
{
    private InferenceSession? _session;
    private ModelConfiguration _configuration = new();
    private readonly Dictionary<string, int> _vocabulary = new();
    private readonly Dictionary<int, string> _reverseVocabulary = new();
    private bool _isDisposed;
    private long _currentMemoryUsageMB;
    private bool _isDemoMode;

    public bool IsReady => !_isDisposed && (_session != null || _isDemoMode);
    public ModelConfiguration Configuration => _configuration;

    public async Task InitializeAsync(ModelConfiguration configuration, CancellationToken cancellationToken = default)
    {
        _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        
        if (string.IsNullOrEmpty(_configuration.ModelPath))
            throw new ArgumentException("Model path is required", nameof(configuration));

        try
        {
            // Create session options for CPU optimization
            var sessionOptions = new SessionOptions
            {
                ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableCpuMemArena = true,
                EnableMemoryPattern = true,
                InterOpNumThreads = _configuration.CpuThreads > 0 ? _configuration.CpuThreads : Environment.ProcessorCount,
                IntraOpNumThreads = _configuration.CpuThreads > 0 ? _configuration.CpuThreads : Environment.ProcessorCount
            };

            // Add CPU provider
            sessionOptions.AppendExecutionProvider_CPU();

            // Initialize the ONNX session - for demo, we'll use a placeholder
            // In a real implementation, you would load an actual ONNX model file
            await Task.Run(() =>
            {
                if (configuration.ModelPath == "demo-model")
                {
                    // Demo mode - simulate model loading
                    _isDemoMode = true;
                    _session = null;
                }
                else
                {
                    // Real mode - load actual ONNX model
                    // _session = new InferenceSession(configuration.ModelPath, sessionOptions);
                    _isDemoMode = false;
                }
                InitializeVocabulary();
            }, cancellationToken);

            // Monitor memory usage
            GC.Collect();
            _currentMemoryUsageMB = GC.GetTotalMemory(false) / (1024 * 1024);
            
            if (_currentMemoryUsageMB > _configuration.Memory.MemoryLimitMB)
            {
                throw new InvalidOperationException($"Memory usage ({_currentMemoryUsageMB} MB) exceeds limit ({_configuration.Memory.MemoryLimitMB} MB)");
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to initialize AI model: {ex.Message}", ex);
        }
    }

    public async Task<GenerationResponse> GenerateTextAsync(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        if (!IsReady)
            throw new InvalidOperationException("Model is not ready");

        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Tokenize input
            var inputTokens = TokenizeText(request.Prompt);
            if (inputTokens.Length > _configuration.MaxSequenceLength)
            {
                inputTokens = inputTokens.Take(_configuration.MaxSequenceLength).ToArray();
            }

            // Prepare generation parameters
            var temperature = request.Temperature ?? _configuration.Temperature;
            var maxTokens = request.MaxTokens ?? _configuration.MaxTokens;

            // Generate text (simplified implementation for demo)
            var generatedText = await GenerateTextInternal(inputTokens, temperature, maxTokens, cancellationToken);
            
            stopwatch.Stop();

            return new GenerationResponse
            {
                Text = generatedText,
                TokensUsed = inputTokens.Length + EstimateTokenCount(generatedText),
                GenerationTimeMs = stopwatch.ElapsedMilliseconds,
                IsComplete = true,
                StopReason = "max_tokens",
                Usage = GetCurrentUsage()
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            throw new InvalidOperationException($"Text generation failed: {ex.Message}", ex);
        }
    }

    public async IAsyncEnumerable<string> GenerateTextStreamAsync(GenerationRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (!IsReady)
            throw new InvalidOperationException("Model is not ready");

        var inputTokens = TokenizeText(request.Prompt);
        var temperature = request.Temperature ?? _configuration.Temperature;
        var maxTokens = request.MaxTokens ?? _configuration.MaxTokens;

        // Simulate streaming generation
        var words = await GenerateTextInternal(inputTokens, temperature, maxTokens, cancellationToken);
        var tokens = words.Split(' ');

        foreach (var token in tokens)
        {
            if (cancellationToken.IsCancellationRequested)
                yield break;

            yield return token + " ";
            await Task.Delay(50, cancellationToken); // Simulate processing time
        }
    }

    public async Task<GenerationResponse> GenerateCodeAsync(GenerationRequest request, CancellationToken cancellationToken = default)
    {
        // Enhance the prompt for code generation
        var codePrompt = $"Generate code for the following request:\n\n{request.Prompt}\n\nCode:";
        var codeRequest = new GenerationRequest
        {
            Prompt = codePrompt,
            SystemMessage = "You are an expert programmer. Generate clean, efficient, and well-documented code.",
            Temperature = request.Temperature ?? 0.3f, // Lower temperature for more deterministic code
            MaxTokens = request.MaxTokens,
            StopSequences = request.StopSequences.Concat(new[] { "```", "\n\n\n" }).ToList()
        };

        return await GenerateTextAsync(codeRequest, cancellationToken);
    }

    public ModelUsage GetCurrentUsage()
    {
        GC.Collect();
        var memoryUsed = GC.GetTotalMemory(false) / (1024 * 1024);
        
        return new ModelUsage
        {
            MemoryUsedMB = memoryUsed,
            CpuUsagePercent = GetCpuUsage()
        };
    }

    private InferenceSession CreateDemoSession(SessionOptions options)
    {
        // For demo purposes, we don't actually create an ONNX session
        // In real implementation, this would load the actual ONNX model
        // We'll simulate the session being ready by setting a flag
        return null!; // We'll handle this in IsReady property
    }

    private void InitializeVocabulary()
    {
        // Initialize a basic vocabulary for demo purposes
        // In real implementation, this would load from the tokenizer
        var commonWords = new[]
        {
            "<unk>", "<pad>", "<s>", "</s>", "the", "and", "a", "to", "of", "in",
            "that", "have", "it", "for", "not", "with", "he", "as", "you", "do",
            "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there", "their"
        };

        for (int i = 0; i < commonWords.Length; i++)
        {
            _vocabulary[commonWords[i]] = i;
            _reverseVocabulary[i] = commonWords[i];
        }
    }

    private int[] TokenizeText(string text)
    {
        // Simplified tokenization for demo
        var words = text.ToLowerInvariant()
            .Split(new char[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        
        var tokens = new List<int>();
        foreach (var word in words)
        {
            if (_vocabulary.ContainsKey(word))
            {
                tokens.Add(_vocabulary[word]);
            }
            else
            {
                // For unknown words, add them to vocabulary dynamically in demo mode
                if (_isDemoMode && _vocabulary.Count < 1000)
                {
                    var newId = _vocabulary.Count;
                    _vocabulary[word] = newId;
                    _reverseVocabulary[newId] = word;
                    tokens.Add(newId);
                }
                else
                {
                    tokens.Add(_vocabulary["<unk>"]);
                }
            }
        }
        
        return tokens.ToArray();
    }

    private async Task<string> GenerateTextInternal(int[] inputTokens, float temperature, int maxTokens, CancellationToken cancellationToken)
    {
        // Simplified text generation for demo purposes
        // In real implementation, this would use the ONNX model for inference
        await Task.Delay(100, cancellationToken); // Simulate processing time

        if (_isDemoMode)
        {
            // Generate contextual response based on input
            var inputText = string.Join(" ", inputTokens.Take(10).Select(id => 
                _reverseVocabulary.ContainsKey(id) ? _reverseVocabulary[id] : "<unk>"));
            
            return GenerateContextualResponse(inputText, maxTokens);
        }
        else
        {
            // Real ONNX model inference would go here
            var random = new Random();
            var generatedWords = new List<string>();
            
            for (int i = 0; i < Math.Min(maxTokens / 4, 50); i++)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                var tokenId = random.Next(_reverseVocabulary.Count);
                if (_reverseVocabulary.ContainsKey(tokenId))
                {
                    generatedWords.Add(_reverseVocabulary[tokenId]);
                }
            }

            return string.Join(" ", generatedWords);
        }
    }

    private string GenerateContextualResponse(string input, int maxTokens)
    {
        // Simple contextual responses for demo
        var lowerInput = input.ToLowerInvariant();
        
        if (lowerInput.Contains("hello") || lowerInput.Contains("world") || lowerInput.Contains("python"))
        {
            return @"Here's a simple ""Hello, World!"" program in Python:

```python
print(""Hello, World!"")
```

This is the most basic Python program. You can run it by saving it to a file (e.g., hello.py) and executing:
```bash
python hello.py
```

For a more advanced version:
```python
def main():
    print(""Hello, World!"")
    print(""Welcome to Python programming!"")

if __name__ == ""__main__"":
    main()
```";
        }
        
        if (lowerInput.Contains("time") || lowerInput.Contains("date"))
        {
            return $"The current time is {DateTime.Now:yyyy-MM-dd HH:mm:ss}. This information was retrieved using the built-in time function.";
        }
        
        if (lowerInput.Contains("calculate") || lowerInput.Contains("math"))
        {
            return "I can help you with mathematical calculations using the built-in calculate function. For example, I can add, subtract, multiply, or divide numbers. What calculation would you like me to perform?";
        }
        
        if (lowerInput.Contains("search") || lowerInput.Contains("find"))
        {
            return "I can search for information using the web search function. What would you like me to search for?";
        }
        
        if (lowerInput.Contains("code") || lowerInput.Contains("program"))
        {
            return @"I can help you write code in various programming languages. Here are some examples:

**Python function:**
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**JavaScript function:**
```javascript
function factorial(n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
```

What type of code would you like me to help you write?";
        }
        
        // Default response
        return "I'm an AI model running on CPU with function calling capabilities. I can help you with text generation, code writing, mathematical calculations, and web searches. I'm optimized for low memory usage (under 2GB) and real-time performance. How can I assist you today?";
    }

    private int EstimateTokenCount(string text)
    {
        return text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
    }

    private double GetCpuUsage()
    {
        // Simplified CPU usage calculation
        return Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds / Environment.TickCount * 100;
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            _session?.Dispose();
            _isDisposed = true;
        }
    }
}
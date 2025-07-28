using AIModel.Core.Interfaces;
using AIModel.Core.Models;
using System.Text.Json;
using System.Diagnostics;

namespace AIModel.Core.Services;

/// <summary>
/// Implementation of function calling capabilities
/// </summary>
public class FunctionCaller : IFunctionCaller
{
    private readonly Dictionary<string, FunctionDefinition> _functions = new();
    private readonly Dictionary<string, Func<JsonElement, Task<JsonElement>>> _handlers = new();

    public void RegisterFunction(FunctionDefinition function, Func<JsonElement, Task<JsonElement>> handler)
    {
        if (function == null)
            throw new ArgumentNullException(nameof(function));
        if (handler == null)
            throw new ArgumentNullException(nameof(handler));
        if (string.IsNullOrEmpty(function.Name))
            throw new ArgumentException("Function name cannot be empty", nameof(function));

        _functions[function.Name] = function;
        _handlers[function.Name] = handler;
    }

    public void RegisterFunction<TInput, TOutput>(
        string name, 
        string description, 
        Func<TInput, Task<TOutput>> handler,
        bool requiresConfirmation = false)
    {
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Function name cannot be empty", nameof(name));
        if (handler == null)
            throw new ArgumentNullException(nameof(handler));

        // Create function definition
        var function = new FunctionDefinition
        {
            Name = name,
            Description = description,
            RequiresConfirmation = requiresConfirmation,
            Parameters = CreateParametersSchema<TInput>()
        };

        // Create wrapper handler
        async Task<JsonElement> WrapperHandler(JsonElement args)
        {
            try
            {
                var input = JsonSerializer.Deserialize<TInput>(args.GetRawText());
                if (input == null)
                    throw new ArgumentException("Failed to deserialize input arguments");

                var result = await handler(input);
                var json = JsonSerializer.Serialize(result);
                using var doc = JsonDocument.Parse(json);
                return doc.RootElement.Clone();
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Function '{name}' execution failed: {ex.Message}", ex);
            }
        }

        RegisterFunction(function, WrapperHandler);
    }

    public async Task<FunctionCallResponse> ExecuteFunctionAsync(FunctionCall call, CancellationToken cancellationToken = default)
    {
        if (call == null)
            throw new ArgumentNullException(nameof(call));

        var stopwatch = Stopwatch.StartNew();
        var response = new FunctionCallResponse
        {
            CallId = call.CallId
        };

        try
        {
            if (!_functions.ContainsKey(call.Name))
            {
                response.Success = false;
                response.ErrorMessage = $"Function '{call.Name}' is not registered";
                return response;
            }

            var handler = _handlers[call.Name];
            var result = await handler(call.Arguments);
            
            response.Success = true;
            response.Result = result;
        }
        catch (Exception ex)
        {
            response.Success = false;
            response.ErrorMessage = ex.Message;
        }
        finally
        {
            stopwatch.Stop();
            response.ExecutionTimeMs = stopwatch.ElapsedMilliseconds;
        }

        return response;
    }

    public IReadOnlyList<FunctionDefinition> GetAvailableFunctions()
    {
        return _functions.Values.ToList().AsReadOnly();
    }

    public bool IsFunctionRegistered(string name)
    {
        return !string.IsNullOrEmpty(name) && _functions.ContainsKey(name);
    }

    public bool RemoveFunction(string name)
    {
        if (string.IsNullOrEmpty(name))
            return false;

        var removed = _functions.Remove(name);
        if (removed)
        {
            _handlers.Remove(name);
        }
        return removed;
    }

    private JsonElement CreateParametersSchema<T>()
    {
        // Simplified schema generation for demo
        // In real implementation, this would use reflection or attributes to generate proper JSON schema
        var schema = new
        {
            type = "object",
            properties = new Dictionary<string, object>(),
            required = new List<string>()
        };

        // For demo purposes, create a basic schema
        var type = typeof(T);
        if (type == typeof(string))
        {
            schema.properties["value"] = new { type = "string" };
            schema.required.Add("value");
        }
        else if (type.IsClass && type != typeof(string))
        {
            var properties = type.GetProperties();
            foreach (var prop in properties)
            {
                var propType = GetJsonType(prop.PropertyType);
                schema.properties[prop.Name.ToLowerInvariant()] = new { type = propType };
                schema.required.Add(prop.Name.ToLowerInvariant());
            }
        }

        var json = JsonSerializer.Serialize(schema);
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }

    private string GetJsonType(Type type)
    {
        if (type == typeof(string)) return "string";
        if (type == typeof(int) || type == typeof(long) || type == typeof(double) || type == typeof(float)) return "number";
        if (type == typeof(bool)) return "boolean";
        if (type.IsArray || (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(List<>))) return "array";
        return "object";
    }
}
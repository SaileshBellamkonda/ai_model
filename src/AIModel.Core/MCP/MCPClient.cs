using AIModel.Core.Interfaces;
using AIModel.Core.Models;
using System.Net.Http;
using System.Text.Json;
using System.Text;

namespace AIModel.Core.MCP;

/// <summary>
/// Model Context Protocol (MCP) client implementation
/// </summary>
public class MCPClient : IMCPClient, IDisposable
{
    private readonly HttpClient _httpClient;
    private string? _serverUrl;
    private MCPCapabilities? _serverCapabilities;
    private bool _isConnected;
    private bool _isDisposed;

    public bool IsConnected => _isConnected && !_isDisposed;
    public MCPCapabilities? ServerCapabilities => _serverCapabilities;

    public MCPClient(HttpClient? httpClient = null)
    {
        _httpClient = httpClient ?? new HttpClient();
        _httpClient.Timeout = TimeSpan.FromSeconds(30);
    }

    public async Task ConnectAsync(string serverUrl, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(serverUrl))
            throw new ArgumentException("Server URL cannot be empty", nameof(serverUrl));

        try
        {
            _serverUrl = serverUrl.TrimEnd('/');
            
            // Initialize connection with the MCP server
            var initRequest = new
            {
                jsonrpc = "2.0",
                id = 1,
                method = "initialize",
                @params = new
                {
                    protocolVersion = "1.0",
                    capabilities = new
                    {
                        tools = new { },
                        resources = new { }
                    },
                    clientInfo = new
                    {
                        name = "AIModel",
                        version = "1.0.0"
                    }
                }
            };

            var response = await SendHttpRequestAsync<InitializeResponse>("initialize", initRequest, cancellationToken);
            
            if (response?.capabilities != null)
            {
                _serverCapabilities = new MCPCapabilities
                {
                    SupportsTools = response.capabilities.tools != null,
                    SupportsResources = response.capabilities.resources != null,
                    SupportsPrompts = response.capabilities.prompts != null,
                    ProtocolVersion = response.protocolVersion ?? "1.0"
                };
                
                _isConnected = true;
            }
            else
            {
                throw new InvalidOperationException("Failed to initialize MCP connection");
            }
        }
        catch (Exception ex)
        {
            _isConnected = false;
            throw new InvalidOperationException($"Failed to connect to MCP server: {ex.Message}", ex);
        }
    }

    public async Task DisconnectAsync()
    {
        if (_isConnected)
        {
            try
            {
                // Send shutdown notification to the server
                var shutdownRequest = new
                {
                    jsonrpc = "2.0",
                    method = "notifications/shutdown"
                };
                
                await SendHttpRequestAsync<object>("notifications/shutdown", shutdownRequest, CancellationToken.None);
            }
            catch
            {
                // Ignore errors during shutdown
            }
            finally
            {
                _isConnected = false;
                _serverCapabilities = null;
            }
        }
    }

    public async Task<T> SendRequestAsync<T>(string method, object parameters, CancellationToken cancellationToken = default)
    {
        if (!IsConnected)
            throw new InvalidOperationException("Not connected to MCP server");

        var request = new
        {
            jsonrpc = "2.0",
            id = Guid.NewGuid().ToString(),
            method = method,
            @params = parameters
        };

        return await SendHttpRequestAsync<T>(method, request, cancellationToken);
    }

    public async Task<FunctionCallResponse> CallToolAsync(string toolName, JsonElement arguments, CancellationToken cancellationToken = default)
    {
        if (!IsConnected)
            throw new InvalidOperationException("Not connected to MCP server");

        if (_serverCapabilities?.SupportsTools != true)
            throw new InvalidOperationException("Server does not support tools");

        var callId = Guid.NewGuid().ToString();
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            var toolCall = new
            {
                name = toolName,
                arguments = arguments
            };

            var result = await SendRequestAsync<ToolCallResult>("tools/call", toolCall, cancellationToken);
            stopwatch.Stop();

            return new FunctionCallResponse
            {
                CallId = callId,
                Success = !result.isError,
                Result = result.content,
                ErrorMessage = result.isError ? result.content?.ToString() : null,
                ExecutionTimeMs = stopwatch.ElapsedMilliseconds
            };
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            return new FunctionCallResponse
            {
                CallId = callId,
                Success = false,
                ErrorMessage = ex.Message,
                ExecutionTimeMs = stopwatch.ElapsedMilliseconds
            };
        }
    }

    public async Task<List<FunctionDefinition>> GetAvailableToolsAsync(CancellationToken cancellationToken = default)
    {
        if (!IsConnected)
            throw new InvalidOperationException("Not connected to MCP server");

        if (_serverCapabilities?.SupportsTools != true)
            return new List<FunctionDefinition>();

        try
        {
            var response = await SendRequestAsync<ToolsListResponse>("tools/list", new { }, cancellationToken);
            
            var functions = new List<FunctionDefinition>();
            if (response?.tools != null)
            {
                foreach (var tool in response.tools)
                {
                    functions.Add(new FunctionDefinition
                    {
                        Name = tool.name ?? string.Empty,
                        Description = tool.description ?? string.Empty,
                        Parameters = tool.inputSchema
                    });
                }
            }

            return functions;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to get available tools: {ex.Message}", ex);
        }
    }

    private async Task<T> SendHttpRequestAsync<T>(string endpoint, object request, CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(_serverUrl))
            throw new InvalidOperationException("Server URL is not set");

        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var url = $"{_serverUrl}/{endpoint.TrimStart('/')}";
        var response = await _httpClient.PostAsync(url, content, cancellationToken);
        
        response.EnsureSuccessStatusCode();
        
        var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
        
        if (typeof(T) == typeof(object))
            return default!;

        return JsonSerializer.Deserialize<T>(responseJson) ?? throw new InvalidOperationException("Failed to deserialize response");
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            DisconnectAsync().Wait(TimeSpan.FromSeconds(5));
            _httpClient?.Dispose();
            _isDisposed = true;
        }
    }

    // Internal response models for MCP protocol
    private class InitializeResponse
    {
        public string? protocolVersion { get; set; }
        public McpServerCapabilities? capabilities { get; set; }
        public ServerInfo? serverInfo { get; set; }
    }

    private class McpServerCapabilities
    {
        public object? tools { get; set; }
        public object? resources { get; set; }
        public object? prompts { get; set; }
    }

    private class ServerInfo
    {
        public string? name { get; set; }
        public string? version { get; set; }
    }

    private class ToolsListResponse
    {
        public List<ToolInfo>? tools { get; set; }
    }

    private class ToolInfo
    {
        public string? name { get; set; }
        public string? description { get; set; }
        public JsonElement inputSchema { get; set; }
    }

    private class ToolCallResult
    {
        public bool isError { get; set; }
        public JsonElement? content { get; set; }
    }
}
using AIModel.Core.Models;
using System.Text.Json;

namespace AIModel.Core.Interfaces;

/// <summary>
/// Interface for Model Context Protocol (MCP) support
/// </summary>
public interface IMCPClient : IDisposable
{
    /// <summary>
    /// Connect to an MCP server
    /// </summary>
    Task ConnectAsync(string serverUrl, CancellationToken cancellationToken = default);

    /// <summary>
    /// Disconnect from the MCP server
    /// </summary>
    Task DisconnectAsync();

    /// <summary>
    /// Send a request to the MCP server
    /// </summary>
    Task<T> SendRequestAsync<T>(string method, object parameters, CancellationToken cancellationToken = default);

    /// <summary>
    /// Call a tool via MCP
    /// </summary>
    Task<FunctionCallResponse> CallToolAsync(string toolName, JsonElement arguments, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get available tools from the MCP server
    /// </summary>
    Task<List<FunctionDefinition>> GetAvailableToolsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Whether the client is connected
    /// </summary>
    bool IsConnected { get; }

    /// <summary>
    /// Server capabilities
    /// </summary>
    MCPCapabilities? ServerCapabilities { get; }
}

/// <summary>
/// MCP server capabilities
/// </summary>
public class MCPCapabilities
{
    /// <summary>
    /// Server supports tool calling
    /// </summary>
    public bool SupportsTools { get; set; }

    /// <summary>
    /// Server supports resource access
    /// </summary>
    public bool SupportsResources { get; set; }

    /// <summary>
    /// Server supports prompts
    /// </summary>
    public bool SupportsPrompts { get; set; }

    /// <summary>
    /// Protocol version
    /// </summary>
    public string ProtocolVersion { get; set; } = "1.0";
}
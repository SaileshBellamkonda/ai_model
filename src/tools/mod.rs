use crate::{AIError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Function call request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
    pub call_id: Option<String>,
}

/// Function call response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallResult {
    pub call_id: Option<String>,
    pub result: serde_json::Value,
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

/// Tool definition structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema for parameters
    pub required_permissions: Vec<String>,
}

/// Trait for implementing tool handlers
#[async_trait::async_trait]
pub trait ToolHandler: Send + Sync {
    /// Execute the tool with given arguments
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value>;
    
    /// Get the tool definition
    fn definition(&self) -> ToolDefinition;
    
    /// Check if the tool requires specific permissions
    fn required_permissions(&self) -> Vec<String> {
        vec![]
    }
}

/// Registry for managing available tools and functions
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolHandler>>,
    permissions: HashMap<String, bool>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            permissions: HashMap::new(),
        }
    }
    
    /// Register a new tool
    pub fn register_tool(
        &mut self, 
        _name: String, 
        _description: String, 
        handler: Box<dyn ToolHandler>
    ) -> Result<()> {
        let handler: Arc<dyn ToolHandler> = Arc::from(handler);
        
        // Check if tool already exists
        let tool_name = handler.definition().name.clone();
        if self.tools.contains_key(&tool_name) {
            return Err(AIError::ToolError(format!("Tool '{}' already registered", tool_name)));
        }
        
        self.tools.insert(tool_name, handler);
        Ok(())
    }
    
    /// Call a tool by name with arguments
    pub async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let start_time = std::time::Instant::now();
        
        let handler = self.tools.get(name)
            .ok_or_else(|| AIError::ToolError(format!("Tool '{}' not found", name)))?;
        
        // Check permissions
        for permission in handler.required_permissions() {
            if !self.permissions.get(&permission).unwrap_or(&false) {
                return Err(AIError::ToolError(
                    format!("Missing permission '{}' for tool '{}'", permission, name)
                ));
            }
        }
        
        // Execute the tool
        let result = handler.execute(args).await?;
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        log::info!("Tool '{}' executed in {}ms", name, execution_time);
        
        Ok(result)
    }
    
    /// Get list of available tools
    pub fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.values()
            .map(|handler| handler.definition())
            .collect()
    }
    
    /// Grant permission for a specific capability
    pub fn grant_permission(&mut self, permission: String) {
        self.permissions.insert(permission, true);
    }
    
    /// Revoke permission for a specific capability
    pub fn revoke_permission(&mut self, permission: String) {
        self.permissions.insert(permission, false);
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in tools that come with the AI model

/// HTTP request tool for external API calls
pub struct HttpRequestTool {
    client: reqwest::Client,
}

impl HttpRequestTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl ToolHandler for HttpRequestTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let url = args.get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AIError::ToolError("Missing 'url' parameter".to_string()))?;
        
        let method = args.get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET");
        
        let headers = args.get("headers")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        
        let body = args.get("body");
        
        // Build request
        let mut request = match method.to_uppercase().as_str() {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "DELETE" => self.client.delete(url),
            _ => return Err(AIError::ToolError(format!("Unsupported HTTP method: {}", method))),
        };
        
        // Add headers
        for (key, value) in headers {
            if let Some(value_str) = value.as_str() {
                request = request.header(key, value_str);
            }
        }
        
        // Add body if present
        if let Some(body_value) = body {
            request = request.json(body_value);
        }
        
        // Execute request
        let response = request.send().await
            .map_err(|e| AIError::ToolError(format!("HTTP request failed: {}", e)))?;
        
        let status = response.status().as_u16();
        let text = response.text().await
            .map_err(|e| AIError::ToolError(format!("Failed to read response: {}", e)))?;
        
        Ok(serde_json::json!({
            "status": status,
            "body": text,
            "success": status >= 200 && status < 300
        }))
    }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "http_request".to_string(),
            description: "Make HTTP requests to external APIs".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to make the request to"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET",
                        "description": "HTTP method"
                    },
                    "headers": {
                        "type": "object",
                        "description": "HTTP headers to include"
                    },
                    "body": {
                        "description": "Request body (for POST/PUT requests)"
                    }
                },
                "required": ["url"]
            }),
            required_permissions: vec!["network_access".to_string()],
        }
    }
    
    fn required_permissions(&self) -> Vec<String> {
        vec!["network_access".to_string()]
    }
}

/// File operations tool
pub struct FileOperationsTool;

#[async_trait::async_trait]
impl ToolHandler for FileOperationsTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let operation = args.get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AIError::ToolError("Missing 'operation' parameter".to_string()))?;
        
        let path = args.get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AIError::ToolError("Missing 'path' parameter".to_string()))?;
        
        match operation {
            "read" => {
                let content = tokio::fs::read_to_string(path).await
                    .map_err(|e| AIError::ToolError(format!("Failed to read file: {}", e)))?;
                Ok(serde_json::json!({ "content": content }))
            },
            "write" => {
                let content = args.get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| AIError::ToolError("Missing 'content' parameter for write operation".to_string()))?;
                
                tokio::fs::write(path, content).await
                    .map_err(|e| AIError::ToolError(format!("Failed to write file: {}", e)))?;
                Ok(serde_json::json!({ "success": true }))
            },
            "exists" => {
                let exists = tokio::fs::metadata(path).await.is_ok();
                Ok(serde_json::json!({ "exists": exists }))
            },
            _ => Err(AIError::ToolError(format!("Unsupported file operation: {}", operation))),
        }
    }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "file_operations".to_string(),
            description: "Perform file system operations".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "exists"],
                        "description": "File operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write (for write operation)"
                    }
                },
                "required": ["operation", "path"]
            }),
            required_permissions: vec!["file_system_access".to_string()],
        }
    }
    
    fn required_permissions(&self) -> Vec<String> {
        vec!["file_system_access".to_string()]
    }
}

/// Calculator tool for mathematical operations
pub struct CalculatorTool;

#[async_trait::async_trait]
impl ToolHandler for CalculatorTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value> {
        let expression = args.get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| AIError::ToolError("Missing 'expression' parameter".to_string()))?;
        
        // Simple calculator implementation (in practice, you'd use a proper expression parser)
        let result = match self.evaluate_simple_expression(expression) {
            Ok(value) => value,
            Err(e) => return Err(AIError::ToolError(format!("Calculation error: {}", e))),
        };
        
        Ok(serde_json::json!({ "result": result }))
    }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "Perform mathematical calculations".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
            required_permissions: vec![],
        }
    }
}

impl CalculatorTool {
    fn evaluate_simple_expression(&self, expr: &str) -> std::result::Result<f64, String> {
        // Very basic calculator - in practice you'd use a proper parser
        let expr = expr.replace(" ", "");
        
        if let Ok(num) = expr.parse::<f64>() {
            return Ok(num);
        }
        
        // Handle simple binary operations
        for op in ['+', '-', '*', '/'] {
            if let Some(pos) = expr.rfind(op) {
                let left = &expr[..pos];
                let right = &expr[pos + 1..];
                
                let left_val = self.evaluate_simple_expression(left)?;
                let right_val = self.evaluate_simple_expression(right)?;
                
                return match op {
                    '+' => Ok(left_val + right_val),
                    '-' => Ok(left_val - right_val),
                    '*' => Ok(left_val * right_val),
                    '/' => {
                        if right_val == 0.0 {
                            Err("Division by zero".to_string())
                        } else {
                            Ok(left_val / right_val)
                        }
                    },
                    _ => unreachable!(),
                };
            }
        }
        
        Err(format!("Invalid expression: {}", expr))
    }
}
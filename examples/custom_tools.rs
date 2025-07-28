use ai_model::{AIModel, ModelConfig, tools::{ToolHandler, ToolDefinition}};
use async_trait::async_trait;
use serde_json::Value;

/// Custom weather tool that simulates weather API calls
pub struct WeatherTool {
    api_key: String,
}

impl WeatherTool {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}

#[async_trait]
impl ToolHandler for WeatherTool {
    async fn execute(&self, args: Value) -> ai_model::Result<Value> {
        let location = args.get("location")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ai_model::AIError::ToolError("Missing 'location' parameter".to_string()))?;
        
        // Simulate API call (in real implementation, you'd call a weather API)
        let weather_data = match location.to_lowercase().as_str() {
            "london" => serde_json::json!({
                "location": "London",
                "temperature": 15,
                "condition": "Cloudy",
                "humidity": 70,
                "wind_speed": 12
            }),
            "new york" => serde_json::json!({
                "location": "New York",
                "temperature": 22,
                "condition": "Sunny",
                "humidity": 60,
                "wind_speed": 8
            }),
            "tokyo" => serde_json::json!({
                "location": "Tokyo",
                "temperature": 18,
                "condition": "Rainy",
                "humidity": 85,
                "wind_speed": 5
            }),
            _ => serde_json::json!({
                "location": location,
                "temperature": 20,
                "condition": "Unknown",
                "humidity": 50,
                "wind_speed": 10,
                "note": "Weather data not available for this location"
            }),
        };
        
        Ok(weather_data)
    }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "weather".to_string(),
            description: "Get current weather information for a location".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for (e.g., 'London', 'New York')"
                    }
                },
                "required": ["location"]
            }),
            required_permissions: vec!["network_access".to_string()],
        }
    }
    
    fn required_permissions(&self) -> Vec<String> {
        vec!["network_access".to_string()]
    }
}

/// Custom time zone tool
pub struct TimezoneConverter;

#[async_trait]
impl ToolHandler for TimezoneConverter {
    async fn execute(&self, args: Value) -> ai_model::Result<Value> {
        let from_tz = args.get("from_timezone")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ai_model::AIError::ToolError("Missing 'from_timezone' parameter".to_string()))?;
        
        let to_tz = args.get("to_timezone")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ai_model::AIError::ToolError("Missing 'to_timezone' parameter".to_string()))?;
        
        let time = args.get("time")
            .and_then(|v| v.as_str())
            .unwrap_or("12:00");
        
        // Simulate timezone conversion (simplified)
        let converted_time = match (from_tz, to_tz) {
            ("UTC", "EST") => format!("{}EST (UTC-5)", time),
            ("UTC", "PST") => format!("{}PST (UTC-8)", time),
            ("EST", "UTC") => format!("{}UTC (EST+5)", time),
            ("PST", "UTC") => format!("{}UTC (PST+8)", time),
            _ => format!("{}(converted from {} to {})", time, from_tz, to_tz),
        };
        
        Ok(serde_json::json!({
            "original_time": time,
            "from_timezone": from_tz,
            "to_timezone": to_tz,
            "converted_time": converted_time
        }))
    }
    
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "timezone_converter".to_string(),
            description: "Convert time between different timezones".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "from_timezone": {
                        "type": "string",
                        "description": "Source timezone (e.g., 'UTC', 'EST', 'PST')"
                    },
                    "to_timezone": {
                        "type": "string",
                        "description": "Target timezone (e.g., 'UTC', 'EST', 'PST')"
                    },
                    "time": {
                        "type": "string",
                        "description": "Time to convert (e.g., '14:30')",
                        "default": "12:00"
                    }
                },
                "required": ["from_timezone", "to_timezone"]
            }),
            required_permissions: vec![],
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Create model with default configuration
    let config = ModelConfig::default();
    let model = AIModel::new(config).await?;
    
    println!("=== Custom Tool Development Example ===\n");
    
    // Register weather tool
    let weather_tool = WeatherTool::new("demo_api_key".to_string());
    model.register_tool(
        "weather".to_string(),
        "Weather information tool".to_string(),
        Box::new(weather_tool),
    ).await?;
    
    // Register timezone converter tool
    model.register_tool(
        "timezone_converter".to_string(),
        "Timezone conversion tool".to_string(),
        Box::new(TimezoneConverter),
    ).await?;
    
    // Grant necessary permissions
    // Note: In a real application, you'd have a proper permission system
    println!("Registered custom tools successfully!\n");
    
    // Test weather tool
    println!("=== Testing Weather Tool ===");
    let locations = vec!["London", "New York", "Tokyo", "Unknown City"];
    
    for location in locations {
        let args = serde_json::json!({
            "location": location
        });
        
        match model.call_tool("weather", args).await {
            Ok(result) => {
                println!("Weather in {}: {}", location, result);
            },
            Err(e) => {
                println!("Error getting weather for {}: {}", location, e);
            }
        }
    }
    
    println!();
    
    // Test timezone converter tool
    println!("=== Testing Timezone Converter Tool ===");
    let conversions = vec![
        ("UTC", "EST", "14:00"),
        ("PST", "UTC", "09:30"),
        ("EST", "PST", "17:45"),
    ];
    
    for (from_tz, to_tz, time) in conversions {
        let args = serde_json::json!({
            "from_timezone": from_tz,
            "to_timezone": to_tz,
            "time": time
        });
        
        match model.call_tool("timezone_converter", args).await {
            Ok(result) => {
                println!("Conversion result: {}", result);
            },
            Err(e) => {
                println!("Error converting timezone: {}", e);
            }
        }
    }
    
    println!("\n=== Tool Integration with AI Model ===");
    
    // Demonstrate how the AI model could use these tools in real scenarios
    let scenarios = vec![
        ("What's the weather like in London?", "weather", serde_json::json!({"location": "London"})),
        ("Convert 3 PM EST to UTC", "timezone_converter", serde_json::json!({"from_timezone": "EST", "to_timezone": "UTC", "time": "15:00"})),
    ];
    
    for (query, tool_name, tool_args) in scenarios {
        println!("Query: '{}'", query);
        
        // In a real implementation, the AI model would:
        // 1. Understand the query
        // 2. Determine which tool to use
        // 3. Extract the necessary parameters
        // 4. Call the tool
        // 5. Format the response
        
        match model.call_tool(tool_name, tool_args).await {
            Ok(result) => {
                println!("AI Response: Based on the {} tool, here's the information: {}", tool_name, result);
            },
            Err(e) => {
                println!("Error: {}", e);
            }
        }
        println!();
    }
    
    println!("=== Available Tools ===");
    // In a real implementation, you could list all registered tools
    println!("The AI model now has access to:");
    println!("- calculator: Perform mathematical calculations");
    println!("- weather: Get weather information for any location");
    println!("- timezone_converter: Convert time between timezones");
    println!("- http_request: Make HTTP requests to external APIs");
    println!("- file_operations: Perform file system operations");
    
    Ok(())
}
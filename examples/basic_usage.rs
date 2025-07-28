use ai_model::{AIModel, ModelConfig, TaskType};
use ai_model::tools::CalculatorTool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Create a lightweight model configuration
    let config = ModelConfig {
        max_memory_mb: 1024,  // 1GB limit for this example
        hidden_size: 256,     // Smaller model for faster inference
        num_layers: 4,        // Fewer layers for efficiency
        vocab_size: 16000,    // Smaller vocabulary
        cpu_threads: 4,       // Use 4 CPU threads
        ..Default::default()
    };
    
    println!("Creating AI model with lightweight configuration...");
    let model = AIModel::new(config).await?;
    
    // Register the calculator tool
    model.register_tool(
        "calculator".to_string(),
        "Mathematical calculator tool".to_string(),
        Box::new(CalculatorTool),
    ).await?;
    
    println!("Model initialized successfully!\n");
    
    // Example 1: Text Generation
    println!("=== Text Generation Example ===");
    let task = TaskType::TextGeneration {
        max_tokens: 50,
        temperature: 0.7,
    };
    
    let result = model.execute_task(task, "The future of artificial intelligence").await?;
    println!("Prompt: 'The future of artificial intelligence'");
    println!("Generated: {}", result.text_output());
    println!();
    
    // Example 2: Code Completion
    println!("=== Code Completion Example ===");
    let task = TaskType::CodeCompletion {
        language: "python".to_string(),
        context_lines: 5,
    };
    
    let result = model.execute_task(task, "def calculate_fibonacci(n):").await?;
    println!("Code prompt: 'def calculate_fibonacci(n):'");
    println!("Completion: {}", result.text_output());
    println!();
    
    // Example 3: Question Answering
    println!("=== Question Answering Example ===");
    let task = TaskType::QuestionAnswer {
        context: Some("Rust is a systems programming language focused on safety, speed, and concurrency.".to_string()),
    };
    
    let result = model.execute_task(task, "What is Rust?").await?;
    println!("Question: 'What is Rust?'");
    println!("Answer: {}", result.text_output());
    println!();
    
    // Example 4: Text Summarization
    println!("=== Text Summarization Example ===");
    let long_text = "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. Instead of being explicitly programmed to perform a task, machine learning algorithms build mathematical models based on training data in order to make predictions or decisions without being explicitly programmed to perform the task.";
    
    let task = TaskType::Summarization {
        max_length: 30,
        style: "brief".to_string(),
    };
    
    let result = model.execute_task(task, long_text).await?;
    println!("Original text: {} chars", long_text.len());
    println!("Summary: {}", result.text_output());
    println!();
    
    // Example 5: Tool Usage
    println!("=== Tool Usage Example ===");
    let calc_args = serde_json::json!({
        "expression": "15 * 8 + 4"
    });
    
    let tool_result = model.call_tool("calculator", calc_args).await?;
    println!("Calculator tool result: {}", tool_result);
    println!();
    
    // Example 6: Performance Metrics
    println!("=== Performance Metrics ===");
    let metrics = model.get_metrics().await;
    println!("Memory usage: {:.2} MB", metrics.memory_usage_mb);
    println!("Last inference time: {:.2} ms", metrics.inference_time_ms);
    println!("Throughput: {:.2} tokens/sec", metrics.throughput_tokens_per_sec);
    
    Ok(())
}
use ai_model::{
    AIModel, ModelConfig, TaskType,
    tools::{HttpRequestTool, FileOperationsTool, CalculatorTool},
    utils::{setup_logging, get_system_info, load_config_from_env}
};
use clap::{Parser, Subcommand};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "ai_model")]
#[command(about = "A lightweight, high-accuracy machine learning model")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,
    
    /// Maximum memory usage in MB
    #[arg(long, default_value = "2048")]
    max_memory_mb: usize,
    
    /// Number of CPU threads to use
    #[arg(long)]
    cpu_threads: Option<usize>,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate text based on a prompt
    Generate {
        /// Input prompt
        prompt: String,
        /// Maximum tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        /// Temperature for generation (0.0 - 2.0)
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },
    
    /// Complete code based on partial input
    Complete {
        /// Partial code to complete
        code: String,
        /// Programming language
        #[arg(long, default_value = "python")]
        language: String,
        /// Number of context lines
        #[arg(long, default_value = "10")]
        context_lines: usize,
    },
    
    /// Answer a question
    Answer {
        /// Question to answer
        question: String,
        /// Optional context
        #[arg(long)]
        context: Option<String>,
    },
    
    /// Summarize text
    Summarize {
        /// Text to summarize
        text: String,
        /// Maximum summary length
        #[arg(long, default_value = "100")]
        max_length: usize,
        /// Summary style
        #[arg(long, default_value = "brief")]
        style: String,
    },
    
    /// Analyze an image (basic placeholder)
    Analyze {
        /// Image description
        description: String,
        /// Path to image file
        #[arg(long)]
        image_path: Option<String>,
    },
    
    /// Interactive mode
    Interactive,
    
    /// Call a tool/function
    Tool {
        /// Tool name
        name: String,
        /// Tool arguments in JSON format
        args: String,
    },
    
    /// Show system information and model status
    Info,
    
    /// Run benchmark tests
    Benchmark,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Setup logging
    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else {
        std::env::set_var("RUST_LOG", "info");
    }
    setup_logging()?;
    
    // Load configuration
    let mut config = if let Some(config_path) = &cli.config {
        load_config_from_file(config_path).await?
    } else {
        load_config_from_env()
    };
    
    // Override with CLI arguments
    config.max_memory_mb = cli.max_memory_mb;
    if let Some(threads) = cli.cpu_threads {
        config.cpu_threads = threads;
    }
    
    log::info!("Initializing AI model with config: {:?}", config);
    
    // Initialize the AI model
    let model = AIModel::new(config).await?;
    
    // Register built-in tools
    register_built_in_tools(&model).await?;
    
    match cli.command {
        Commands::Generate { prompt, max_tokens, temperature } => {
            let task = TaskType::TextGeneration { max_tokens, temperature };
            execute_task(&model, task, &prompt).await?;
        },
        
        Commands::Complete { code, language, context_lines } => {
            let task = TaskType::CodeCompletion { language, context_lines };
            execute_task(&model, task, &code).await?;
        },
        
        Commands::Answer { question, context } => {
            let task = TaskType::QuestionAnswer { context };
            execute_task(&model, task, &question).await?;
        },
        
        Commands::Summarize { text, max_length, style } => {
            let task = TaskType::Summarization { max_length, style };
            execute_task(&model, task, &text).await?;
        },
        
        Commands::Analyze { description, image_path } => {
            let image_data = if let Some(path) = image_path {
                tokio::fs::read(&path).await
                    .map_err(|e| format!("Failed to read image file: {}", e))?
            } else {
                vec![] // Placeholder for image data
            };
            
            let task = TaskType::VisualAnalysis { image_data };
            execute_task(&model, task, &description).await?;
        },
        
        Commands::Interactive => {
            run_interactive_mode(&model).await?;
        },
        
        Commands::Tool { name, args } => {
            let args_json: serde_json::Value = serde_json::from_str(&args)
                .map_err(|e| format!("Invalid JSON arguments: {}", e))?;
            
            let result = model.call_tool(&name, args_json).await?;
            println!("Tool result: {}", serde_json::to_string_pretty(&result)?);
        },
        
        Commands::Info => {
            show_system_info(&model).await?;
        },
        
        Commands::Benchmark => {
            run_benchmark(&model).await?;
        },
    }
    
    Ok(())
}

async fn execute_task(model: &AIModel, task: TaskType, input: &str) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = std::time::Instant::now();
    
    log::info!("Executing task: {:?}", task);
    let result = model.execute_task(task, input).await?;
    
    let elapsed = start_time.elapsed();
    
    println!("\n=== Result ===");
    println!("{}", result.text_output());
    
    println!("\n=== Performance ===");
    println!("Execution time: {:.2}ms", elapsed.as_millis());
    
    let metrics = model.get_metrics().await;
    println!("Memory usage: {:.2}MB", metrics.memory_usage_mb);
    println!("Throughput: {:.2} tokens/sec", metrics.throughput_tokens_per_sec);
    
    Ok(())
}

async fn register_built_in_tools(model: &AIModel) -> Result<(), Box<dyn std::error::Error>> {
    // Register HTTP request tool
    model.register_tool(
        "http_request".to_string(),
        "Make HTTP requests to external APIs".to_string(),
        Box::new(HttpRequestTool::new())
    ).await?;
    
    // Register file operations tool
    model.register_tool(
        "file_operations".to_string(),
        "Perform file system operations".to_string(),
        Box::new(FileOperationsTool)
    ).await?;
    
    // Register calculator tool
    model.register_tool(
        "calculator".to_string(),
        "Perform mathematical calculations".to_string(),
        Box::new(CalculatorTool)
    ).await?;
    
    log::info!("Registered built-in tools");
    Ok(())
}

async fn run_interactive_mode(model: &AIModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AI Model Interactive Mode ===");
    println!("Commands:");
    println!("  generate <prompt> - Generate text");
    println!("  complete <code> - Complete code");
    println!("  answer <question> - Answer question");
    println!("  summarize <text> - Summarize text");
    println!("  tool <name> <args> - Call a tool");
    println!("  info - Show model info");
    println!("  quit - Exit");
    println!();
    
    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = input.splitn(2, ' ').collect();
        let command = parts[0];
        let args = parts.get(1).unwrap_or(&"");
        
        match command {
            "quit" | "exit" => break,
            
            "generate" => {
                if args.is_empty() {
                    println!("Usage: generate <prompt>");
                    continue;
                }
                
                let task = TaskType::TextGeneration { max_tokens: 100, temperature: 0.7 };
                if let Err(e) = execute_task(model, task, args).await {
                    println!("Error: {}", e);
                }
            },
            
            "complete" => {
                if args.is_empty() {
                    println!("Usage: complete <code>");
                    continue;
                }
                
                let task = TaskType::CodeCompletion { 
                    language: "python".to_string(), 
                    context_lines: 10 
                };
                if let Err(e) = execute_task(model, task, args).await {
                    println!("Error: {}", e);
                }
            },
            
            "answer" => {
                if args.is_empty() {
                    println!("Usage: answer <question>");
                    continue;
                }
                
                let task = TaskType::QuestionAnswer { context: None };
                if let Err(e) = execute_task(model, task, args).await {
                    println!("Error: {}", e);
                }
            },
            
            "summarize" => {
                if args.is_empty() {
                    println!("Usage: summarize <text>");
                    continue;
                }
                
                let task = TaskType::Summarization { 
                    max_length: 100, 
                    style: "brief".to_string() 
                };
                if let Err(e) = execute_task(model, task, args).await {
                    println!("Error: {}", e);
                }
            },
            
            "tool" => {
                let tool_parts: Vec<&str> = args.splitn(2, ' ').collect();
                if tool_parts.len() < 2 {
                    println!("Usage: tool <name> <args_json>");
                    continue;
                }
                
                let tool_name = tool_parts[0];
                let tool_args = tool_parts[1];
                
                match serde_json::from_str::<serde_json::Value>(tool_args) {
                    Ok(args_json) => {
                        match model.call_tool(tool_name, args_json).await {
                            Ok(result) => {
                                println!("Tool result: {}", serde_json::to_string_pretty(&result)?);
                            },
                            Err(e) => println!("Tool error: {}", e),
                        }
                    },
                    Err(_) => println!("Invalid JSON arguments"),
                }
            },
            
            "info" => {
                if let Err(e) = show_system_info(model).await {
                    println!("Error: {}", e);
                }
            },
            
            _ => {
                println!("Unknown command: {}. Type 'quit' to exit.", command);
            }
        }
        
        println!();
    }
    
    println!("Goodbye!");
    Ok(())
}

async fn show_system_info(model: &AIModel) -> Result<(), Box<dyn std::error::Error>> {
    let system_info = get_system_info();
    let config = model.get_config();
    let metrics = model.get_metrics().await;
    
    println!("=== System Information ===");
    println!("OS: {} ({})", system_info.os, system_info.arch);
    println!("CPU cores: {}", system_info.cpu_count);
    println!("Available memory: {}MB", system_info.available_memory_mb);
    
    println!("\n=== Model Configuration ===");
    println!("Max memory: {}MB", config.max_memory_mb);
    println!("CPU threads: {}", config.cpu_threads);
    println!("Hidden size: {}", config.hidden_size);
    println!("Number of layers: {}", config.num_layers);
    println!("Vocabulary size: {}", config.vocab_size);
    println!("Max sequence length: {}", config.max_sequence_length);
    
    println!("\n=== Performance Metrics ===");
    println!("Memory usage: {:.2}MB", metrics.memory_usage_mb);
    println!("Last inference time: {:.2}ms", metrics.inference_time_ms);
    println!("Throughput: {:.2} tokens/sec", metrics.throughput_tokens_per_sec);
    
    Ok(())
}

async fn run_benchmark(model: &AIModel) -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Running Benchmark Tests ===");
    
    let test_prompts = vec![
        "The quick brown fox",
        "def fibonacci(n):",
        "What is the capital of France?",
        "Summarize: The weather today is sunny and warm.",
    ];
    
    let mut total_time = 0.0;
    let mut total_tokens = 0;
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("Test {}/{}: {}", i + 1, test_prompts.len(), prompt);
        
        let start = std::time::Instant::now();
        let task = TaskType::TextGeneration { max_tokens: 50, temperature: 0.7 };
        let result = model.execute_task(task, prompt).await?;
        let elapsed = start.elapsed().as_millis() as f64;
        
        let tokens = result.output_tokens().unwrap_or(0);
        total_time += elapsed;
        total_tokens += tokens;
        
        println!("  Time: {:.2}ms, Tokens: {}, Rate: {:.2} tokens/sec",
                elapsed, tokens, (tokens as f64 * 1000.0) / elapsed);
    }
    
    println!("\n=== Benchmark Results ===");
    println!("Total time: {:.2}ms", total_time);
    println!("Total tokens: {}", total_tokens);
    println!("Average throughput: {:.2} tokens/sec", 
             (total_tokens as f64 * 1000.0) / total_time);
    
    let metrics = model.get_metrics().await;
    println!("Peak memory usage: {:.2}MB", metrics.memory_usage_mb);
    
    Ok(())
}

async fn load_config_from_file(path: &str) -> Result<ModelConfig, Box<dyn std::error::Error>> {
    let content = tokio::fs::read_to_string(path).await?;
    let config: ModelConfig = serde_json::from_str(&content)?;
    Ok(config)
}

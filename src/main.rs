use ai_model::{
    AIModel, ModelConfig, TaskType,
    tools::{HttpRequestTool, FileOperationsTool, CalculatorTool},
    utils::{setup_logging, get_system_info, load_config_from_env},
    tokenizer::{BpeTokenizer, BpeTokenizerConfig, BigcodeDatasetConfig},
    inference::{InferenceEngineFactory, InferenceConfig, InferenceEngineType},
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
    
    /// Train BPE tokenizer on bigcode dataset
    TrainTokenizer {
        /// Output path for trained tokenizer
        #[arg(short, long, default_value = "models/bpe_tokenizer.json")]
        output: String,
        /// Programming languages to include
        #[arg(long, value_delimiter = ',')]
        languages: Vec<String>,
        /// Maximum samples per language
        #[arg(long, default_value = "1000")]
        max_samples: usize,
    },
    
    /// Test inference engines (llama.cpp or ONNX)
    TestInference {
        /// Engine type (llamacpp, onnx, native)
        #[arg(short, long, default_value = "native")]
        engine: String,
        /// Model path
        #[arg(short, long)]
        model_path: Option<String>,
        /// Test prompt
        #[arg(short, long, default_value = "Hello, world!")]
        prompt: String,
    },
    
    /// Convert model format (for inference engines)
    ConvertModel {
        /// Input model path
        #[arg(short, long)]
        input: String,
        /// Output model path
        #[arg(short, long)]
        output: String,
        /// Target format (onnx, gguf)
        #[arg(short, long)]
        format: String,
    },
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
        
        Commands::TrainTokenizer { output, languages, max_samples } => {
            train_bpe_tokenizer(&output, &languages, max_samples).await?;
        },
        
        Commands::TestInference { engine, model_path, prompt } => {
            test_inference_engine(&engine, model_path.as_deref(), &prompt).await?;
        },
        
        Commands::ConvertModel { input, output, format } => {
            convert_model_format(&input, &output, &format).await?;
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

/// Train BPE tokenizer on bigcode dataset
async fn train_bpe_tokenizer(output_path: &str, languages: &[String], max_samples: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Training BPE tokenizer...");
    println!("Languages: {:?}", languages);
    println!("Max samples: {}", max_samples);
    println!("Output path: {}", output_path);
    
    // Create tokenizer config
    let config = BpeTokenizerConfig::default();
    let mut tokenizer = BpeTokenizer::new(config)?;
    
    // Create dataset config
    let dataset_config = BigcodeDatasetConfig {
        languages: languages.to_vec(),
        max_samples_per_language: max_samples / languages.len().max(1),
        max_total_samples: max_samples,
        ..Default::default()
    };
    
    // Train on bigcode dataset
    tokenizer.train_on_bigcode_dataset(&dataset_config).await?;
    
    // Save tokenizer
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokenizer.save(output_path)?;
    
    println!("BPE tokenizer training completed successfully!");
    println!("Tokenizer saved to: {}", output_path);
    
    Ok(())
}

/// Test different inference engines
async fn test_inference_engine(engine_type: &str, model_path: Option<&str>, prompt: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing inference engine: {}", engine_type);
    
    let engine_type = match engine_type.to_lowercase().as_str() {
        "llamacpp" | "llama" => InferenceEngineType::LlamaCpp,
        "onnx" => InferenceEngineType::Onnx,
        "native" => InferenceEngineType::Native,
        _ => {
            println!("Unsupported engine type: {}. Use: llamacpp, onnx, or native", engine_type);
            return Ok(());
        }
    };
    
    let config = InferenceConfig {
        engine_type,
        model_path: model_path.unwrap_or("").to_string(),
        ..Default::default()
    };
    
    println!("Engine config: {:?}", config);
    
    match &config.engine_type {
        InferenceEngineType::LlamaCpp | InferenceEngineType::Onnx => {
            if model_path.is_none() {
                println!("Model path required for {:?} engine", config.engine_type);
                return Ok(());
            }
            
            let mut engine = InferenceEngineFactory::create_engine(&config)?;
            
            if let Some(path) = model_path {
                println!("Loading model from: {}", path);
                match engine.load_model(path).await {
                    Ok(_) => println!("Model loaded successfully"),
                    Err(e) => {
                        println!("Failed to load model: {}", e);
                        println!("Note: This is expected in demo mode - actual model files not provided");
                        return Ok(());
                    }
                }
            }
            
            println!("Generating text with prompt: '{}'", prompt);
            match engine.generate_text(prompt, 50, 0.7).await {
                Ok(result) => {
                    println!("Generated text: {}", result);
                    
                    if let Ok(info) = engine.get_model_info() {
                        println!("Model info: {:?}", info);
                    }
                },
                Err(e) => println!("Generation failed: {}", e),
            }
        },
        InferenceEngineType::Native => {
            println!("Testing native engine (current implementation)");
            let config = ModelConfig::default();
            let model = AIModel::new(config).await?;
            
            let task = TaskType::TextGeneration { 
                max_tokens: 50, 
                temperature: 0.7 
            };
            
            match model.execute_task(task, prompt).await {
                Ok(result) => println!("Generated text: {}", result.text_output()),
                Err(e) => println!("Generation failed: {}", e),
            }
        }
    }
    
    Ok(())
}

/// Convert model between different formats
async fn convert_model_format(input_path: &str, output_path: &str, target_format: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting model format...");
    println!("Input: {}", input_path);
    println!("Output: {}", output_path);
    println!("Target format: {}", target_format);
    
    // Validate input file exists
    if !std::path::Path::new(input_path).exists() {
        return Err(format!("Input file does not exist: {}", input_path).into());
    }
    
    // Create output directory if needed
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    
    match target_format.to_lowercase().as_str() {
        "onnx" => {
            println!("Converting to ONNX format...");
            // In a real implementation, this would:
            // 1. Load the source model
            // 2. Convert to ONNX format using onnx crate
            // 3. Save the converted model
            
            // For demo purposes, create a placeholder ONNX model
            let placeholder_model = serde_json::json!({
                "format": "onnx",
                "version": "1.0",
                "converted_from": input_path,
                "conversion_time": chrono::Utc::now().to_rfc3339(),
                "note": "This is a placeholder - real implementation would convert actual model weights"
            });
            
            tokio::fs::write(output_path, serde_json::to_string_pretty(&placeholder_model)?).await?;
            println!("Model converted to ONNX format (placeholder)");
        },
        
        "gguf" | "ggml" => {
            println!("Converting to GGUF format (for llama.cpp)...");
            // In a real implementation, this would:
            // 1. Load the source model weights
            // 2. Quantize weights if specified  
            // 3. Save in GGUF format for llama.cpp
            
            // For demo purposes, create a placeholder GGUF model
            let placeholder_model = format!(
                "# GGUF Model File (Placeholder)\n# Converted from: {}\n# Format: GGUF\n# This is a demo placeholder file\n",
                input_path
            );
            
            tokio::fs::write(output_path, placeholder_model).await?;
            println!("Model converted to GGUF format (placeholder)");
        },
        
        _ => {
            return Err(format!("Unsupported target format: {}. Supported: onnx, gguf", target_format).into());
        }
    }
    
    println!("Model conversion completed successfully!");
    println!("Converted model saved to: {}", output_path);
    
    Ok(())
}
